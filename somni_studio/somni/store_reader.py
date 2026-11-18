"""Read-only access to a Somni memory store (memory.db).

Opened with SQLite `mode=ro` on its own short-lived connection per call, so it is safe from any
thread, never takes a write lock, and cannot change the store even by accident (the Somni
MemoryStore constructor runs schema migrations, which is why the studio does not use it here).
The SQL follows the schema in sleepai/memory/store.py.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

VISIBLE_STATUSES = ("active", "disputed")      # superseded memories are the archive: not drawn
ORDINAL_DAY_MIN = 700_000                      # Somni day numbers above this are date ordinals


def somni_today(max_store_day: int) -> int:
    """The day number salience is computed against. Real homes count days as date ordinals
    (sleepai.sleep.scheduler.today); simulated and benchmark homes count from 0, and for those
    the store's own latest day is "today"."""
    if max_store_day >= ORDINAL_DAY_MIN:
        return dt.date.today().toordinal()
    return max_store_day


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    key: str
    type: str
    summary: str
    value: str
    strength: str
    status: str
    tags: tuple[str, ...]
    retrieval_count: int
    last_used_day: int
    created_day: int
    supersedes: str
    consolidated: bool


@dataclass(frozen=True)
class FileRecord:
    episode_id: int
    title: str
    chars: int


@dataclass
class StoreSnapshot:
    memories: list[MemoryRecord] = field(default_factory=list)
    files: list[FileRecord] = field(default_factory=list)
    source_of: dict[str, int] = field(default_factory=dict)     # memory id -> episode id
    today: int = 0
    totals: dict[str, int] = field(default_factory=dict)
    truncated: bool = False


def _json_list(raw) -> list:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _record(r: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=r["id"], key=r["key"], type=r["type"], summary=r["summary"], value=r["value"] or "",
        strength=r["strength"], status=r["status"], tags=tuple(str(t) for t in _json_list(r["tags"]) if t),
        retrieval_count=int(r["retrieval_count"] or 0), last_used_day=int(r["last_used_day"]),
        created_day=int(r["created_day"]), supersedes=r["supersedes"] or "",
        consolidated=bool(_json_list(r["consolidated_in"])))


class StoreReader:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @property
    def exists(self) -> bool:
        return self.db_path.is_file()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        db = sqlite3.connect(uri, uri=True, timeout=5.0)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _has_table(db: sqlite3.Connection, name: str) -> bool:
        """Stores made before a feature existed lack its table (tags came in a later Somni version)."""
        return db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None

    @staticmethod
    def _columns(db: sqlite3.Connection, table: str) -> set[str]:
        return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}

    def snapshot(self, max_memories: int) -> StoreSnapshot:
        if not self.exists:
            return StoreSnapshot()
        with self._connect() as db:
            cols = self._columns(db, "memories")
            tags = "tags" if "tags" in cols else "'[]' AS tags"
            last_used = "last_used_day" if "last_used_day" in cols else "-1 AS last_used_day"
            marks = ",".join("?" * len(VISIBLE_STATUSES))
            total = db.execute(f"SELECT COUNT(*) FROM memories WHERE status IN ({marks})",
                               VISIBLE_STATUSES).fetchone()[0]
            rows = db.execute(
                f"SELECT id, key, type, summary, value, strength, status, {tags}, retrieval_count, "
                f"{last_used}, created_day, supersedes, consolidated_in FROM memories "
                f"WHERE status IN ({marks}) ORDER BY retrieval_count DESC, created_at DESC LIMIT ?",
                (*VISIBLE_STATUSES, max_memories)).fetchall()
            memories = sorted((_record(r) for r in rows), key=lambda m: m.id)
            files = [FileRecord(int(r[0]), r[1], int(r[2])) for r in db.execute(
                "SELECT episode_id, title, length(text) FROM files_fts ORDER BY CAST(episode_id AS INTEGER)")]
            source_of: dict[str, int] = {}
            for eid, formed in db.execute("SELECT id, formed_memories FROM episodes WHERE formed = 1"):
                for mid in _json_list(formed):
                    source_of.setdefault(str(mid), int(eid))
            max_day = max(db.execute("SELECT COALESCE(MAX(day), 0) FROM episodes").fetchone()[0],
                          db.execute("SELECT COALESCE(MAX(created_day), 0) FROM memories").fetchone()[0])
            totals = {
                "episodes": db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "memories": total,
                "disputed": db.execute("SELECT COUNT(*) FROM memories WHERE status = 'disputed'").fetchone()[0],
                "superseded": db.execute("SELECT COUNT(*) FROM memories WHERE status = 'superseded'").fetchone()[0],
                "files": len(files),
                "open_questions": db.execute("SELECT COUNT(*) FROM open_questions WHERE status = 'open'").fetchone()[0],
            }
        return StoreSnapshot(memories, files, source_of, somni_today(int(max_day)), totals,
                             truncated=total > len(memories))

    def memory(self, memory_id: str) -> dict | None:
        if not self.exists:
            return None
        with self._connect() as db:
            r = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if r is None:
                return None
            out = dict(r)
            for key in ("sources", "consolidated_in", "tags"):
                if key in out:
                    out[key] = _json_list(out[key])
            ep = db.execute("SELECT id FROM episodes WHERE formed_memories LIKE ? ORDER BY id LIMIT 1",
                            (f'%"{memory_id}"%',)).fetchone()
            out["episode_id"] = int(ep[0]) if ep else None
            out["has_file"] = bool(ep) and db.execute(
                "SELECT 1 FROM files_fts WHERE episode_id = ?", (ep[0],)).fetchone() is not None
        return out

    def related(self, memory_id: str, tags: list[str], min_shared: int = 2, k: int = 8) -> list[dict]:
        """Memories sharing at least `min_shared` tags (Somni's link rule, MemoryStore.related)."""
        tags = [t for t in dict.fromkeys(tags) if t]
        if len(tags) < min_shared or not self.exists:
            return []
        with self._connect() as db:
            if not self._has_table(db, "memory_tags"):
                return []
            rows = db.execute(
                "SELECT m.id, m.key, m.summary, COUNT(*) shared "
                "FROM memory_tags t JOIN memories m ON m.id = t.memory_id "
                f"WHERE t.tag IN ({','.join('?' * len(tags))}) AND m.status = 'active' AND m.id != ? "
                "GROUP BY m.id HAVING shared >= ? ORDER BY shared DESC, m.retrieval_count DESC LIMIT ?",
                (*tags, memory_id, min_shared, k)).fetchall()
        return [dict(r) for r in rows]

    def file(self, episode_id: int) -> dict | None:
        if not self.exists:
            return None
        with self._connect() as db:
            r = db.execute("SELECT episode_id, title, text FROM files_fts WHERE episode_id = ?",
                           (episode_id,)).fetchone()
            if r is None:
                return None
            ep = db.execute("SELECT day, kind, formed_memories FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return {"episode_id": int(r["episode_id"]), "title": r["title"], "text": r["text"],
                "day": ep["day"] if ep else None, "kind": ep["kind"] if ep else None,
                "memories": _json_list(ep["formed_memories"]) if ep else []}

    def conversations(self, limit: int = 40) -> list[dict]:
        """The latest chat exchanges (oldest first): what was asked and answered, from the episode log."""
        if not self.exists:
            return []
        with self._connect() as db:
            rows = db.execute("SELECT id, created_at, transcript FROM episodes WHERE kind = 'conversation' "
                              "ORDER BY id DESC LIMIT ?", (limit * 2,)).fetchall()
        out = []
        for r in reversed(rows):                 # oldest first; one episode is a whole conversation
            asked = None
            for t in _json_list(r["transcript"]):
                role, content = t.get("role"), t.get("content", "")
                if role == "user":
                    asked = content
                elif role == "assistant" and asked:
                    out.append({"episode_id": int(r["id"]), "at": r["created_at"], "question": asked,
                                "answer": content})
                    asked = None
        return out[-limit:]

    def tag(self, tag: str, k: int = 60) -> dict | None:
        if not self.exists:
            return None
        with self._connect() as db:
            if not self._has_table(db, "memory_tags"):
                return None
            count = db.execute("SELECT COUNT(*) FROM memory_tags t JOIN memories m ON m.id = t.memory_id "
                               "WHERE t.tag = ? AND m.status = 'active'", (tag,)).fetchone()[0]
            if not count:
                return None
            rows = db.execute("SELECT m.id, m.key, m.summary FROM memory_tags t JOIN memories m ON m.id = t.memory_id "
                              "WHERE t.tag = ? AND m.status = 'active' ORDER BY m.retrieval_count DESC, m.key LIMIT ?",
                              (tag, k)).fetchall()
        return {"tag": tag, "count": count, "memories": [dict(r) for r in rows]}
