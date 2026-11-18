"""Somni's own recall, instrumented.

The studio answers through Somni's Orchestrator, so chat retrieves exactly what wake retrieves.
The subclass only records where each recalled item came from (keyword search, subject tags, or a
memory file) and reports it the moment recall finishes, before the model starts answering.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from ..graph.memory_graph import file_id, memory_id
from .types import RecallItem

RecallCallback = Callable[[list[RecallItem]], None]


def open_read_only_store(db_path: Path):
    """A Somni MemoryStore on a read-only SQLite connection. The normal constructor runs schema
    migrations (writes); this one cannot write at all, so read-only mode is enforced by SQLite,
    not by care. Every read method of MemoryStore works unchanged (they only use `self.db`)."""
    from sleepai.memory.store import MemoryStore
    store = MemoryStore.__new__(MemoryStore)
    store.path = str(db_path)
    store.db = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True,
                               check_same_thread=True, timeout=5.0)
    store.db.row_factory = sqlite3.Row
    return store


IDENTITY_TAG = "identity"
MAX_IDENTITY = 8
PROFILE_TAG = "about-user"           # facts about the user (name, family, routines): in every recall
PROFILE_KEYS = ("user/%", "the-user/%")   # ... and what sleep's formation files under the user
MAX_PROFILE = 8


def profile_memories(store, k: int = MAX_PROFILE) -> list:
    """What the assistant knows about the user, newest first. A keyword search cannot be trusted
    to find these ("what's my name?" shares no word with "The user wants to be called Sir.")."""
    rows = store.db.execute(
        "SELECT m.* FROM memories m WHERE m.status = 'active' AND (m.key LIKE ? OR m.key LIKE ? OR m.id IN "
        "(SELECT memory_id FROM memory_tags WHERE tag = ?)) ORDER BY m.created_at DESC LIMIT ?",
        (*PROFILE_KEYS, PROFILE_TAG, k)).fetchall()
    return [store._row_to_memory(r) for r in rows]


def is_profile(m) -> bool:
    return PROFILE_TAG in (getattr(m, "tags", None) or []) or str(getattr(m, "key", "")).startswith(("user/",
                                                                                                    "the-user/"))


def classify(context: list, from_tags: set[str], file_title=None) -> list[RecallItem]:
    """Recall items for the UI. `file_title(episode_id)` names memory files (else their key)."""
    out = []
    for rank, m in enumerate(context or []):
        if IDENTITY_TAG in (getattr(m, "tags", None) or []):
            out.append(RecallItem(memory_id(m.id), "identity", rank, m.key))
        elif is_profile(m):
            out.append(RecallItem(memory_id(m.id), "profile", rank, m.key))
        elif getattr(m, "type", "") == "file" and str(m.id).startswith("file:"):
            eid = int(m.id.split(":", 1)[1])
            out.append(RecallItem(file_id(eid), "file", rank, (file_title(eid) if file_title else None) or m.key))
        else:
            out.append(RecallItem(memory_id(m.id), "tags" if m.id in from_tags else "search", rank, m.key))
    return out


def recording_orchestrator(store, student, use_tags: bool, on_recall: RecallCallback):
    from sleepai.wake.orchestrator import Orchestrator

    class RecordingOrchestrator(Orchestrator):
        def recall(self, question: str, k: int = 5, files: int = 2) -> list:
            reused = getattr(self, "_reuse", None)
            self._reuse = None
            if reused is not None and reused[0] == question:
                return reused[1]                     # already recalled (and reported) for this question
            self._from_tags: set[str] = set()
            context = self._with_identity(super().recall(question, k=k, files=files))
            self.last_recall = classify(context, self._from_tags, self._file_title)
            on_recall(self.last_recall)
            return context

        def _with_identity(self, context: list) -> list:
            """Core identity memories (somni/identity.py) and what is known about the user lead
            every context: neither should depend on a keyword search hitting them."""
            if not hasattr(self.store, "by_tags"):
                return context
            seen = {m.id for m in context}
            core = []
            for m in self.store.by_tags([IDENTITY_TAG], k=MAX_IDENTITY) + profile_memories(self.store):
                if m.id not in seen:
                    seen.add(m.id)
                    core.append(m)
            return core + context

        def _file_title(self, episode_id: int) -> str | None:
            found = self.store.file_text(episode_id) if hasattr(self.store, "file_text") else None
            return found[0] if found else None

        def _by_tags(self, question, merged, seen, extra: int = 3):
            found = super()._by_tags(question, merged, seen, extra)
            self._from_tags.update(m.id for m in found)
            return found

    orchestrator = RecordingOrchestrator(store, student, use_memory=True)
    orchestrator.use_tags = use_tags
    orchestrator.last_recall = []
    orchestrator._reuse = None
    return orchestrator


def reuse_recall(orchestrator, question: str, context: list) -> None:
    """The next recall of `question` returns `context` instead of searching (and lighting) again."""
    orchestrator._reuse = (question, context)
