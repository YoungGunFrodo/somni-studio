"""Test fixtures: a small but real Somni home, written with Somni's own MemoryStore so the
studio's read-only SQL is tested against the actual schema."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from somni_studio.settings import StudioSettings, load_settings, resolve_path, settings_path


def _somni_repo() -> Path:
    """SOMNI_REPO, else the repo in this install's studio.json, else the default (../somni)."""
    if os.environ.get("SOMNI_REPO"):
        return resolve_path(os.environ["SOMNI_REPO"])
    try:
        return load_settings(settings_path()).somni.repo_path
    except Exception:          # a broken studio.json must not hide the tests
        return StudioSettings().somni.repo_path


SOMNI_REPO = _somni_repo()
HAVE_SOMNI = (SOMNI_REPO / "sleepai" / "__init__.py").exists()

DOC_NORMANS = ("[document: Normans] The Normans were the people who gave their name to Normandy, a region in "
               "France. They descended from Norse raiders and pirates from Denmark, Iceland and Norway. "
               "Their identity emerged in the first half of the 10th century and evolved over centuries.")
DOC_TESLA = ("[document: Nikola Tesla] Nikola Tesla was a Serbian-American inventor, electrical engineer and "
             "futurist best known for his contributions to the design of the modern alternating current "
             "electricity supply system. He was born in 1856 in Smiljan.")


def require_somni(test: unittest.TestCase) -> None:
    if not HAVE_SOMNI:
        test.skipTest(f"Somni checkout not found at {SOMNI_REPO} (set SOMNI_REPO)")


def _ensure_path() -> None:
    from somni_studio.somni.runtime import ensure_sleepai
    ensure_sleepai(SOMNI_REPO)


def make_home(root: Path, *, untagged: bool = True, experts: int = 2) -> Path:
    """A Somni home with two documents, five memories (four tagged, one untagged), a registry
    and an expert folder."""
    _ensure_path()
    from sleepai.memory.store import Memory, MemoryStore

    home = root / "home"
    store = MemoryStore(home / "memory.db")
    e1 = store.log_episode("tool_call", [{"role": "tool", "content": DOC_NORMANS}], outcome="success", day=3)
    e2 = store.log_episode("tool_call", [{"role": "tool", "content": DOC_TESLA}], outcome="success", day=4)

    def add(key, summary, value, tags, ep):
        return store.insert(Memory(key=key, type="fact", summary=summary, value=value, strength="verified",
                                   sources=[f"episode:{ep}", "verified"], tags=list(tags), created_day=3)).id

    ids1 = [add("normans/named", "The Normans gave their name to Normandy.", "Normandy",
                ("history", "normans", "normandy"), e1),
            add("normans/origin", "The Normans descended from Norse raiders.", "Norse raiders",
                ("history", "normans", "origin"), e1)]
    ids2 = [add("nikola-tesla/born", "Nikola Tesla was born in 1856.", "1856",
                ("science", "nikola-tesla", "birth"), e2),
            add("nikola-tesla/known-for", "Tesla is known for alternating current.", "alternating current",
                ("science", "nikola-tesla", "ac"), e2)]
    if untagged:
        ids2.append(add("tesla-coil/inventor", "The Tesla coil was invented by Tesla.", "Tesla", (), e2))
    store.mark_formed(e1, ids1)
    store.mark_formed(e2, ids2)
    store.db.close()
    # A model whose config.json is not on this machine: the brain falls back to a guessed shape.
    (home / "config.json").write_text(json.dumps({"backend": "llamacpp", "trainer": "experts",
                                                  "base_model": "test/missing-model"}), encoding="utf-8")

    ckpt = home / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "registry.json").write_text(json.dumps({"base": "ckpt-0000", "active": "ckpt-0001", "history": [],
                                                    "candidates": {}, "next": 2}), encoding="utf-8")
    adapter = ckpt / "ckpt-0001" / "adapter"
    adapter.mkdir(parents=True)
    route = [None] * (experts - 1) + [0.6] if experts else []
    (adapter / "experts.json").write_text(json.dumps({
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "experts": experts, "rank": 16, "route": route,
        "route_calibration": {"memory": [1.2, 1.4, 1.6], "rehearsal": [0.1, 0.2, 0.3],
                              "train_memory_on": 1.0, "train_rehearsal_on": 0.0}}), encoding="utf-8")
    return home


class HomeTestCase(unittest.TestCase):
    """Gives each test a fresh home and matching settings."""

    def setUp(self) -> None:
        require_somni(self)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = make_home(self.root)
        self.settings = StudioSettings().merged({"somni": {"repo": str(SOMNI_REPO), "home": str(self.home)},
                                                 "voice": {"enabled": False}})

    def tearDown(self) -> None:
        self._tmp.cleanup()


__all__ = ["HAVE_SOMNI", "HomeTestCase", "SOMNI_REPO", "make_home", "require_somni"]
