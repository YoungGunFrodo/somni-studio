"""Core identity memories, immediate remembering, and chat history."""
import json
import sqlite3

from somni_studio.engine.providers import _RecallingProvider
from somni_studio.engine.recall import open_read_only_store, recording_orchestrator
from somni_studio.somni.bootstrap import set_model
from somni_studio.somni.home import HomePaths
from somni_studio.somni.identity import ensure_identity
from somni_studio.somni.store_reader import StoreReader

from .helpers import HomeTestCase


def _identity_rows(db):
    con = sqlite3.connect(db)
    try:
        return dict(con.execute("SELECT key, summary FROM memories WHERE status = 'active' AND key LIKE 'somni/%'"))
    finally:
        con.close()


class IdentityTest(HomeTestCase):
    def setUp(self):
        super().setUp()
        self.paths = HomePaths(self.home)

    def test_added_once(self):
        self.assertEqual(ensure_identity(self.paths), 7)
        self.assertEqual(ensure_identity(self.paths), 0, "running it again changes nothing")
        rows = _identity_rows(self.paths.memory_db)
        self.assertEqual(rows["somni/name"], "The assistant's name is Somni.")
        self.assertIn("test/missing-model", rows["somni/base-model"])

    def test_base_model_memory_follows_the_model(self):
        ensure_identity(self.paths)
        for d in self.home.glob("checkpoints/*/adapter"):           # no experts: the model may change
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
        set_model(self.paths, "org/another-model")
        self.assertIn("org/another-model", _identity_rows(self.paths.memory_db)["somni/base-model"])

    def test_a_users_own_version_is_kept(self):
        from sleepai.memory.store import Memory, MemoryStore
        store = MemoryStore(self.paths.memory_db)
        store.insert(Memory(key="somni/name", type="fact", summary="The assistant's name is Nova.", value="Nova",
                            strength="user-confirmed", sources=["user"]))
        store.db.close()
        ensure_identity(self.paths)
        self.assertEqual(_identity_rows(self.paths.memory_db)["somni/name"], "The assistant's name is Nova.")

    def test_identity_leads_every_recall(self):
        ensure_identity(self.paths)
        store = open_read_only_store(self.paths.memory_db)
        seen = []
        orch = recording_orchestrator(store, None, use_tags=False, on_recall=seen.append)
        context = orch.recall("completely unrelated question about trains")
        store.db.close()
        self.assertIn("somni/name", [m.key for m in context[:5]])
        self.assertEqual({i.source for i in seen[0][:5]}, {"identity"})


class EchoStudent:
    def answer(self, question, context=None):
        from sleepai.models.base import Answer
        return Answer("ok", "ok", 1.0)


class EchoProvider(_RecallingProvider):
    name = "echo"

    def _make_student(self):
        return EchoStudent()


class RememberTest(HomeTestCase):
    def test_remember_makes_a_memory_now(self):
        provider = EchoProvider(self.settings)
        provider.load()
        try:
            memory_id = provider._remember("The user's sister is called Ana.")
            self.assertEqual(provider._remember("The user's sister is called Ana."), memory_id, "no duplicates")
            memory = provider.store.get(memory_id)
            self.assertEqual(memory.summary, "The user's sister is called Ana.")
            self.assertEqual(memory.strength, "user-confirmed")
            self.assertTrue(memory.eligible, "sleep will train it in")
            self.assertEqual(provider.store.count_unformed(), 0, "formation will not read it again")
            found = [m.id for m in provider.store.search("what is my sister called")]
            self.assertIn(memory_id, found, "recall finds it straight away")
        finally:
            provider.close()


class ConversationTest(HomeTestCase):
    def test_history_from_the_episode_log(self):
        from sleepai.memory.store import MemoryStore
        store = MemoryStore(self.home / "memory.db")
        store.log_episode("conversation", [{"role": "user", "content": "hi"}, {"role": "memory", "content": "m1"},
                                           {"role": "assistant", "content": "hello"}])
        store.log_episode("conversation", [{"role": "user", "content": "Remember: x"}])     # no answer: skipped
        store.log_episode("conversation", [{"role": "user", "content": "and you?"},
                                           {"role": "assistant", "content": "fine"}])
        store.db.close()
        rows = StoreReader(self.home / "memory.db").conversations(limit=10)
        self.assertEqual([(r["question"], r["answer"]) for r in rows], [("hi", "hello"), ("and you?", "fine")])
        self.assertEqual(json.loads(json.dumps(rows))[0]["question"], "hi")
