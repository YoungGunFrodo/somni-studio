"""Recall through Somni's real Orchestrator, read-only enforcement, and the provider ask flow."""
import sqlite3

from somni_studio.engine.providers import _RecallingProvider
from somni_studio.engine.recall import classify, open_read_only_store, recording_orchestrator
from somni_studio.events import EventType

from .helpers import HomeTestCase


class EchoStudent:
    """Answers with the first recalled memory's value and cites it, like Somni's students do."""

    def answer(self, question, context=None):
        from sleepai.models.base import Answer
        mem = next((m for m in context or [] if m.type != "file"), None)
        return Answer(mem.value if mem else "I'm not sure", "", 1.0, used_memory=mem.id if mem else "")


class EchoProvider(_RecallingProvider):
    name = "echo"

    def _make_student(self):
        return EchoStudent()


def _episodes(db_path):
    db = sqlite3.connect(db_path)
    try:
        return db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    finally:
        db.close()


class RecallTest(HomeTestCase):
    def test_recording_orchestrator_reports_sources(self):
        store = open_read_only_store(self.home / "memory.db")
        seen = []
        orch = recording_orchestrator(store, EchoStudent(), use_tags=True, on_recall=seen.append)
        context = orch.recall("Who gave their name to Normandy?")
        self.assertTrue(context)
        items = seen[0]
        self.assertEqual(items, orch.last_recall)
        self.assertIn("mem:", items[0].node_id)
        self.assertTrue(any(i.source == "file" and i.node_id == "file:1" for i in items))
        self.assertEqual([i.rank for i in items], list(range(len(items))))
        store.db.close()

    def test_read_only_store_refuses_writes(self):
        store = open_read_only_store(self.home / "memory.db")
        with self.assertRaises(sqlite3.OperationalError):
            store.log_episode("conversation", [{"role": "user", "content": "hi"}])
        store.db.close()

    def test_classify(self):
        class M:
            def __init__(self, id, type="fact", key="k"):
                self.id, self.type, self.key = id, type, key
        items = classify([M("m1"), M("m2"), M("file:7", "file", "file/7")], from_tags={"m2"})
        self.assertEqual([(i.node_id, i.source) for i in items],
                         [("mem:m1", "search"), ("mem:m2", "tags"), ("file:7", "file")])

    def test_read_only_provider_answers_without_logging(self):
        db = self.home / "memory.db"
        before = _episodes(db)
        provider = EchoProvider(self.settings.merged({"somni": {"read_only": True}}))
        provider.load()
        events = []
        outcome = provider.ask("Who gave their name to Normandy?", lambda kind, **p: events.append((kind, p)))
        provider.close()
        self.assertEqual(outcome.text, "Normandy")
        self.assertTrue(outcome.used_node.startswith("mem:"))
        self.assertIsNone(outcome.episode_id)
        self.assertEqual(events[0][0], EventType.RECALL)
        self.assertEqual(_episodes(db), before)

    def test_writable_provider_logs_an_episode_like_somni_chat(self):
        db = self.home / "memory.db"
        before = _episodes(db)
        provider = EchoProvider(self.settings.merged({"somni": {"read_only": False}}))
        provider.load()
        outcome = provider.ask("Who gave their name to Normandy?", lambda *a, **k: None)
        provider.close()
        self.assertIsNotNone(outcome.episode_id)
        self.assertEqual(_episodes(db), before + 1)


class CitedNodeTest(HomeTestCase):
    def test_memory_and_file_citations(self):
        from somni_studio.engine.providers import cited_node
        self.assertEqual(cited_node("m000001-x"), "mem:m000001-x")
        self.assertEqual(cited_node("file:111"), "file:111")
        self.assertIsNone(cited_node(""))

    def test_file_items_are_named_by_title(self):
        store = open_read_only_store(self.home / "memory.db")
        seen = []
        orch = recording_orchestrator(store, EchoStudent(), use_tags=False, on_recall=seen.append)
        orch.recall("Who gave their name to Normandy?")
        files = [i for i in seen[0] if i.source == "file"]
        self.assertTrue(files)
        self.assertEqual(files[0].label, "Normans")
        store.db.close()
