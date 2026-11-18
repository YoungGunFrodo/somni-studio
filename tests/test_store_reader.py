import sqlite3
import tempfile
import unittest
from pathlib import Path

from somni_studio.somni.store_reader import ORDINAL_DAY_MIN, StoreReader, somni_today

from .helpers import HomeTestCase


class StoreReaderTest(HomeTestCase):
    def setUp(self):
        super().setUp()
        self.reader = StoreReader(self.home / "memory.db")

    def test_snapshot(self):
        snap = self.reader.snapshot(max_memories=100)
        self.assertEqual(len(snap.memories), 5)
        self.assertEqual([f.title for f in snap.files], ["Normans", "Nikola Tesla"])
        self.assertEqual(len(snap.source_of), 5)
        self.assertEqual(snap.today, 4)                 # benchmark-style day numbers: latest day
        self.assertFalse(snap.truncated)
        self.assertEqual(snap.totals["memories"], 5)

    def test_snapshot_limit_keeps_most_used(self):
        db = sqlite3.connect(self.home / "memory.db")
        db.execute("UPDATE memories SET retrieval_count = 9 WHERE key = 'nikola-tesla/born'")
        db.commit()
        db.close()
        snap = self.reader.snapshot(max_memories=1)
        self.assertTrue(snap.truncated)
        self.assertEqual([m.key for m in snap.memories], ["nikola-tesla/born"])

    def test_memory_detail_and_related(self):
        mid = next(m.id for m in self.reader.snapshot(100).memories if m.key == "normans/named")
        detail = self.reader.memory(mid)
        self.assertEqual(detail["value"], "Normandy")
        self.assertEqual(detail["tags"], ["history", "normans", "normandy"])
        self.assertTrue(detail["has_file"])
        related = self.reader.related(mid, detail["tags"])
        self.assertEqual([r["key"] for r in related], ["normans/origin"])     # shares history + normans
        self.assertIsNone(self.reader.memory("nope"))

    def test_file_and_tag(self):
        f = self.reader.file(1)
        self.assertEqual(f["title"], "Normans")
        self.assertIn("Normandy", f["text"])
        self.assertEqual(len(f["memories"]), 2)
        self.assertIsNone(self.reader.file(99))
        tag = self.reader.tag("history")
        self.assertEqual(tag["count"], 2)
        self.assertIsNone(self.reader.tag("unknown-tag"))

    def test_reader_cannot_write(self):
        with self.reader._connect() as db, self.assertRaises(sqlite3.OperationalError):
            db.execute("DELETE FROM memories")


class MissingStoreTest(unittest.TestCase):
    def test_missing_db_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            reader = StoreReader(Path(d) / "memory.db")
            self.assertEqual(reader.snapshot(10).memories, [])
            self.assertIsNone(reader.memory("x"))
            self.assertFalse((Path(d) / "memory.db").exists(), "reading must not create a store")

    def test_today(self):
        self.assertEqual(somni_today(12), 12)
        self.assertGreater(somni_today(ORDINAL_DAY_MIN + 1), ORDINAL_DAY_MIN)


if __name__ == "__main__":
    unittest.main()
