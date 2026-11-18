"""Switching models keeps every memory and makes the next sleep relearn them."""
import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from somni_studio.settings import STUDIO_ROOT
from somni_studio.somni.bootstrap import ModelLocked, model_locked_by, pending_transfer, set_model
from somni_studio.somni.home import HomePaths, base_model

from .helpers import HomeTestCase


class SwitchTest(HomeTestCase):
    def setUp(self):
        super().setUp()
        self.paths = HomePaths(self.home)
        db = sqlite3.connect(self.paths.memory_db)
        db.execute("UPDATE memories SET consolidated_in = '[\"ckpt-0001\"]'")
        db.commit()
        db.close()
        (self.home / "rehearsal-general-abc123.json").write_text("[]", encoding="utf-8")

    def test_needs_relearn_when_experts_exist(self):
        self.assertIsNotNone(model_locked_by(self.paths))
        with self.assertRaises(ModelLocked):
            set_model(self.paths, "org/new-model")
        self.assertEqual(base_model(self.paths), "test/missing-model", "nothing changed")

    def test_relearn_archives_and_marks(self):
        db = sqlite3.connect(self.paths.memory_db)
        before = {r[0] for r in db.execute("SELECT key FROM memories")}
        db.close()
        result = set_model(self.paths, "org/new-model", relearn=True)
        self.assertTrue(result["relearn"])
        self.assertEqual(base_model(self.paths), "org/new-model")
        archive = Path(result["archived"])
        self.assertTrue((archive / "checkpoints" / "registry.json").exists(), "old experts are archived, not deleted")
        self.assertTrue((archive / "rehearsal-general-abc123.json").exists(), "the old model's own answers too")
        self.assertFalse((self.home / "checkpoints").exists())
        self.assertIsNone(model_locked_by(self.paths), "no experts left for the new model yet")
        self.assertEqual(pending_transfer(self.paths)["to"], "org/new-model")
        db = sqlite3.connect(self.paths.memory_db)
        learned = db.execute("SELECT COUNT(*) FROM memories WHERE consolidated_in != '[]'").fetchone()[0]
        after = {r[0] for r in db.execute("SELECT key FROM memories")}
        db.close()
        self.assertEqual(learned, 0, "every memory is waiting to be learned again")
        self.assertLessEqual(before, after, "and every memory is still there")

    def test_same_model_is_a_no_op(self):
        self.assertEqual(set_model(self.paths, "test/missing-model"), {"changed": False})


FAKE = {
    "sleepai/__init__.py": "",
    "sleepai/cli.py": textwrap.dedent('''
        def _system(args):
            return "system for " + args.home

        def main(argv):
            print("normal sleep", flush=True)
            return 0
    '''),
    "sleepai/sleep/__init__.py": "",
    "sleepai/sleep/pipeline.py": textwrap.dedent('''
        def run_sleep(system, day=0, force_rebuild=False, full_store=False):
            print(f"run_sleep force_rebuild={force_rebuild} full_store={full_store}", flush=True)
            return {"sleep_no": 0, "status": "awaiting review", "recommendation": "adopt"}
    '''),
    "sleepai/sleep/scheduler.py": "def today():\n    return 1\n",
}


class RelearnSleepTest(unittest.TestCase):
    def test_marker_turns_the_next_sleep_into_a_relearn(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel, text in FAKE.items():
                (root / "repo" / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / "repo" / rel).write_text(text, encoding="utf-8")
            home = root / "home"
            home.mkdir()
            (home / "TRANSFER").write_text(json.dumps({"to": "org/new-model"}), encoding="utf-8")
            env_path = f"{root / 'repo'};{STUDIO_ROOT}" if sys.platform == "win32" else f"{root / 'repo'}:{STUDIO_ROOT}"
            out = subprocess.run([sys.executable, "-m", "somni_studio.somni.sleep_entry", "--home", str(home)],
                                 capture_output=True, text=True, timeout=60, env={"PYTHONPATH": env_path,
                                                                                 "SYSTEMROOT": "C:\\Windows"})
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("run_sleep force_rebuild=True full_store=True", out.stdout)
            self.assertNotIn("normal sleep", out.stdout)
            self.assertFalse((home / "TRANSFER").exists(), "done: the next sleep is a normal one")


if __name__ == "__main__":
    unittest.main()
