import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from somni_studio.somni import sleep_runner
from somni_studio.somni.home import HomePaths
from somni_studio.somni.sleep_runner import SleepError, SleepRunner, SleepState

from .helpers import HomeTestCase

FAKE_CLI = '''
import sys, time


def main(argv):
    home = argv[argv.index("--home") + 1]
    print("forming memories from 3 episodes")
    print("[train] step 50/100 loss 0.5")
    print("training expert 1")
    if "--slow" in open(home + "/mode.txt").read():
        time.sleep(30)
    print("sleep 0: awaiting review, recommendation: adopt")
    return 0
'''


class SleepRunnerTest(unittest.TestCase):
    """Runs the real process plumbing against a stand-in `sleepai.cli`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo = root / "repo"
        (self.repo / "sleepai").mkdir(parents=True)
        (self.repo / "sleepai" / "__init__.py").write_text("", encoding="utf-8")
        (self.repo / "sleepai" / "cli.py").write_text(FAKE_CLI, encoding="utf-8")
        self.paths = HomePaths(root / "home")
        self.paths.home.mkdir()
        self.paths.config.write_text("{}", encoding="utf-8")
        self.events: list[tuple[str, dict]] = []
        self.done = threading.Event()

        def notify(kind, payload):
            self.events.append((kind, payload))
            if kind == "status" and payload["state"] in ("finished", "failed", "idle"):
                self.done.set()
        self.runner = SleepRunner(notify)

    def tearDown(self):
        if self.runner.running:
            self.runner.stop()
        self._tmp.cleanup()

    def _mode(self, text):
        (self.paths.home / "mode.txt").write_text(text, encoding="utf-8")

    def test_runs_and_streams_output(self):
        self._mode("fast")
        self.runner.start(self.repo, self.paths, python=sys.executable)
        self.assertTrue(self.done.wait(30))
        lines = [p["line"] for k, p in self.events if k == "log"]
        self.assertIn("forming memories from 3 episodes", lines)
        self.assertIn("training expert 1", lines)
        phases = [p for k, p in self.events if k == "phase"]
        # the stand-in Somni has no pipeline to narrate: the sleep still runs, minus the narration
        self.assertEqual(phases[0]["phase"], "setup")
        self.assertIn({"phase": "training", "state": "progress", "step": 50, "total": 100}, phases)
        status = self.runner.status(self.paths)
        self.assertEqual(status["state"], SleepState.FINISHED.value)
        self.assertEqual(status["summary"], "sleep 0: awaiting review, recommendation: adopt")

    def test_wake_and_stop(self):
        self._mode("--slow")
        self.runner.start(self.repo, self.paths, python=sys.executable)
        with self.assertRaises(SleepError):
            self.runner.start(self.repo, self.paths, python=sys.executable)
        deadline = time.time() + 10
        while not any(k == "log" for k, _ in self.events) and time.time() < deadline:
            time.sleep(0.05)
        self.runner.wake()
        self.assertTrue((self.paths.home / "WAKE").exists(), "Somni's own wake signal")
        self.assertEqual(self.runner.status()["state"], SleepState.WAKING.value)
        self.runner.stop()
        deadline = time.time() + 15
        while self.runner.running and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        status = self.runner.status()
        self.assertEqual(status["state"], SleepState.IDLE.value, "stopping on purpose is not a failure")
        self.assertIn("stopped", status["summary"])
        self.assertFalse((self.paths.home / "WAKE").exists(), "WAKE is cleared, or the next sleep would stop at once")

    def test_needs_a_somni_home(self):
        self.paths.config.unlink()
        with self.assertRaises(SleepError):
            self.runner.start(self.repo, self.paths, python=sys.executable)


class ReviewTest(HomeTestCase):
    """Morning review through Somni's real review code."""

    def setUp(self):
        super().setUp()
        self.paths = HomePaths(self.home)
        reg = json.loads(self.paths.registry.read_text(encoding="utf-8"))
        reg["candidates"]["ckpt-0002"] = {"created_at": 1.0, "sleep_no": 1, "memories": [], "passed": True,
                                          "status": "awaiting review"}
        reg["history"] = [{"id": "ckpt-0000", "adopted_at": 0, "note": "base"}]
        self.paths.registry.write_text(json.dumps(reg), encoding="utf-8")
        night = self.home / "sleeps" / "sleep-0001"
        night.mkdir(parents=True)
        (night / "report.json").write_text(json.dumps({
            "sleep_no": 1, "status": "awaiting review", "recommendation": "adopt", "candidate": "ckpt-0002",
            "formation": {"created": ["m1", "m2"]}, "consolidation": {"new_memories": ["m1"]},
            "evaluation": {"passed": True, "gates": [{"name": "learning", "passed": True, "current": None,
                                                      "candidate": 1.0, "note": ""}]}}), encoding="utf-8")

    def test_pending_and_report(self):
        self.assertEqual([c["checkpoint"] for c in sleep_runner.pending_review(self.paths)], ["ckpt-0002"])
        report = sleep_runner.last_report(self.paths)
        self.assertEqual((report["memories_formed"], report["learned"]), (2, 1))
        self.assertEqual(report["gates"][0]["name"], "learning")

    def test_approve_adopts_the_checkpoint(self):
        self.assertEqual(sleep_runner.approve(self.paths, "ckpt-0002"), "ckpt-0002")
        reg = json.loads(self.paths.registry.read_text(encoding="utf-8"))
        self.assertEqual(reg["active"], "ckpt-0002")
        self.assertEqual(sleep_runner.pending_review(self.paths), [])

    def test_reject(self):
        sleep_runner.reject(self.paths, "ckpt-0002")
        reg = json.loads(self.paths.registry.read_text(encoding="utf-8"))
        self.assertEqual(reg["candidates"]["ckpt-0002"]["status"], "rejected")
        self.assertEqual(reg["active"], "ckpt-0001")


if __name__ == "__main__":
    unittest.main()
