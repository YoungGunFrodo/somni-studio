import asyncio
import os
import unittest

from somni_studio.events import EventBus
from somni_studio.gpu_guard import BUSY_SAMPLES, GpuGuard, GuardHooks
from somni_studio.gpu_watch import GpuPolicy, GpuUser
from somni_studio.settings import GpuSettings

GAME = GpuUser(4242, "game.exe", 88.0)
BROWSER = GpuUser(5151, "chrome.exe", 12.0)


class FakeMonitor:
    available = True

    def sample(self):
        return []

    def close(self):
        pass


class PolicyTest(unittest.TestCase):
    def test_sustained_use_pauses_and_quiet_resumes(self):
        policy = GpuPolicy(40, 3, free_seconds=10, ignore=set(), own_pids={1})
        self.assertFalse(policy.update([GAME], now=0).busy)
        self.assertFalse(policy.update([GAME], now=3).busy, "a blip is not enough")
        self.assertTrue(policy.update([GAME], now=6).busy)
        self.assertTrue(policy.update([], now=9).busy, "stays paused until quiet long enough")
        self.assertTrue(policy.update([BROWSER], now=15).busy)
        self.assertFalse(policy.update([], now=20).busy)

    def test_own_and_ignored_programs_never_count(self):
        policy = GpuPolicy(40, 1, 10, ignore={"game.exe"}, own_pids={os.getpid()})
        self.assertFalse(policy.update([GAME, GpuUser(os.getpid(), "python.exe", 100.0)], now=0).busy)
        self.assertFalse(policy.update([GpuUser(1, "dwm.exe", 90.0)], now=1).busy, "the desktop compositor")


class Recorder:
    def __init__(self, loaded=True, sleeping=False):
        self.calls, self.loaded, self.is_sleeping = [], loaded, sleeping

    def hooks(self):
        async def unload():
            self.calls.append("unload")
            self.loaded = False

        async def stop_sleep(reason):
            self.calls.append(("stop_sleep", reason))
            self.is_sleeping = False

        async def start_sleep():
            self.calls.append("start_sleep")

        return GuardHooks(model_loaded=lambda: self.loaded, unload_model=unload,
                          load_model=lambda: self.calls.append("load"), sleeping=lambda: self.is_sleeping,
                          stop_sleep=stop_sleep, start_sleep=start_sleep)


class GuardTest(unittest.TestCase):
    def run_ticks(self, guard, frames):
        async def go():
            bus = EventBus()
            bus.attach(asyncio.get_running_loop())
            guard._bus = bus
            for users in frames:
                await guard.tick(users)
        asyncio.run(go())

    def test_pauses_model_and_sleep_then_brings_both_back(self):
        rec = Recorder(loaded=True, sleeping=True)
        guard = GpuGuard(EventBus(), GpuSettings(resume_after_s=5), rec.hooks(), monitor=FakeMonitor())
        self.run_ticks(guard, [[GAME]] * BUSY_SAMPLES)
        self.assertTrue(guard.paused)
        self.assertEqual(guard.status()["by"]["name"], "game.exe")
        self.assertEqual(rec.calls[0][0], "stop_sleep")
        self.assertIn("game.exe", rec.calls[0][1])
        self.assertEqual(rec.calls[1], "unload")
        self.assertIn("game.exe", guard.pause_message())
        guard._policy._free_since = -100.0                     # pretend the game closed long ago
        self.run_ticks(guard, [[]])
        self.assertFalse(guard.paused)
        self.assertEqual(rec.calls[2:], ["load", "start_sleep"])

    def test_brings_back_only_what_it_paused(self):
        rec = Recorder(loaded=False, sleeping=False)
        guard = GpuGuard(EventBus(), GpuSettings(), rec.hooks(), monitor=FakeMonitor())
        self.run_ticks(guard, [[GAME]] * BUSY_SAMPLES)
        asyncio.run(guard.resume("test"))
        self.assertEqual(rec.calls, [], "nothing was running, so nothing is started")

    def test_switched_off(self):
        rec = Recorder()
        guard = GpuGuard(EventBus(), GpuSettings(auto_pause=False), rec.hooks(), monitor=FakeMonitor())
        self.run_ticks(guard, [[GAME]] * (BUSY_SAMPLES + 2))
        self.assertFalse(guard.paused)
        self.assertEqual(rec.calls, [])


class ApiPauseTest(unittest.TestCase):
    """While paused, the API refuses to load the model or answer with it."""

    def test_chat_is_refused_while_paused(self):
        from fastapi.testclient import TestClient

        from somni_studio.api.app import create_app
        from somni_studio.settings import StudioSettings

        from .fakes import factory
        from .helpers import HAVE_SOMNI
        if not HAVE_SOMNI:
            self.skipTest("needs Somni")
        import tempfile
        from pathlib import Path

        from .helpers import SOMNI_REPO, make_home
        with tempfile.TemporaryDirectory() as d:
            home = make_home(Path(d))
            settings = StudioSettings().merged({"somni": {"repo": str(SOMNI_REPO), "home": str(home)},
                                                "voice": {"enabled": False}, "listen": {"enabled": False}})
            with TestClient(create_app(settings, Path(d) / "s.json", provider_factory=factory(),
                                       data_dir=Path(d) / "data", gpu_monitor=FakeMonitor())) as client:
                studio = client.app.state.studio
                client.portal.call(studio.gpu._pause, GAME)
                r = client.post("/api/chat", json={"question": "hi"})
                self.assertEqual(r.status_code, 409)
                self.assertIn("game.exe", r.json()["detail"])
                self.assertEqual(client.get("/api/gpu").json()["by"]["name"], "game.exe")
                ok = client.post("/api/gpu/resume", json={"ignore_program": True})
                self.assertFalse(ok.json()["paused"])
                self.assertIn("game.exe", client.get("/api/settings").json()["settings"]["gpu"]["ignore"])
                self.assertEqual(client.post("/api/chat", json={"question": "hi"}).status_code, 202)


if __name__ == "__main__":
    unittest.main()
