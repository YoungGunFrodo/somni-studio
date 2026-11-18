"""Playing music, honest answers when actions fail, and sleeps that cannot outlive the studio."""
import json
import subprocess
import sys
import time
import unittest
from unittest import mock

import psutil

from somni_studio.devices import radio
from somni_studio.devices.base import Device, DeviceError
from somni_studio.engine.agent import Step, ToolAgent
from somni_studio.somni.sleep_runner import end_tree
from somni_studio.tools.catalog import CallRecord, ToolContext, run_tool


class FakeHub:
    links = [{"id": "cast:1", "driver": "cast", "address": "192.168.1.9", "name": "Living room display"}]

    def __init__(self, fail=False):
        self.played, self.fail = [], fail

    def devices(self, max_age=0):
        return [Device("cast:1", "Living room display", "speaker", False, link="cast:1")]

    def find(self, name):
        return [d for d in self.devices() if all(w in d.name.lower() for w in name.lower().split())]

    def play(self, device, url, title, content_type, live=False):
        if self.fail:
            raise DeviceError("Living room display is not answering")
        self.played.append((device.name, url, title, content_type, live))


def ctx(question, hub):
    return ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: False,
                       allowed_domains=frozenset({"media_player"}), question=question, devices=hub)


STATION = {"name": "Classic FM", "url": "https://stream.example/classic.mp3", "content_type": "audio/mpeg",
           "country": "UK"}


class PlayMusicTest(unittest.TestCase):
    def test_plays_a_station_on_the_speaker(self):
        hub = FakeHub()
        with mock.patch.object(radio, "find_station", return_value=STATION) as found:
            record = run_tool("play_music", {"device": "living room", "what": "classical"},
                              ctx("play some classical music in the living room", hub))
        self.assertTrue(record.ok, record.error)
        self.assertEqual(record.result, {"playing": "Classic FM", "on": "Living room display"})
        self.assertEqual(hub.played, [("Living room display", STATION["url"], "Classic FM", "audio/mpeg", True)])
        found.assert_called_once_with("classical")

    def test_only_when_asked_and_failures_are_reported(self):
        with mock.patch.object(radio, "find_station", return_value=STATION):
            self.assertFalse(run_tool("play_music", {}, ctx("what music do you like?", FakeHub())).ok)
            failed = run_tool("play_music", {}, ctx("play some music", FakeHub(fail=True)))
        self.assertIn("not answering", failed.error)

    def test_station_choice(self):
        stations = [{"name": "HLS only", "codec": "UNKNOWN", "url_resolved": "https://a"},
                    {"name": "Plain", "codec": "MP3", "url_resolved": "http://b"},
                    {"name": "Secure", "codec": "AAC", "url_resolved": "https://c"}]
        with mock.patch.object(radio, "_get", return_value=stations):
            self.assertEqual(radio.find_station("jazz")["name"], "Secure", "playable, https first")


class Scripted:
    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def step(self, messages, tools, first):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self.steps.pop(0)


class HonestyTest(unittest.TestCase):
    def test_a_failed_action_is_never_reported_as_done(self):
        backend = Scripted(Step(calls=[{"name": "play_music", "arguments": {}}]),
                           Step(text="Music should now start playing."),
                           Step(text="I couldn't play music, sir: the living room display is not answering."))
        failed = CallRecord("play_music", {}, False, error="Living room display is not answering")
        with mock.patch("somni_studio.engine.agent.run_tool", lambda *a: failed):
            out = ToolAgent(backend, ctx("play music", FakeHub()), lambda *a, **k: None).run("play music", [], [])
        self.assertTrue(out.text.startswith("I couldn't play music"))
        self.assertIn("every action failed", backend.seen[-1][-1]["content"])


class EndTreeTest(unittest.TestCase):
    def test_the_child_ends_with_its_launcher(self):
        code = ("import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)")
        parent = subprocess.Popen([sys.executable, "-c", code])
        try:
            deadline = time.time() + 10
            while not psutil.Process(parent.pid).children() and time.time() < deadline:
                time.sleep(0.1)
            child = psutil.Process(parent.pid).children()[0]
            end_tree(parent.pid, wait_s=5)
            self.assertFalse(child.is_running() and child.status() != psutil.STATUS_ZOMBIE, "no orphan left")
        finally:
            parent.kill()


if __name__ == "__main__":
    unittest.main()
