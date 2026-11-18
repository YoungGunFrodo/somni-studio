"""Knowing itself: its own systems, what it is connected to (kept as memories), and what tools found."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from somni_studio.devices import hub as hub_module
from somni_studio.devices import net
from somni_studio.devices.hub import DeviceHub
from somni_studio.somni.home import HomePaths
from somni_studio.somni.identity import note_connection
from somni_studio.tools.catalog import ToolContext, run_tool

from .helpers import HomeTestCase
from .test_devices import FakeDriver


class ConnectionMemoryTest(HomeTestCase):
    def test_connect_then_disconnect(self):
        from sleepai.memory.store import MemoryStore
        paths = HomePaths(self.home)
        note_connection(paths, "web", "Somni is connected to web lookup, searching with tavily, and weather.")
        note_connection(paths, "web", "Somni is connected to web lookup, searching with tavily, and weather.")
        note_connection(paths, "web", "Somni is no longer connected to Web lookup.")
        store = MemoryStore(paths.memory_db)
        try:
            memory = store.active_by_key("somni/connected/web")
            self.assertEqual(memory.summary, "Somni is no longer connected to Web lookup.")
            self.assertEqual(memory.strength, "verified")
            self.assertEqual(len(store.by_key_any("somni/connected/web")), 2, "the old one is kept as history")
        finally:
            store.db.close()


class HubChangesTest(unittest.TestCase):
    def test_connecting_and_forgetting_a_device_are_noted(self):
        driver, notes = FakeDriver(), []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(hub_module, "DRIVERS", (driver,)), mock.patch.object(hub_module, "BY_ID",
                                                                                     {"fake": driver}), \
                mock.patch.object(net, "ssdp", return_value={}), mock.patch.object(net, "mdns", return_value={}):
            hub = DeviceHub(Path(tmp) / "devices.json")
            hub.on_change = lambda thing, sentence: notes.append(sentence)
            hub.connect("desk lamp")
            hub.forget("fake:aa11")
            found = run_tool("devices_find", {}, ToolContext(integration_config=lambda i: {},
                                                             integration_ready=lambda i: False,
                                                             allowed_domains=frozenset(), devices=hub))
        self.assertIn("connected to the Desk lamp", notes[0])
        self.assertIn("no longer connected to the Desk lamp", notes[1])
        self.assertEqual(found.result["learned"], "The user's home network has: Desk lamp (Fake), Hall bulb (Fake).")


class SystemStatusTest(unittest.TestCase):
    def test_it_can_check_itself(self):
        ctx = ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: False,
                          allowed_domains=frozenset(), system_info=lambda: {"model": "Qwen3-4B", "memories": 9})
        record = run_tool("system_status", {}, ctx)
        self.assertEqual((record.result["model"], record.result["memories"]), ("Qwen3-4B", 9))
        self.assertIn("Not connected:", record.result["abilities"])


class ServiceChangesTest(unittest.TestCase):
    def test_saving_and_removing_a_connection_are_noted(self):
        from somni_studio.integrations.base import State, Status
        from somni_studio.integrations.service import ConnectionsService, ConnectionStore
        with tempfile.TemporaryDirectory() as tmp:
            service = ConnectionsService(ConnectionStore(Path(tmp) / "c.json"), lambda payload: None)
            notes = []
            service.on_change = lambda thing, sentence: notes.append((thing, sentence))
            with mock.patch.object(service, "check", return_value=Status(State.CONNECTED)):
                service.save("web", {"provider": "tavily", "api_key": "k"})
            service.remove("web")
        self.assertEqual(notes[0], ("web", "Somni is connected to web lookup, searching with tavily, and weather."))
        self.assertEqual(notes[1], ("web", "Somni is no longer connected to Web lookup."))


if __name__ == "__main__":
    unittest.main()
