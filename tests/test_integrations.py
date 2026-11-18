import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from somni_studio.integrations.base import CheckFailed, State
from somni_studio.integrations.email_imap import EmailImap
from somni_studio.integrations.home_assistant import HomeAssistant
from somni_studio.integrations.service import ConnectionsService, ConnectionStore, clean_values, public_values

TOKEN = "test-token"
STATES = [{"entity_id": "light.kitchen"}, {"entity_id": "light.hall"}, {"entity_id": "switch.fan"}]


class FakeHomeAssistant(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_response(401)
            self.end_headers()
            return
        body = {"/api/": {"message": "API running."}, "/api/states": STATES}.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass


class HomeAssistantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeHomeAssistant)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_connected_with_entity_counts(self):
        status = HomeAssistant().check({"url": self.url, "token": TOKEN})
        self.assertEqual(status.state, State.CONNECTED)
        self.assertEqual(status.details["entities"], 3)
        self.assertEqual(status.details["domains"], {"light": 2, "switch": 1})

    def test_bad_token_and_bad_url(self):
        with self.assertRaisesRegex(CheckFailed, "token was refused"):
            HomeAssistant().check({"url": self.url, "token": "wrong"})
        with self.assertRaisesRegex(CheckFailed, "http"):
            HomeAssistant().check({"url": "homeassistant.local", "token": TOKEN})
        with self.assertRaisesRegex(CheckFailed, "cannot reach"):
            HomeAssistant().check({"url": "http://127.0.0.1:9", "token": TOKEN})


class EmailTest(unittest.TestCase):
    def test_missing_fields(self):
        with self.assertRaisesRegex(CheckFailed, "needed"):
            EmailImap().check({"host": "imap.example.com", "username": "", "password": ""})


class ValuesTest(unittest.TestCase):
    def test_secrets_are_kept_and_never_shown(self):
        ha = HomeAssistant()
        values = clean_values(ha, {"url": " http://h:8123 ", "token": "abc"}, {})
        self.assertEqual({k: values[k] for k in ("url", "token")}, {"url": "http://h:8123", "token": "abc"})
        again = clean_values(ha, {"url": "http://h:8123", "token": ""}, values)
        self.assertEqual(again["token"], "abc", "an empty secret keeps the stored one")
        shown = public_values(ha, again)
        self.assertEqual(shown, {"url": "http://h:8123", "token": "", "token_set": True, "vm": "",
                                 "signed_in": False})

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "required"):
            clean_values(HomeAssistant(), {"url": ""}, {})
        with self.assertRaisesRegex(ValueError, "unknown field"):
            clean_values(HomeAssistant(), {"url": "http://x", "token": "t", "extra": 1}, {})
        with self.assertRaisesRegex(ValueError, "number"):
            clean_values(EmailImap(), {"host": "h", "port": "abc", "username": "u", "password": "p"}, {})


class ServiceTest(unittest.TestCase):
    def test_save_list_remove_and_tools(self):
        with tempfile.TemporaryDirectory() as d:
            server = HTTPServer(("127.0.0.1", 0), FakeHomeAssistant)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                events = []
                svc = ConnectionsService(ConnectionStore(Path(d) / "connections.json"), events.append)
                def tools():
                    return {t["name"]: t for t in svc.tools(True, True, tools_enabled=True, writable=True)}
                self.assertFalse(tools()["home_control"]["ready"])
                self.assertIn("connected device", tools()["home_control"]["needs"])
                self.assertTrue(tools()["memory.recall"]["ready"])
                self.assertTrue(tools()["clock_now"]["ready"])
                status = svc.save("home_assistant", {"url": f"http://127.0.0.1:{server.server_port}", "token": TOKEN})
                self.assertEqual(status.state, State.CONNECTED)
                listed = {c["id"]: c for c in svc.list()}
                self.assertEqual(listed["home_assistant"]["values"]["token"], "")
                self.assertTrue(listed["home_assistant"]["values"]["token_set"])
                self.assertNotIn(TOKEN, json.dumps(svc.list()), "the token never leaves the server")
                self.assertTrue(tools()["home_control"]["ready"])
                off = {t["name"]: t for t in svc.tools(True, True, tools_enabled=False, writable=False)}
                self.assertFalse(off["clock_now"]["ready"])
                self.assertFalse(off["memory.learn"]["ready"])
                ha_events = [e for e in events if e["id"] == "home_assistant"]
                self.assertEqual(ha_events[-1]["status"]["state"], "connected")
                google = [e for e in events if e["id"] == "google_home"][-1]
                self.assertEqual(google["status"]["state"], "not_configured", "the fake has no Google SDK service")
                svc.remove("home_assistant")
                self.assertFalse({c["id"]: c for c in svc.list()}["home_assistant"]["configured"])
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
