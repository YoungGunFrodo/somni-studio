import json
import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from somni_studio.api.app import _same_origin, create_app

from .fakes import FakeProvider, factory
from .helpers import HomeTestCase


class ApiTest(HomeTestCase):
    def setUp(self):
        super().setUp()
        FakeProvider.instances.clear()
        self.settings_path = self.root / "studio.json"
        self.client = TestClient(create_app(self.settings, self.settings_path, provider_factory=factory(),
                                            data_dir=self.root / "data"))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        super().tearDown()

    def test_health_and_security_headers(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        csp = r.headers["content-security-policy"]
        self.assertIn("default-src 'self'", csp)
        self.assertIn("'sha256-", csp, "the inline import map must be allowed by hash, not unsafe-inline")
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")

    def test_ui_is_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Somni Studio", r.text)
        self.assertEqual(self.client.get("/js/main.js").status_code, 200)

    def test_graph_snapshot(self):
        g = self.client.get("/api/graph").json()
        self.assertEqual(g["revision"], 1)
        kinds = {n["kind"] for n in g["nodes"]}
        self.assertTrue({"memory", "tag", "file", "layer", "expert", "hub"} <= kinds)
        self.assertTrue(all("id" not in link for link in g["links"]), "link ids are derived by clients")

    def test_gzip(self):
        r = self.client.get("/api/graph", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(r.headers.get("content-encoding"), "gzip")

    def test_node_details(self):
        nodes = {n["label"]: n for n in self.client.get("/api/graph").json()["nodes"]}
        mem = self.client.get(f"/api/nodes/{nodes['normans/named']['id']}").json()
        self.assertEqual(mem["detail"]["value"], "Normandy")
        self.assertEqual([r["key"] for r in mem["detail"]["related"]], ["normans/origin"])
        self.assertTrue(mem["detail"]["file_node"].startswith("file:"))
        f = self.client.get(f"/api/nodes/{mem['detail']['file_node']}").json()
        self.assertIn("Normandy", f["detail"]["text"])
        tag = self.client.get("/api/nodes/tag:history").json()
        self.assertEqual(tag["detail"]["count"], 2)
        expert = self.client.get("/api/nodes/expert:1:0").json()
        self.assertEqual(expert["detail"]["threshold"], 0.6)
        self.assertEqual(self.client.get("/api/nodes/nope").status_code, 404)

    def test_files(self):
        self.assertEqual(self.client.get("/api/files/1").json()["title"], "Normans")
        self.assertEqual(self.client.get("/api/files/99").status_code, 404)
        # TestClient's peer is not loopback: opening files on the desktop is refused.
        self.assertEqual(self.client.post("/api/files/1/open").status_code, 403)

    def test_settings_validation_and_save(self):
        bad = self.client.put("/api/settings", json={"server": {"port": "x"}})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("server.port", bad.json()["detail"])
        self.assertFalse(self.settings_path.exists())
        ok = self.client.put("/api/settings", json={"graph": {"poll_seconds": 3}})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(json.loads(self.settings_path.read_text(encoding="utf-8"))["graph"]["poll_seconds"], 3.0)
        got = self.client.get("/api/settings").json()
        self.assertEqual(got["types"]["graph.poll_seconds"], "float")

    def test_chat_over_websocket(self):
        with self.client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            self.assertEqual(hello["type"], "hello")
            rid = self.client.post("/api/chat", json={"question": "Who named Normandy?"}).json()["request_id"]
            seen = []
            while not seen or seen[-1]["type"] != "chat.done":
                event = ws.receive_json()
                if event.get("request_id") == rid:
                    seen.append(event)
            self.assertEqual([e["type"] for e in seen],
                             ["chat.start", "recall", "activity", "token", "token", "chat.done"])
            self.assertEqual(seen[-1]["text"], "Normandy")

    def test_chat_validation(self):
        self.assertEqual(self.client.post("/api/chat", json={"question": ""}).status_code, 422)

    def test_model_endpoint(self):
        m = self.client.get("/api/model").json()
        self.assertEqual(m["current"], "test/missing-model")
        self.assertIn("learned experts for", m["locked"])        # the fixture home has experts
        self.assertIsInstance(m["local"], list)
        r = self.client.put("/api/model", json={"model": "org/other"})
        self.assertEqual(r.status_code, 409)

    def test_sleep_endpoints(self):
        s = self.client.get("/api/sleep").json()
        self.assertEqual(s["state"], "idle")
        self.assertEqual(s["review"], [])
        self.assertEqual(self.client.post("/api/sleep/wake").status_code, 409)      # not asleep
        self.assertEqual(self.client.post("/api/review/ckpt-9999/approve").status_code, 404)
        self.assertEqual(self.client.post("/api/review/ckpt-0001/maybe").status_code, 404)

    def test_connections_and_tools(self):
        listed = self.client.get("/api/connections").json()["connections"]
        self.assertEqual({c["id"] for c in listed}, {"home_assistant", "google_home", "web", "custom_api", "email"})
        cards = {c["id"]: c for c in listed}
        self.assertTrue(cards["home_assistant"]["setup"] and cards["home_assistant"]["examples"])
        self.assertEqual(cards["google_home"]["derived_from"], "home_assistant")
        self.assertEqual(cards["web"]["fields"][0]["options"], ["wikipedia", "tavily", "brave", "searxng"])
        # TestClient is not a loopback peer: credentials can only be set from this computer.
        r = self.client.put("/api/connections/home_assistant", json={"values": {"url": "http://x", "token": "t"}})
        self.assertEqual(r.status_code, 403)
        tools = {t["name"] for t in self.client.get("/api/tools").json()["tools"]}
        self.assertTrue({"memory.recall", "speech.listen", "clock_now", "sleep_start", "home_control",
                         "email_unread"} <= tools)

    def test_listen_endpoint(self):
        status = self.client.get("/api/listen").json()
        self.assertEqual(status["wake_word"], "somni")
        self.assertEqual(self.client.post("/api/listen?mode=shout", content=b"x").status_code, 400)
        self.assertEqual(self.client.post("/api/listen", content=b"not a wav").status_code, 400)

    def test_tools_appear_in_the_brain(self):
        nodes = {n["id"]: n for n in self.client.get("/api/graph").json()["nodes"]}
        self.assertEqual(nodes["hub:tools"]["kind"], "hub")
        self.assertEqual(nodes["tool:home_control"]["kind"], "tool")

    def test_speech_endpoints(self):
        s = self.client.get("/api/speech").json()
        self.assertFalse(s["enabled"])          # off in the test settings
        self.assertEqual(self.client.get("/api/speech/clips/not-a-clip.wav").status_code, 404)
        self.assertEqual(self.client.get("/api/speech/clips/" + "0" * 32 + ".wav").status_code, 404)

    def test_websocket_rejects_other_origins(self):
        self.assertTrue(_same_origin(None, "localhost:8765"))
        self.assertTrue(_same_origin("http://localhost:8765", "localhost:8765"))
        self.assertFalse(_same_origin("https://evil.example", "localhost:8765"))
        with self.assertRaises(WebSocketDisconnect), self.client.websocket_connect(
                "/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


if __name__ == "__main__":
    unittest.main()
