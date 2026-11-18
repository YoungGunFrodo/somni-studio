"""Home Assistant set up from Somni: its VM, and "Sign in with Home Assistant" instead of a pasted token."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from somni_studio.integrations import ha_vm, home_assistant
from somni_studio.integrations.base import CheckFailed
from somni_studio.integrations.home_assistant import HomeAssistant, token_for
from somni_studio.integrations.service import public_values


class VmTest(unittest.TestCase):
    def fake(self, vm_state="poweroff"):
        calls = []

        def run(*args):
            calls.append(args)
            if args[0] == "list":
                return '"Ubuntu" {1}\n"Home Assistant" {2}\n'
            if args[0] == "showvminfo":
                return f'name="Home Assistant"\nVMState="{vm_state}"\n'
            return ""
        return calls, run

    def test_finds_starts_and_stops(self):
        calls, run = self.fake()
        with mock.patch.object(ha_vm, "_run", run), mock.patch.object(ha_vm, "vboxmanage", return_value="vbox"):
            self.assertEqual(ha_vm.find_vm(), "Home Assistant", "found by name")
            self.assertIsNone(ha_vm.find_vm("Other"))
            ha_vm.start("Home Assistant")
            ha_vm.stop("Home Assistant")                    # powered off: nothing to stop
        self.assertIn(("startvm", "Home Assistant", "--type", "headless"), calls)
        self.assertFalse(any(c[0] == "controlvm" for c in calls))
        calls, run = self.fake("running")
        with mock.patch.object(ha_vm, "_run", run), mock.patch.object(ha_vm, "vboxmanage", return_value="vbox"):
            ha_vm.stop("Home Assistant")
            self.assertEqual(ha_vm.status(), {"vm": "Home Assistant", "state": "running", "virtualbox": True})
        self.assertIn(("controlvm", "Home Assistant", "acpipowerbutton"), calls, "a clean shutdown")

    def test_no_virtualbox(self):
        with mock.patch.object(ha_vm, "vboxmanage", return_value=None):
            self.assertEqual(ha_vm.status()["state"], "none")


class SignInTokenTest(unittest.TestCase):
    def setUp(self):
        home_assistant._access.clear()

    def test_refresh_token_gives_short_lived_tokens(self):
        replies = iter([{"access_token": "a1", "expires_in": 1800}, {"access_token": "a2", "expires_in": 1800}])
        config = {"url": "http://ha:8123", "refresh_token": "r", "client_id": "http://localhost:8765/"}
        with mock.patch.object(home_assistant, "auth_token", side_effect=lambda url, form: next(replies)) as auth:
            self.assertEqual(token_for(config), "a1")
            self.assertEqual(token_for(config), "a1", "reused until it is about to expire")
            home_assistant._access[("http://ha:8123", "r")] = ("a1", 0)
            self.assertEqual(token_for(config), "a2")
        form = auth.call_args.args[1]
        self.assertEqual((form["grant_type"], form["client_id"]), ("refresh_token", "http://localhost:8765/"))
        self.assertEqual(token_for({"token": "long-lived"}), "long-lived")
        with self.assertRaisesRegex(CheckFailed, "Sign in"):
            token_for({"url": "http://ha:8123"})

    def test_sign_in_data_never_reaches_the_browser(self):
        values = {"url": "http://ha:8123", "token": "", "vm": "", "refresh_token": "secret", "client_id": "x"}
        shown = public_values(HomeAssistant(), values)
        self.assertTrue(shown["signed_in"])
        self.assertNotIn("refresh_token", shown)
        self.assertNotIn("secret", str(shown))
        self.assertNotIn("refresh_token", [f["key"] for f in HomeAssistant().to_wire()["fields"]])


class SignInFlowTest(unittest.TestCase):
    def test_round_trip(self):
        from fastapi.testclient import TestClient

        from somni_studio.api.app import create_app
        from somni_studio.settings import StudioSettings

        from .fakes import FakeProvider
        with tempfile.TemporaryDirectory() as tmp:
            settings = StudioSettings().merged({"somni": {"home": str(Path(tmp) / "home")},
                                                "voice": {"enabled": False}, "listen": {"enabled": False}})
            app = create_app(settings, Path(tmp) / "studio.json", provider_factory=lambda s: FakeProvider(),
                             data_dir=Path(tmp) / "data")
            with mock.patch("somni_studio.api.app.start_home_assistant", mock.AsyncMock()), \
                    TestClient(app, base_url="http://localhost:8765", client=("127.0.0.1", 50000)) as client:
                go = client.get("/api/connections/home_assistant/signin?url=http://ha.test:8123",
                                follow_redirects=False)
                self.assertEqual(go.status_code, 303)
                target = go.headers["location"]
                self.assertTrue(target.startswith("http://ha.test:8123/auth/authorize?"))
                self.assertIn("client_id=http%3A%2F%2Flocalhost%3A8765%2F", target)
                state = target.split("state=")[1].split("&")[0]
                with mock.patch.object(home_assistant, "auth_token", return_value={"refresh_token": "r1"}), \
                        mock.patch.object(HomeAssistant, "check", return_value=home_assistant.Status(
                            home_assistant.State.CONNECTED, "1 entities")):
                    back = client.get(f"/api/connections/home_assistant/callback?code=c1&state={state}",
                                      follow_redirects=False)
                self.assertEqual(back.headers["location"], "/?signin=connected")
                again = client.get(f"/api/connections/home_assistant/callback?code=c1&state={state}",
                                   follow_redirects=False)
                self.assertEqual(again.headers["location"], "/?signin=expired", "a sign-in link works once")
                stored = app.state.studio.connections.config("home_assistant")
                self.assertEqual((stored["refresh_token"], stored["url"]), ("r1", "http://ha.test:8123"))


if __name__ == "__main__":
    unittest.main()
