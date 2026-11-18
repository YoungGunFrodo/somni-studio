"""Connecting things: the action guard, both tool-call formats, Google Home status, web lookup."""
import json
import unittest
from unittest import mock

from somni_studio.engine.agent import parse_tool_calls
from somni_studio.integrations import web
from somni_studio.integrations.base import CheckFailed, State, Status
from somni_studio.integrations.google_home import GoogleHome
from somni_studio.tools.catalog import BY_NAME, ToolContext, asked_to_act, available_tools


class ActionGuardTest(unittest.TestCase):
    def test_requests_and_questions(self):
        for q in ("Turn on the kitchen light", "lights off", "Can you dim the bedroom?", "Set the thermostat to 21",
                  "kitchen light on please", "turn it up", "unlock the front door", "ask google to start the vacuum"):
            with self.subTest(q=q):
                self.assertTrue(asked_to_act(q))
        for q in ("Is the fan on?", "are the lights off", "What is the temperature?", "check if the door light is on",
                  "how warm is it inside"):
            with self.subTest(q=q):
                self.assertFalse(asked_to_act(q))


class XmlToolCallTest(unittest.TestCase):
    def test_qwen35_format(self):
        text = ("<tool_call>\n<function=home_control>\n<parameter=entity_id>\nlight.kitchen\n</parameter>\n"
                "<parameter=action>\nturn_on\n</parameter>\n<parameter=brightness_pct>\n40\n</parameter>\n"
                "</function>\n</tool_call>")
        self.assertEqual(parse_tool_calls(text), [{"name": "home_control", "arguments": {
            "entity_id": "light.kitchen", "action": "turn_on", "brightness_pct": 40}}])

    def test_two_calls_and_mixed_formats(self):
        text = ('<tool_call>{"name": "clock_now", "arguments": {}}</tool_call>'
                "<tool_call><function=web_search><parameter=query>tallest tower</parameter></function></tool_call>")
        self.assertEqual([c["name"] for c in parse_tool_calls(text)], ["clock_now", "web_search"])


class GoogleHomeTest(unittest.TestCase):
    def test_status_follows_home_assistant(self):
        self.assertEqual(GoogleHome.from_home_assistant(Status(State.NOT_CONFIGURED)).state, State.NOT_CONFIGURED)
        without = Status(State.CONNECTED, details={"services": ["light", "switch"]})
        self.assertIn("Google Assistant SDK", GoogleHome.from_home_assistant(without).message)
        with_sdk = Status(State.CONNECTED, details={"services": ["light", "google_assistant_sdk"]})
        self.assertEqual(GoogleHome.from_home_assistant(with_sdk).state, State.CONNECTED)

    def test_google_tool_needs_the_sdk_service(self):
        def ctx(services):
            return ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: True,
                               allowed_domains=frozenset(), integration_details=lambda i: {"services": services})
        self.assertNotIn("google_command", {t.name for t in available_tools(ctx(["light"]))})
        self.assertIn("google_command", {t.name for t in available_tools(ctx(["light", "google_assistant_sdk"]))})
        self.assertTrue(BY_NAME["google_command"].acts)


class WebSearchTest(unittest.TestCase):
    def fake_fetch(self, responses):
        calls = []

        def fetch(url, headers=None, public_only=True):
            calls.append((url, headers or {}, public_only))
            for prefix, body in responses.items():
                if url.startswith(prefix):
                    return json.dumps(body).encode(), "application/json"
            raise CheckFailed("unexpected " + url)
        return fetch, calls

    def test_wikipedia(self):
        fetch, calls = self.fake_fetch({
            "https://en.wikipedia.org/w/api.php": {"query": {"search": [
                {"title": "Eiffel Tower", "snippet": "The <span>Eiffel</span> Tower is 330 m"}]}},
            "https://en.wikipedia.org/api/rest_v1/page/summary/": {"extract": "Wrought-iron tower in Paris."}})
        with mock.patch.object(web, "_fetch", fetch):
            r = web.WebLookup().search({"provider": "wikipedia"}, "eiffel tower height")
        self.assertEqual(r["results"][0]["snippet"], "The Eiffel Tower is 330 m")
        self.assertEqual(r["summary"], "Wrought-iron tower in Paris.")
        self.assertTrue(r["results"][0]["url"].endswith("/wiki/Eiffel_Tower"))

    def test_brave_needs_a_key_and_sends_it(self):
        with self.assertRaisesRegex(CheckFailed, "API key"):
            web.WebLookup().search({"provider": "brave"}, "x")
        fetch, calls = self.fake_fetch({"https://api.search.brave.com/": {"web": {"results": [
            {"title": "T", "url": "https://t.example", "description": "<b>d</b>"}]}}})
        with mock.patch.object(web, "_fetch", fetch):
            r = web.WebLookup().search({"provider": "brave", "api_key": "k"}, "x")
        self.assertEqual(r["results"], [{"title": "T", "url": "https://t.example", "snippet": "d"}])
        self.assertEqual(calls[0][1]["X-Subscription-Token"], "k")

    def test_searxng_may_be_on_the_local_network(self):
        fetch, calls = self.fake_fetch({"http://localhost:8888/search": {"results": [
            {"title": "T", "url": "https://t.example", "content": "c"}]}})
        with mock.patch.object(web, "_fetch", fetch):
            r = web.WebLookup().search({"provider": "searxng", "url": "http://localhost:8888"}, "x")
        self.assertEqual(r["results"][0]["snippet"], "c")
        self.assertFalse(calls[0][2], "the user's own search server is allowed to be local")


class WebReadTest(unittest.TestCase):
    def test_private_addresses_are_refused(self):
        for url in ("http://192.168.1.1/", "http://localhost:8765/", "http://10.0.0.5/", "http://[::1]/",
                    "file:///etc/passwd", "ftp://example.com/"):
            with self.subTest(url=url), self.assertRaises(CheckFailed):
                web.WebLookup().read(url)

    def test_redirect_to_the_lan_is_refused(self):
        handler = web._PublicOnlyRedirects()
        with self.assertRaises(web.Blocked):
            handler.redirect_request(None, None, 302, "Found", {}, "http://192.168.0.10/admin")

    def test_page_text(self):
        page = (b"<html><head><title>Hi</title><style>p{}</style><script>evil()</script></head>"
                b"<body><nav>menu</nav><h1>Tower</h1><p>It is <b>330 m</b> tall.</p></body></html>")
        title, text = web.page_text(page, "text/html; charset=utf-8")
        self.assertEqual(title, "Hi")
        self.assertIn("330 m tall", text)
        self.assertNotIn("evil", text)
        self.assertNotIn("menu", text)


class CitationTest(unittest.TestCase):
    def test_invented_citations_are_removed(self):
        from somni_studio.engine.providers import strip_citations
        self.assertEqual(strip_citations('It is 324 m tall. [memory: somni/fact: "324 m"]'), "It is 324 m tall.")
        self.assertEqual(strip_citations("No citation here."), "No citation here.")


if __name__ == "__main__":
    unittest.main()
