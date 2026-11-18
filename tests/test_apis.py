"""Keys for outside services: Tavily search, any API through the Custom API card, and finishing a task."""
import json
import unittest
from unittest import mock

from somni_studio.engine.agent import Step, ToolAgent
from somni_studio.integrations import web
from somni_studio.integrations.base import CheckFailed
from somni_studio.integrations.custom_api import CustomApi
from somni_studio.tools.catalog import CallRecord, ToolContext, available_tools


class TavilyTest(unittest.TestCase):
    def test_search_sends_the_key_and_keeps_page_text(self):
        sent = {}

        def fetch(url, headers=None, public_only=True, method="GET", body=None):
            sent.update(url=url, headers=headers, method=method, body=json.loads(body))
            hit = {"title": "2025 World Series", "url": "https://x.example", "content": "The Dodgers beat..."}
            return json.dumps({"answer": "The Dodgers.", "results": [hit]}).encode(), ""
        with mock.patch.object(web, "_fetch", fetch):
            found = web.WebLookup().search({"provider": "tavily", "api_key": "tvly-1"}, "who won the world series")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer tvly-1")
        self.assertEqual((sent["method"], sent["body"]["query"]), ("POST", "who won the world series"))
        self.assertEqual(found["summary"], "The Dodgers.")
        self.assertEqual(found["results"][0]["snippet"], "The Dodgers beat...")
        with self.assertRaisesRegex(CheckFailed, "API key"):
            web.WebLookup().search({"provider": "tavily"}, "x")


class CustomApiTest(unittest.TestCase):
    config = {"name": "News", "url": "https://api.news.example/v2", "auth": "bearer", "api_key": "k-123",
              "about": "Headlines: GET /top?country=us"}

    def test_the_key_is_added_by_somni(self):
        url, headers = CustomApi.request(self.config, "GET", "/top?country=us")
        self.assertEqual(url, "https://api.news.example/v2/top?country=us")
        self.assertEqual(headers["Authorization"], "Bearer k-123")
        url, headers = CustomApi.request({**self.config, "auth": "query", "auth_name": "apikey"}, "GET", "top")
        self.assertTrue(url.endswith("/top?apikey=k-123"))
        _, headers = CustomApi.request({**self.config, "auth": "header", "auth_name": "X-Key"}, "GET", "")
        self.assertEqual(headers["X-Key"], "k-123")

    def test_calls_stay_on_the_api(self):
        for path in ("https://evil.example/steal", "//evil.example/x"):
            with self.subTest(path=path), self.assertRaises(CheckFailed):
                CustomApi.request(self.config, "GET", path)
        with self.assertRaises(CheckFailed):
            CustomApi.request(self.config, "DELETE", "/x")

    def test_the_model_is_told_what_it_is_for_never_the_key(self):
        note = CustomApi.note(self.config)
        self.assertIn("Headlines: GET /top?country=us", note)
        self.assertNotIn("k-123", note)
        ctx = ToolContext(integration_config=lambda i: self.config, integration_ready=lambda i: i == "custom_api",
                          allowed_domains=frozenset())
        self.assertEqual(ctx.notes(), [note])
        self.assertIn("api_call", {t.name for t in available_tools(ctx)})

    def test_a_refused_call_does_not_leak_the_key(self):
        def fetch(url, *a, **k):
            raise CheckFailed(f"cannot reach {url}")
        with mock.patch.object(web, "_fetch", fetch), self.assertRaises(CheckFailed) as caught:
            CustomApi().call({**self.config, "auth": "query"}, "GET", "/top")
        self.assertNotIn("k-123", str(caught.exception))


class AbilitiesTest(unittest.TestCase):
    def test_it_says_what_is_really_connected(self):
        from somni_studio.tools.catalog import abilities
        bare = ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: False,
                           allowed_domains=frozenset())
        text = abilities(bare)
        self.assertIn("Not connected: web search, smart devices (none connected yet), email, calendar", text)
        tavily = ToolContext(integration_config=lambda i: {"provider": "tavily"},
                             integration_ready=lambda i: i == "web", allowed_domains=frozenset())
        self.assertIn("search the whole web (with tavily)", abilities(tavily))


class Scripted:
    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def step(self, messages, tools, first):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self.steps.pop(0)


class FinishTheTaskTest(unittest.TestCase):
    def test_one_nudge_when_it_stops_halfway(self):
        ctx = ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: False,
                          allowed_domains=frozenset())
        backend = Scripted(Step(calls=[{"name": "weather", "arguments": {}}]),
                           Step(calls=[{"name": "computer_info", "arguments": {}}]),
                           Step(text="The weather is not available."),           # gave up halfway
                           Step(calls=[{"name": "weather", "arguments": {"place": "Portland"}}]),
                           Step(text="Clear and 39°F in Portland."))
        results = iter([CallRecord("weather", {}, False, error="weather needs a town"),
                        CallRecord("computer_info", {}, True, {"location": "Portland"}),
                        CallRecord("weather", {"place": "Portland"}, True, {"place": "Portland"})])
        with mock.patch("somni_studio.engine.agent.run_tool", lambda *a: next(results)):
            out = ToolAgent(backend, ctx, lambda *a, **k: None, max_rounds=4).run("weather?", [], [])
        self.assertEqual(out.text, "Clear and 39°F in Portland.")
        self.assertIn("Call weather again", backend.seen[3][-1]["content"])


if __name__ == "__main__":
    unittest.main()
