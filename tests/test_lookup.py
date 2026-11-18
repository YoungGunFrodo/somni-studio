"""Looking things up instead of saying "I'm not sure": the weather tool and the fallback."""
import json
import unittest
from unittest import mock

from somni_studio import location
from somni_studio.engine.agent import Step
from somni_studio.engine.learning import fallback_call, home_place
from somni_studio.engine.providers import _RecallingProvider
from somni_studio.integrations import web
from somni_studio.integrations.base import CheckFailed
from somni_studio.tools.catalog import ToolContext, run_tool

from .helpers import HomeTestCase

GEO = {"results": [{"name": "Paris", "admin1": "Texas", "country": "United States", "country_code": "US",
                    "latitude": 33.6, "longitude": -95.5},
                   {"name": "Paris", "admin1": "Île-de-France", "country": "France", "country_code": "FR",
                    "latitude": 48.8, "longitude": 2.3}]}
FORECAST = {"current": {"temperature_2m": 15.4, "apparent_temperature": 12.1, "relative_humidity_2m": 55,
                        "weather_code": 61, "wind_speed_10m": 19.2},
            "daily": {"time": ["2026-11-18", "2026-11-19", "2026-11-20"], "weather_code": [61, 3, 0],
                      "temperature_2m_max": [16, 14, 12], "temperature_2m_min": [10, 9, 5],
                      "precipitation_probability_max": [80, 20, None]}}


def fake_fetch(calls):
    def fetch(url, headers=None, public_only=True):
        calls.append(url)
        return json.dumps(GEO if "geocoding" in url else FORECAST).encode(), "application/json"
    return fetch


class WeatherTest(unittest.TestCase):
    def test_place_region_and_units(self):
        calls = []
        with mock.patch.object(web, "_fetch", fake_fetch(calls)):
            report = web.WebLookup().weather("Paris, France")
        self.assertEqual(report["place"], "Paris, Île-de-France, France")
        self.assertEqual(report["now"]["sky"], "light rain")
        self.assertEqual(report["now"]["temperature"], "15°C")
        self.assertEqual(report["forecast"][1]["day"], "tomorrow (Thursday)")
        self.assertIn("latitude=48.8", calls[1])
        self.assertIn("temperature_unit=celsius", calls[1])
        with mock.patch.object(web, "_fetch", fake_fetch(calls)):
            self.assertIn("°F", web.WebLookup().weather("Paris, Texas")["now"]["temperature"])

    def test_the_tool_needs_a_town(self):
        ctx = ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: True,
                          allowed_domains=frozenset())
        missing = run_tool("weather", {"place": "here"}, ctx)
        self.assertIn("computer_info", missing.error, "it says how to find the town, it does not guess")


class ComputerInfoTest(unittest.TestCase):
    def setUp(self):
        location._cache.update(at=0.0, place="")

    def test_where_the_computer_is(self):
        learned = []
        ctx = ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: True,
                          allowed_domains=frozenset(), learn_place=learned.append)
        with mock.patch.object(location, "computer_location", return_value=(45.52, -122.68)) as asked, \
                mock.patch.object(location, "town_name", return_value="Portland, Oregon"):
            first = run_tool("computer_info", {}, ctx)
            run_tool("computer_info", {}, ctx)
        self.assertEqual(first.result["location"], "Portland, Oregon")
        self.assertIn("time_zone", first.result)
        self.assertEqual(learned[0], "Portland, Oregon", "kept as a memory")
        self.assertEqual(asked.call_count, 1, "looked up once, then cached")

    def test_location_switched_off(self):
        ctx = ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: True,
                          allowed_domains=frozenset())
        with mock.patch.object(location, "computer_location", return_value=None):
            self.assertIn("unknown", run_tool("computer_info", {}, ctx).result["location"])


class FallbackRulesTest(unittest.TestCase):
    def test_what_gets_looked_up(self):
        self.assertEqual(fallback_call("whats the weather in Chicago?"),
                         {"name": "weather", "arguments": {"place": "Chicago"}})
        self.assertEqual(fallback_call("will it rain tomorrow?"), {"name": "weather", "arguments": {}})
        self.assertEqual(fallback_call("who won the world series in 2025?")["name"], "web_search")
        for q in ("whats my dogs name?", "hello there", "you can call me sir", "what do I like?"):
            self.assertIsNone(fallback_call(q), q)
        self.assertEqual(home_place(["The user lives in denver."]), "denver")


class Student:
    def answer(self, question, context=None):
        from sleepai.models.base import Answer
        return Answer("I'm not sure.", "i'm not sure", 1.0)


class Backend:
    """Never picks a tool itself (like the model that said "I'm not sure"), answers from results."""

    def __init__(self):
        self.seen = []

    def step(self, messages, tools, first):
        if first:
            return None
        self.seen.append(messages[-1])
        return Step(text="It is 15°C and raining in Paris.")

    def complete(self, messages, max_new):
        return "NONE"


class Provider(_RecallingProvider):
    name = "test"

    def _make_student(self):
        return Student()

    def _tool_backend(self):
        return self.backend


class FallbackTest(HomeTestCase):
    def test_not_sure_becomes_a_lookup(self):
        from somni_studio.engine.providers import ToolEnvironment

        class Connections:
            def config(self, i):
                return {"provider": "wikipedia", "home": "Paris, France"}

            def details(self, i):
                return {}

            def integrations(self):
                return []

            def status_of(self, integ):
                raise AssertionError

        provider = Provider(self.settings)
        provider.backend = Backend()
        provider.tool_env = ToolEnvironment(Connections())
        provider.load()
        events = []
        try:
            with mock.patch("somni_studio.engine.providers.ready_checker", lambda c: lambda i: i == "web"), \
                    mock.patch.object(web, "_fetch", fake_fetch([])):
                out = provider.ask("what's the weather like in Paris, France?",
                                   lambda kind, **p: events.append((kind.value, p)))
                self.assertEqual(out.text, "It is 15°C and raining in Paris.")
                self.assertEqual([t["name"] for t in out.tools], ["weather"], "weather goes straight to its tool")
                self.assertIn("light rain", provider.backend.seen[0]["content"])
                self.assertNotIn(("token", {"text": "", "replace": True}), events, "no 'not sure' first")
                events.clear()
                out = provider.ask("how tall is the eiffel tower?", lambda kind, **p: events.append((kind.value, p)))
                self.assertEqual([t["name"] for t in out.tools], ["web_search", "memory_remember"],
                                 "not sure, so it looked it up (and kept what it found)")
                self.assertIn(("token", {"text": "", "replace": True}), events, "the 'not sure' is taken back")
                mine = provider.ask("whats my dogs name?", lambda *a, **k: None)
                self.assertEqual(mine.text, "I'm not sure.", "questions about the user never go to the web")
        finally:
            provider.close()


class NoRealNetworkTest(unittest.TestCase):
    def test_geocoding_failure_is_an_answer(self):
        with mock.patch.object(web, "_fetch", lambda url, **k: (b'{"results": []}', "application/json")), \
                self.assertRaisesRegex(CheckFailed, "could not find a place"):
            web.WebLookup().weather("Nowhereville")


if __name__ == "__main__":
    unittest.main()
