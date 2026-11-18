"""Tools: validation, the guards, device control rules, and the agent loop (with a scripted model)."""
import json
import unittest

from somni_studio.engine.agent import Step, ToolAgent, could_be_tool_call, parse_tool_calls, system_message
from somni_studio.events import EventType
from somni_studio.tools import catalog
from somni_studio.tools.catalog import BY_NAME, ToolContext, ToolFailed, available_tools, run_tool, validate


def context(question="", ha_ready=False, remember=None, request_sleep=None, domains=("light", "switch")):
    return ToolContext(integration_config=lambda i: {"url": "http://ha", "token": "t"},
                       integration_ready=lambda i: ha_ready, allowed_domains=frozenset(domains),
                       remember=remember, request_sleep=request_sleep, question=question)


class ValidateTest(unittest.TestCase):
    def test_arguments_are_checked(self):
        tool = BY_NAME["home_control"]
        self.assertEqual(validate(tool, {"entity_id": "light.a", "action": "turn_on"}),
                         {"entity_id": "light.a", "action": "turn_on"})
        for bad in ({"entity_id": "light.a"}, {"entity_id": "light.a", "action": "explode"},
                    {"entity_id": "light.a", "action": "turn_on", "colour": "red"},
                    {"entity_id": 5, "action": "turn_on"}, "not an object"):
            with self.subTest(bad=bad), self.assertRaises(ToolFailed):
                validate(tool, bad)
        args = validate(tool, {"entity_id": "x.y", "action": "set", "brightness_pct": 40.0})
        self.assertEqual(args["brightness_pct"], 40, "a whole float is accepted as an integer")

    def test_tool_names_are_safe_for_openai_style_servers(self):
        import re
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]+", t.name) for t in catalog.TOOLS))


class AvailabilityTest(unittest.TestCase):
    def test_tools_need_their_connection_and_permissions(self):
        names = {t.name for t in available_tools(context())}
        self.assertEqual(names, {"clock_now", "computer_info", "system_status"}, "nothing connected: self and computer")
        names = {t.name for t in available_tools(context(ha_ready=True, remember=lambda t: 1,
                                                         request_sleep=lambda: None))}
        self.assertTrue({"clock_now", "memory_remember", "sleep_start", "home_states", "home_control"} <= names)

    def test_unknown_or_unavailable_tool(self):
        record = run_tool("home_control", {"entity_id": "light.a", "action": "turn_on"}, context())
        self.assertFalse(record.ok)
        self.assertIn("no tool", record.error)


class GuardTest(unittest.TestCase):
    def test_remember_only_when_asked(self):
        saved = []
        ctx = context("Write me a poem about rain", remember=lambda t: saved.append(t) or 1)
        self.assertFalse(run_tool("memory_remember", {"fact": "Rain is wet."}, ctx).ok)
        self.assertEqual(saved, [], "a poem request must never write to memory")
        ctx.question = "Please remember that my sister is called Ana"
        record = run_tool("memory_remember", {"fact": "My sister is called Ana."}, ctx)
        self.assertTrue(record.ok)
        self.assertEqual(saved, ["My sister is called Ana."])

    def test_sleep_only_when_asked(self):
        asked = []
        ctx = context("what's the weather", request_sleep=lambda: asked.append(1))
        self.assertFalse(run_tool("sleep_start", {}, ctx).ok)
        ctx.question = "ok, go to sleep now"
        self.assertTrue(run_tool("sleep_start", {}, ctx).ok)
        self.assertEqual(asked, [1])


class HomeControlTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._states, self._call = catalog.HOME.states, catalog.HOME.call
        devices = [{"entity_id": "light.kitchen", "name": "Kitchen light", "state": "off"},
                   {"entity_id": "lock.front_door", "name": "Front door", "state": "locked"}]
        catalog.HOME.states = lambda config, search="", domain="", limit=40: [
            d for d in devices if all(w in d["name"].lower() for w in search.lower().split())]
        catalog.HOME.call = lambda config, entity, service, data=None: (
            self.calls.append((entity, service, data)) or {"entity_id": entity, "name": entity, "state": "on"})

    def tearDown(self):
        catalog.HOME.states, catalog.HOME.call = self._states, self._call

    def test_control_by_name_and_by_id(self):
        ctx = context("turn on the kitchen light", ha_ready=True)
        self.assertTrue(run_tool("home_control", {"entity_id": "kitchen", "action": "turn_on"}, ctx).ok)
        self.assertTrue(run_tool("home_control", {"entity_id": "light.kitchen", "action": "set",
                                                  "brightness_pct": 150}, ctx).ok)
        self.assertEqual(self.calls, [("light.kitchen", "turn_on", {}),
                                      ("light.kitchen", "turn_on", {"brightness_pct": 100})])

    def test_locks_are_not_allowed_by_default(self):
        record = run_tool("home_control", {"entity_id": "lock.front_door", "action": "turn_off"},
                          context("unlock the front door", ha_ready=True))
        self.assertFalse(record.ok)
        self.assertIn("switched off", record.error)
        self.assertEqual(self.calls, [], "nothing was sent to the lock")

    def test_ambiguous_names_are_not_guessed(self):
        record = run_tool("home_control", {"entity_id": "door light", "action": "turn_on"},
                          context("turn on the door light", ha_ready=True))
        self.assertFalse(record.ok)
        self.assertIn("matches 0 devices", record.error)


class ParseTest(unittest.TestCase):
    def test_parse_qwen_tool_calls(self):
        text = ('<tool_call>\n{"name": "home_control", '
                '"arguments": {"entity_id": "light.a", "action": "turn_on"}}\n</tool_call>')
        self.assertEqual(parse_tool_calls(text), [{"name": "home_control",
                                                   "arguments": {"entity_id": "light.a", "action": "turn_on"}}])
        self.assertEqual(parse_tool_calls('<tool_call>{"name": "clock_now", "arguments": "{}"}'),
                         [{"name": "clock_now", "arguments": {}}], "unclosed tag, string arguments")
        self.assertEqual(parse_tool_calls("Paris is the capital."), [])
        self.assertEqual(parse_tool_calls("<tool_call>{broken</tool_call>"), [])

    def test_early_stop_rule(self):
        self.assertTrue(could_be_tool_call(""))
        self.assertTrue(could_be_tool_call("<tool"))
        self.assertTrue(could_be_tool_call(' <tool_call>{"name"'))
        self.assertFalse(could_be_tool_call("Paris"))

    def test_system_message_lists_devices_and_memories(self):
        from somni_studio.somni.runtime import ensure_sleepai

        from .helpers import SOMNI_REPO, require_somni
        require_somni(self)
        ensure_sleepai(SOMNI_REPO)                     # the memory preamble is Somni's own text

        class M:
            status, strength, key, summary = "active", "verified", "sister/name", "My sister is Ana."
        content = system_message([M()], ["Kitchen light (light.kitchen)"])["content"]
        self.assertIn("Kitchen light (light.kitchen)", content)
        self.assertIn("sister/name: My sister is Ana.", content)


class ScriptedBackend:
    """Replays model turns."""

    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def step(self, messages, tools, first):
        self.seen.append(json.loads(json.dumps(messages)))
        return self.steps.pop(0)


class AgentTest(unittest.TestCase):
    def test_no_tool_hands_back_to_somni(self):
        agent = ToolAgent(ScriptedBackend(None), context(), lambda *a, **k: None)
        self.assertIsNone(agent.run("When was Tesla born?", [], available_tools(context())))

    def test_tool_round_trip(self):
        events = []
        backend = ScriptedBackend(Step(calls=[{"name": "clock_now", "arguments": {}}]), Step(text="It is noon."))
        agent = ToolAgent(backend, context(), lambda kind, **p: events.append((kind, p)))
        result = agent.run("What time is it?", [], available_tools(context()))
        self.assertEqual(result.text, "It is noon.")
        self.assertEqual([r.name for r in result.calls], ["clock_now"])
        self.assertEqual([k for k, _ in events], [EventType.TOOL_CALL, EventType.TOOL_RESULT])
        tool_message = backend.seen[1][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn('"time"', tool_message["content"])

    def test_failed_tool_is_reported_to_the_model(self):
        backend = ScriptedBackend(Step(calls=[{"name": "memory_remember", "arguments": {"fact": "x"}}]),
                                  Step(text="Here is your poem."))
        ctx = context("write a poem", remember=lambda t: 1)
        result = ToolAgent(backend, ctx, lambda *a, **k: None).run("write a poem", [], available_tools(ctx))
        self.assertFalse(result.calls[0].ok)
        self.assertIn("did not ask you to remember", backend.seen[1][-1]["content"])

    def test_round_limit(self):
        loop = Step(calls=[{"name": "clock_now", "arguments": {}}])
        backend = ScriptedBackend(loop, loop, loop)
        result = ToolAgent(backend, context(), lambda *a, **k: None, max_rounds=2).run("time?", [],
                                                                                      available_tools(context()))
        self.assertIn("too many steps", result.text)
        self.assertEqual(len(result.calls), 2)


if __name__ == "__main__":
    unittest.main()
