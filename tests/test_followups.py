"""Doing what was asked: no claims without a tool, follow-ups carry on, scenes, exact names."""
import unittest
from unittest import mock

from somni_studio.engine.agent import Step, ToolAgent, _wants_action
from somni_studio.tools import catalog
from somni_studio.tools.catalog import BY_NAME, ToolContext, continued_request, run_tool

CONFIG = {"url": "http://ha.test", "token": "t"}


def context(question):
    return ToolContext(integration_config=lambda _id: CONFIG, integration_ready=lambda i: i == "home_assistant",
                       allowed_domains=frozenset({"light"}), question=question)


class FollowUpTest(unittest.TestCase):
    def test_a_complaint_carries_on_the_request(self):
        history = [{"role": "user", "content": "make them purple"}, {"role": "assistant", "content": "Done."}]
        self.assertIn("make them purple", continued_request("they are still blue", history))
        self.assertEqual(continued_request("are they on?", history), "are they on?", "a question stays a question")
        self.assertEqual(continued_request("turn them off", history), "turn them off")
        self.assertEqual(continued_request("thanks", [{"role": "user", "content": "what time is it"}]), "thanks")

    def test_only_house_requests_must_use_a_tool(self):
        self.assertTrue(_wants_action("put the lights on the cozy setting"))
        self.assertTrue(_wants_action("make them purple"))
        self.assertFalse(_wants_action("write a poem on love"))
        self.assertTrue(_wants_action("switch off the desk lamp", ["Desk lamp (light.desk)"]))


class NoClaimWithoutAToolTest(unittest.TestCase):
    def agent(self, question, steps):
        steps = iter(steps)
        backend = mock.Mock(step=mock.Mock(side_effect=lambda *a, **k: next(steps)))
        ctx = context(question)
        ctx.device_names = lambda: ["Office light (light.office_1)"]
        tools = [BY_NAME[n] for n in ("home_states", "home_control", "memory_remember")]
        return ToolAgent(backend, ctx, mock.Mock()), tools

    def test_it_is_sent_back_to_use_a_tool(self):
        agent, tools = self.agent("put the lights on the cozy setting", [
            None,                                                              # decided: no tool
            Step(calls=[{"name": "home_control", "arguments": {"entity_id": "light.office_1", "action": "set",
                                                               "scene": "Cozy"}}]),
            Step(text="The office light is on Cozy, sir.")])
        with mock.patch("somni_studio.engine.agent.run_tool", side_effect=lambda n, a, c: catalog.CallRecord(
                n, a, True, {"done": "turn_on light.office_1", "now": {"name": "Office light", "state": "on"}})):
            result = agent.run("put the lights on the cozy setting", [], tools)
        self.assertEqual([r.name for r in result.calls], ["home_control"])

    def test_still_no_tool_means_saying_so(self):
        agent, tools = self.agent("make them purple", [None, Step(text="They are purple now.")])
        result = agent.run("make them purple", [], tools)
        self.assertIn("nothing was changed", result.text)

    def test_other_requests_are_left_alone(self):
        agent, tools = self.agent("write a poem on love", [None])
        self.assertIsNone(agent.run("write a poem on love", [], tools))


class ScenesAndNamesTest(unittest.TestCase):
    STATES = [{"entity_id": "light.office_1", "name": "Office light", "state": "on"},
              {"entity_id": "light.office_2", "name": "Office light 2", "state": "on"}]

    def control(self, args, question="put the office light on the cozy setting"):
        with mock.patch.object(catalog.HOME, "states", return_value=self.STATES), \
                mock.patch.object(catalog.HOME, "attributes", return_value={"effect_list": ["Ocean", "Cozy"]}), \
                mock.patch.object(catalog.HOME, "call", return_value={"state": "on"}) as call:
            out = run_tool("home_control", args, context(question))
        return out, call

    def test_a_scene_by_name_or_as_a_color(self):
        out, call = self.control({"entity_id": "Office light", "action": "set", "scene": "cozy setting"})
        self.assertTrue(out.ok, out.error)
        self.assertEqual(call.call_args.args[1:], ("light.office_1", "turn_on", {"effect": "Cozy"}),
                         "'Office light' is the exact name, not also 'Office light 2'")
        out, call = self.control({"entity_id": "light.office_2", "action": "set", "color": "cozy"})
        self.assertEqual(call.call_args.args[3], {"effect": "Cozy"})
        out, _ = self.control({"entity_id": "light.office_2", "action": "set", "scene": "disco"})
        self.assertIn("its scenes are: Ocean, Cozy", out.error)

    def test_lights_report_their_color_and_scene(self):
        from somni_studio.integrations.home_assistant import _friendly
        blue = _friendly({"entity_id": "light.a", "state": "on",
                          "attributes": {"friendly_name": "A", "color_mode": "rgb", "rgb_color": [0, 0, 255]}})
        self.assertEqual(blue["color"], "blue")
        cozy = _friendly({"entity_id": "light.a", "state": "on",
                          "attributes": {"friendly_name": "A", "effect": "Cozy", "effect_list": ["Cozy", "Ocean"]}})
        self.assertEqual((cozy["scene"], cozy["scenes"]), ("Cozy", 2))


if __name__ == "__main__":
    unittest.main()
