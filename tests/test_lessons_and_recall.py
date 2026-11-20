"""Tool lessons for sleep, and recall that leaves out weak matches."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from somni_studio.engine import lessons
from somni_studio.engine.agent import Step, ToolAgent
from somni_studio.engine.recall import relevant
from somni_studio.tools import catalog
from somni_studio.tools.catalog import BY_NAME, CallRecord, ToolContext

ACTS = {n for n, t in BY_NAME.items() if t.acts}


def memory(key, summary):
    return SimpleNamespace(key=key, summary=summary, id=key, tags=[], type="fact")


class RecallTest(unittest.TestCase):
    def test_weak_matches_are_left_out(self):
        found = [memory("learned/whats-the-price-of-silver-right-now", "Silver costs about 30 dollars an ounce."),
                 memory("home/devices-on-network", "The office has two WiZ lights.")]
        kept = relevant("turn the office lights off", found)
        self.assertEqual([m.key for m in kept], ["home/devices-on-network"])

    def test_no_content_words_keeps_everything(self):
        found = [memory("a", "anything")]
        self.assertEqual(relevant("is it?", found), found)


class LessonTest(unittest.TestCase):
    def test_outcomes_are_checked(self):
        off = CallRecord("home_control", {"action": "turn_off"}, True, {"now": {"state": "off"}})
        still_on = CallRecord("home_control", {"action": "turn_off"}, True, {"now": {"state": "on"}})
        look = CallRecord("home_states", {}, True, {"devices": []})
        self.assertEqual(lessons.outcome([look, off], ACTS), "done (checked)")
        self.assertEqual(lessons.outcome([still_on], ACTS), "failed")
        self.assertEqual(lessons.outcome([look], ACTS), "looked")
        self.assertEqual(lessons.outcome([], ACTS), "no tool")

    def test_a_corrected_turn_keeps_the_mistake_and_the_clean_version(self):
        steps = iter([Step(calls=[{"name": "home_states", "arguments": {"search": "office"}}]),
                      Step(text="I've turned them off."),                            # only looked
                      Step(calls=[{"name": "home_control", "arguments": {"entity_id": "light.o", "action":
                                                                         "turn_off"}}]),
                      Step(text="Both office lights are off, sir.")])
        results = {"home_states": {"devices": [], "count": 2}, "home_control": {"now": {"state": "off"}}}
        ctx = ToolContext(integration_config=lambda _id: {}, integration_ready=lambda i: i == "home_assistant",
                          allowed_domains=frozenset({"light"}), question="turn the office lights off")
        agent = ToolAgent(mock.Mock(step=mock.Mock(side_effect=lambda *a, **k: next(steps))), ctx, mock.Mock())
        with mock.patch("somni_studio.engine.agent.run_tool",
                        side_effect=lambda n, a, c: catalog.CallRecord(n, a, True, results[n])):
            result = agent.run(ctx.question, [], [BY_NAME["home_states"], BY_NAME["home_control"]])
        self.assertEqual(result.corrections, [{"kind": "stopped before finishing", "wrong": "I've turned them off."}])
        roles = [m["role"] for m in result.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertNotIn("I've turned them off.", json.dumps(result.messages), "the mistake is not in the lesson")
        self.assertNotIn("Not finished", json.dumps(result.messages), "nor is the correction")
        with tempfile.TemporaryDirectory() as home:
            kept = lessons.record(Path(home), ctx.question, [], result, ACTS, day=1)
            self.assertEqual(kept["outcome"], "done (checked)")
            self.assertEqual(lessons.summary(Path(home)),
                             {"lessons": 1, "outcomes": {"done (checked)": 1}, "needed_correcting": 1})


if __name__ == "__main__":
    unittest.main()
