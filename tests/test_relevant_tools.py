"""Only the tools a request could need go into the prompt (short prompts: fast, clear, small machines)."""
import unittest

from somni_studio.tools.catalog import TOOLS, relevant_tools


def names(question, history=()):
    return {t.name for t in relevant_tools(list(TOOLS), question, history)}


class RelevantToolsTest(unittest.TestCase):
    def test_a_home_request_gets_home_tools_only(self):
        got = names("I've got a new bulb, name it Office light 2 and put it in the office")
        self.assertIn("home_organize", got)
        self.assertIn("devices_find", got)
        self.assertNotIn("email_unread", got)
        self.assertNotIn("play_music", got)
        self.assertNotIn("home_routine", got)
        self.assertLess(len(got), len(TOOLS) // 2 + 2)
        self.assertIn("home_routine", names("turn the lights off every night at 10"))
        self.assertIn("play_music", names("play some jazz in the kitchen"))

    def test_other_requests(self):
        self.assertIn("weather", names("will it rain tomorrow?"))
        self.assertNotIn("home_routine", names("will it rain tomorrow?"))
        self.assertIn("email_unread", names("any new email?"))
        self.assertIn("system_status", names("which model are you running?"))

    def test_follow_ups_keep_the_last_request_s_tools(self):
        history = [{"role": "user", "content": "connect the kitchen lamp"}, {"role": "assistant", "content": "Sure?"}]
        self.assertIn("device_connect", names("yes, do it", history))

    def test_nothing_recognised_offers_everything(self):
        self.assertEqual(names("hmm"), {t.name for t in TOOLS})


if __name__ == "__main__":
    unittest.main()
