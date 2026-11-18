"""Learning during the day: conversations as episodes, memories made right away, profile recall."""
import sqlite3
import unittest

from somni_studio.engine.conversation import GAP_S, Conversation, with_history
from somni_studio.engine.learning import Fact, asked_to_save, is_question, parse_fact, personal_statement
from somni_studio.engine.providers import _RecallingProvider
from somni_studio.engine.recall import open_read_only_store, recording_orchestrator
from somni_studio.somni.store_reader import StoreReader
from somni_studio.tools.catalog import CallRecord

from .helpers import HomeTestCase


class RulesTest(unittest.TestCase):
    def test_what_counts_as_a_request_or_a_statement(self):
        self.assertTrue(asked_to_save("remember that i want you to call me sir"))
        self.assertFalse(asked_to_save("do you remember my name?"))
        self.assertTrue(personal_statement("you can call me sir"))
        self.assertTrue(personal_statement("My sister is called Ana"))
        self.assertTrue(personal_statement("I go to bed at eleven"))
        self.assertFalse(personal_statement("whats my name"))
        self.assertFalse(personal_statement("Write me a poem about rain"))
        self.assertTrue(is_question("whats my name"))

    def test_parse(self):
        self.assertEqual(parse_fact("preferred name | The user wants to be called Sir", False),
                         Fact("preferred name", "The user wants to be called Sir."))
        self.assertIsNone(parse_fact("NONE", False))
        self.assertIsNone(parse_fact("weather | It is raining.", False), "not about the user: not saved unasked")
        self.assertEqual(parse_fact("wifi | The wifi password is on the router.", True).topic, "wifi")
        self.assertTrue(parse_fact("The plumber comes on Tuesday", True).sentence.endswith("Tuesday."))


class FakeStore:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, transcript TEXT, formed INTEGER DEFAULT 0)")

    def log_episode(self, kind, transcript, outcome="unknown", day=0):
        import json
        return self.db.execute("INSERT INTO episodes(transcript) VALUES (?)", (json.dumps(transcript),)).lastrowid


def exchange(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


class ConversationTest(unittest.TestCase):
    def test_one_episode_per_conversation(self):
        store, conv = FakeStore(), Conversation()
        first = conv.log(store, exchange("whats my name", "I'm not sure."), 1)
        self.assertEqual(conv.log(store, exchange("you can call me sir", "Okay, sir."), 1), first)
        self.assertEqual(len(conv.turns), 4)
        self.assertEqual([m["content"] for m in conv.history()][-2:], ["you can call me sir", "Okay, sir."])
        store.db.execute("UPDATE episodes SET formed = 1")          # a sleep formed it
        self.assertNotEqual(conv.log(store, exchange("hi", "hello"), 1), first)
        conv.last_at -= GAP_S + 1                                    # a long pause
        self.assertEqual(conv.history(), [])
        self.assertNotEqual(conv.log(store, exchange("again", "yes"), 1), first + 1)

    def test_history_goes_between_system_and_question(self):
        msgs = with_history([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                            exchange("a", "b"))
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])


class Student:
    def answer(self, question, context=None):
        from sleepai.models.base import Answer
        return Answer("I'm not sure.", "i'm not sure", 1.0)


class Provider(_RecallingProvider):
    name = "test"
    reply = "NONE"

    def _make_student(self):
        return Student()

    def _tool_backend(self):
        provider = self

        class Backend:
            def complete(self, messages, max_new):
                if "just told you" in messages[0]["content"]:     # the acknowledgement
                    return "Got it, Sir."
                provider.prompts.append(messages)
                return provider.reply
        return Backend()


class LearningTest(HomeTestCase):
    def setUp(self):
        super().setUp()
        self.provider = Provider(self.settings)
        self.provider.prompts = []
        self.provider.load()
        self.events = []

    def tearDown(self):
        self.provider.close()
        super().tearDown()

    def emit(self, kind, **payload):
        self.events.append((kind, payload))

    def test_a_statement_about_the_user_is_saved_and_recalled(self):
        self.provider.reply = "preferred name | The user wants to be called Sir."
        out = self.provider.ask("you can call me sir", self.emit)
        self.assertIn("remembered: The user wants to be called Sir.", [t["summary"] for t in out.tools])
        self.assertEqual(out.text, "Got it, Sir.", "a statement is acknowledged, not 'not sure'")
        memory = self.provider.store.active_by_key("user/preferred-name")
        self.assertEqual(memory.strength, "user-confirmed")
        recalled = self.provider._orchestrator.recall("whats my name")
        self.assertIn("user/preferred-name", [m.key for m in recalled], "no shared word, still recalled")
        self.assertIn("profile", {i.source for i in self.provider._orchestrator.last_recall})

    def test_a_newer_statement_replaces_the_old_one(self):
        self.provider.reply = "preferred name | The user wants to be called Sir."
        self.provider.ask("you can call me sir", self.emit)
        self.provider.reply = "preferred name | The user wants to be called Bob."
        self.provider.ask("actually, call me Bob", self.emit)
        self.assertEqual(self.provider.store.active_by_key("user/preferred-name").summary,
                         "The user wants to be called Bob.")

    def test_a_new_home_replaces_the_guessed_one(self):
        self.provider._learn_place("Portland, Oregon")
        self.provider.reply = "where they live | The user lives in Austin, Texas."
        self.provider.ask("actually I live in Austin, Texas", self.emit)
        home = self.provider.store.active_by_key("user/home")
        self.assertEqual((home.summary, home.strength), ("The user lives in Austin, Texas.", "user-confirmed"))

    def test_questions_and_poems_are_never_saved(self):
        self.provider.reply = "topic | The user likes rain."
        for q in ("whats my name", "Write me a poem about rain"):
            self.provider.ask(q, self.emit)
        self.assertEqual(self.provider.prompts, [], "the model was not even asked")

    def test_remember_works_when_the_model_skips_its_tool(self):
        self.provider.reply = "preferred name | The user wants the assistant to call them sir."
        self.provider.ask("remember that i want you to call me sir", self.emit)
        self.assertIsNotNone(self.provider.store.active_by_key("user/preferred-name"))

    def test_looked_up_answers_are_kept_as_inferred(self):
        calls = [CallRecord("web_search", {"query": "eiffel"}, True, {"results": []})]
        self.provider.conversation.episode = None
        made = self.provider._learn_from_lookup("How tall is the Eiffel Tower?", calls,
                                                "The Eiffel Tower is 330 metres tall. It was built in 1889. More.")
        self.assertEqual(len(made), 1)
        memory = self.provider.store.active_by_key("learned/how-tall-is-the-eiffel-tower")
        self.assertEqual(memory.strength, "inferred", "not trained in at night until confirmed")
        self.assertEqual(memory.summary, "The Eiffel Tower is 330 metres tall. It was built in 1889.")

    def test_the_conversation_is_one_episode_and_history_reads_it(self):
        for q in ("hi", "whats my name", "ok"):
            self.provider.ask(q, self.emit)
        self.provider.close()
        rows = StoreReader(self.home / "memory.db").conversations()
        self.assertEqual([r["question"] for r in rows], ["hi", "whats my name", "ok"])
        self.assertEqual(len({r["episode_id"] for r in rows}), 1)
        self.provider.load()

    def test_profile_recall_on_a_read_only_store(self):
        self.provider._save_fact(Fact("sister", "The user's sister is called Ana."))
        store = open_read_only_store(self.home / "memory.db")
        try:
            orch = recording_orchestrator(store, None, use_tags=False, on_recall=lambda items: None)
            self.assertIn("user/sister", [m.key for m in orch.recall("trains")])
        finally:
            store.db.close()


if __name__ == "__main__":
    unittest.main()
