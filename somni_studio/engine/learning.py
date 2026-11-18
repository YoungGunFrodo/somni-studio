"""Learning during the day: memories made the moment something worth keeping is said.

Sleep still forms memories from the whole day's conversations. This makes the obvious ones
straight away, so they are recalled within the same conversation:

  * you ask it to remember something: saved, even when the model forgets to call its tool;
  * you tell it something lasting about yourself ("you can call me sir", "my sister is Ana",
    "I go to bed at eleven"): saved as a fact about you;
  * it looks something up on the web to answer you: the answer is saved, marked as inferred
    (not trained into the model at night until it is confirmed).

The model only rewrites what was said into one sentence; whether to save at all is decided here
from your own words, so a poem request can never turn into a memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..tools.catalog import ASKED_TO_REMEMBER

QUESTION = re.compile(r"^\s*(is|are|was|were|does|do|did|has|have|had|can|could|would|will|should|what|what'?s|whats|"
                      r"which|how|how'?s|who|who'?s|whos|when|where|where'?s|why|tell me|check)\b", re.I)
PERSONAL = re.compile(
    r"\b(call me|my name|i'?m called|i am called|"
    r"my (wife|husband|partner|girlfriend|boyfriend|son|daughter|kids?|children|sister|brother|mom|mum|mother|dad|"
    r"father|parents|family|dog|cat|pet|birthday|job|work|boss|favou?rite|address|email|phone|car|house|room|"
    r"schedule|allerg\w*|bedtime|alarm|doctor|team|school)|"
    r"i (like|love|hate|prefer|enjoy|dislike|work|live|was born|go to (bed|sleep|work|school)|wake up|usually|"
    r"always|never|have (a|an|two|three|four|\d))|"
    r"i'?m (a|an|allergic|from|vegetarian|vegan|married|single|retired)|i don'?t (like|eat|drink)|"
    r"i want you to|please always|from now on|don'?t ever|stop calling|start calling)\b", re.I)

LEARN_PROMPT = (
    "Decide whether the user's last message tells you a lasting fact about the user (their name or what to "
    "call them, family, health and allergies, likes, routines, home, plans) or how they want you to behave. "
    "If it does, reply with "
    "just one line in this form: sister's name | The user's sister is called Ana.\n"
    "The part before | names the one thing the fact sets, so that a later fact on the same thing replaces it: "
    "one to three words like preferred name, sister's name, bedtime, favourite food (never a broad word like "
    "family, preferences or info). The part after | states the fact in the third person and starts with "
    "\"The user\" (or, for how to behave, \"The assistant should\"). Keep names and numbers exactly as "
    "written. If it tells you no such fact, reply with exactly: NONE")
SAVE_PROMPT = (
    "The user asked you to remember something. Reply with just one line in this form: "
    "plumber visit | The plumber comes on Tuesday at 9.\n"
    "The part before | names the one thing it sets, so that a later fact on the same thing replaces it: one to "
    "three words like preferred name, sister's name, wifi password location, plumber visit (never a broad word "
    "like family or info). The part after | states it in the third person (say \"the user\" for them, "
    "\"the assistant\" for yourself). Keep names and numbers exactly as written.")
MAX_FACT_CHARS = 300
LEARN_TOKENS = 60
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|me|my|mine|myself)\b", re.I)


@dataclass(frozen=True)
class Fact:
    topic: str
    sentence: str
    personal: bool = False               # said about the user ("remember that I want you to call me sir")

    @property
    def about_user(self) -> bool:
        return self.personal or self.sentence.lower().startswith(("the user", "the assistant should"))


def is_question(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") or bool(QUESTION.match(t))


def asked_to_save(text: str) -> bool:
    """'remember that I want you to call me sir': yes. 'do you remember my name?': no (a question)."""
    return bool(ASKED_TO_REMEMBER.search(text)) and not is_question(text)


def personal_statement(text: str) -> bool:
    return bool(PERSONAL.search(text)) and not is_question(text)


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-") or "note"


TEMPLATE_WORDS = {"topic", "sentence", "fact"}


def parse_fact(text: str, required: bool) -> Fact | None:
    """'preferred name | The user wants to be called Sir.' -> Fact. NONE (or junk, unless required) -> None.
    Small models often echo the format line first ("topic | sentence"): such lines are skipped."""
    topic_hint = ""
    for line in text.replace("<|im_end|>", "").strip().splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        if line.upper().startswith("NONE"):
            return None
        topic, sep, sentence = line.partition("|")
        if not sep:
            topic, sentence = "", line
        topic, sentence = topic.strip().strip('"').strip(), sentence.strip().strip('"').strip()
        if topic.lower() == "topic" and sentence.lower() not in TEMPLATE_WORDS:
            topic_hint = sentence                     # "topic | favourite colour" on a line of its own
            continue
        if not sentence or sentence.lower() in TEMPLATE_WORDS:
            continue
        if topic.lower() in TEMPLATE_WORDS:
            topic = topic_hint
        if not required and not sentence.lower().startswith(("the user", "the assistant should")):
            continue
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        topic = topic.lower() or " ".join(sentence.split()[2:5]).lower()
        return Fact(topic[:40], sentence[:MAX_FACT_CHARS])
    return None


def extract(complete, message: str, history: list[dict], required: bool) -> Fact | None:
    """Ask the model to put what was said into one sentence (complete: messages -> text)."""
    prompt = SAVE_PROMPT if required else LEARN_PROMPT
    messages = [{"role": "system", "content": prompt}, *history[-2:], {"role": "user", "content": message}]
    fact = parse_fact(complete(messages, LEARN_TOKENS), required)
    if fact is not None and FIRST_PERSON.search(message):
        fact = Fact(fact.topic, fact.sentence, personal=True)
    return fact


WEATHER = re.compile(r"\b(weather|forecast|rain\w*|snow\w*|sunny|cloudy|temperature outside|how (hot|cold|warm) "
                     r"is it|umbrella|jacket|coat|storm\w*|wind(y)?)\b", re.I)
LIVES_IN = re.compile(r"\b(?:lives|live|living|is based|located|resides) in ([a-z][\w .'-]*(?:, ?[a-z][\w .'-]*)?)",
                      re.I)
PLACE_IN_QUESTION = re.compile(r"\b(?:in|at|for|near) ((?!the\b|my\b|this\b|tomorrow\b|today\b)[a-z][\w.'-]*"
                               r"(?: [a-z][\w.'-]*)?(?:, ?[a-z][\w .'-]*)?)", re.I)
WHEN_WORDS = {"today", "tomorrow", "tonight", "morning", "afternoon", "evening", "weekend", "now"}


def weather_call(question: str) -> dict | None:
    """A weather question goes straight to the weather tool: measured, a 4B model asked "will it rain
    tomorrow in Chicago?" searched Wikipedia three times instead (85 s, no answer)."""
    if not WEATHER.search(question):
        return None
    found = PLACE_IN_QUESTION.search(question)
    words = found.group(1).rstrip("?.!").split() if found else []
    place = " ".join(w for w in words if w.lower().strip(",") not in WHEN_WORDS).strip(" ,")
    return {"name": "weather", "arguments": {"place": place}} if place else None


def fallback_call(question: str) -> dict | None:
    """When Somni says it is not sure: the tool call that can answer, or None (a question about the
    user, a statement, small talk: the web cannot know what the user's dog is called)."""
    if WEATHER.search(question):
        return weather_call(question) or {"name": "weather", "arguments": {}}
    if not is_question(question) or FIRST_PERSON.search(question) or len(question.split()) < 3:
        return None
    return {"name": "web_search", "arguments": {"query": question.strip().rstrip("?")}}


def home_place(summaries: list[str]) -> str:
    """'The user lives in Denver, Colorado.' -> 'Denver, Colorado'."""
    for text in summaries:
        found = LIVES_IN.search(text)
        if found:
            return found.group(1).strip().rstrip(".")
    return ""


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])[:MAX_FACT_CHARS]
