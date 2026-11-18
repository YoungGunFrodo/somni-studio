"""Who the assistant is: a few core memories every home starts with.

Without them a fresh model answers "I am Qwen, made by Alibaba": it has no idea it is Somni, that
it runs in this studio, or that it can remember things. These are ordinary Somni memories
(user-confirmed, so sleep trains them in like any other), tagged "identity". Recall always puts
them in the model's context (engine/recall.py), whatever the question.

ensure_identity() adds any that are missing and keeps the "which model am I running on" memory
current; it never overwrites one the user has changed (a memory with the same key but a
different, non-studio source is left alone).
"""
from __future__ import annotations

from .home import HomePaths, base_model

IDENTITY_TAG = "identity"
SOURCE = "studio:identity"


def _model_name(model: str) -> str:
    """A readable name for a model id or a local snapshot folder."""
    text = model.replace("\\", "/")
    if "/snapshots/" in text and "models--" in text:
        text = text.split("models--", 1)[1].split("/snapshots/", 1)[0].replace("--", "/")
    return text


def core_memories(model: str) -> list[tuple[str, str, str]]:
    """(key, summary, value) for each core memory."""
    name = _model_name(model)
    return [
        ("somni/name", "The assistant's name is Somni.", "Somni"),
        ("somni/what-it-is",
         "Somni is a personal assistant for the user's home and schedule. It runs on the user's own computer, "
         "remembers conversations and learns from them while it sleeps.",
         "a personal home assistant that remembers and learns while it sleeps"),
        ("somni/creator",
         "Somni is built from Somni Studio (its interface) and the Somni sleep-learning memory framework, both "
         "open source. The language model underneath is made by someone else and can be swapped; the "
         "memories stay.",
         "Somni Studio and the Somni framework, open source; the model can be swapped"),
        ("somni/base-model", f"Somni currently thinks with the {name} language model.", name),
        ("somni/connections",
         "When the user asks, Somni can look for smart lights, plugs, speakers, displays and TVs on the home "
         "network and connect them. Web search, email and other services are set up in the Connect tab of "
         "Somni Studio. What is actually connected right now is in the 'abilities right now' note.",
         "the Connect tab in Somni Studio"),
        ("somni/style",
         "Somni speaks like a polished, unflappable AI butler: calm, concise and courteous, with a dry wit. It "
         "addresses the user as sir, answers exactly what was asked in a sentence or two, uses no emojis and "
         "does not pad replies with offers of help.",
         "a calm, concise AI butler with a dry wit; calls the user sir; no emojis"),
        ("somni/memory",
         "When the user asks Somni to remember something, Somni saves it as a memory right away. Memories "
         "are recalled when they are relevant, and each night's sleep trains them into Somni's model.",
         "saves memories when asked; recalls them; learns them during sleep"),
    ]


CONNECTION_TAGS = ["somni", "connections"]


def note_connection(paths: HomePaths, thing: str, sentence: str) -> None:
    """A memory of what Somni is connected to, made when something is connected or disconnected (from
    chat or the Connect tab), so it knows and, after sleep, has learned what it is hooked up to."""
    import re

    from sleepai.memory.store import Memory, MemoryStore
    key = "somni/connected/" + (re.sub(r"[^a-z0-9]+", "-", thing.lower()).strip("-")[:48] or "thing")
    store = MemoryStore(paths.memory_db)
    try:
        existing = store.active_by_key(key)
        if existing is not None and existing.summary == sentence:
            return
        memory = Memory(key=key, type="fact", summary=sentence, value=sentence, strength="verified",
                        sources=[SOURCE], tags=CONNECTION_TAGS)
        if existing is None:
            store.insert(memory)
        else:
            store.supersede(existing.id, memory)
    finally:
        store.db.close()


def ensure_identity(paths: HomePaths) -> int:
    """Add missing core memories (and refresh the base-model one). Returns how many changed."""
    from sleepai.memory.store import Memory, MemoryStore
    store = MemoryStore(paths.memory_db)
    changed = 0
    try:
        for key, summary, value in core_memories(base_model(paths)):
            existing = store.active_by_key(key)
            if existing is not None:
                if SOURCE not in existing.sources or existing.summary == summary:
                    continue                     # the user's own version, or already current
                memory = Memory(key=key, type="fact", summary=summary, value=value, strength="user-confirmed",
                                sources=[SOURCE], tags=[IDENTITY_TAG, "somni"])
                store.supersede(existing.id, memory)
            else:
                store.insert(Memory(key=key, type="fact", summary=summary, value=value, strength="user-confirmed",
                                    sources=[SOURCE], tags=[IDENTITY_TAG, "somni"]))
            changed += 1
    finally:
        store.db.close()
    return changed
