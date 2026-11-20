"""The two ways of answering.

Every question goes: recall (Somni's own, engine/recall.py) → does it need a tool (engine/agent.py)?
  * yes: the tool runs and the model answers from its result;
  * no:  Somni answers the normal way (the in-process model with its experts and memory).

Only the in-process model exposes its insides: with a server the brain tower stays dark, because
an API does not show its layers.

Read-only mode recalls and answers but never logs an episode, so pointing the studio at someone's
Somni run cannot change it. Writable mode (the default for the studio's own home) logs each
conversation as one episode (engine/conversation.py) for the next sleep to learn from, and saves
the obvious memories right away (engine/learning.py).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..events import EventType
from ..graph.memory_graph import file_id, memory_id
from ..settings import StudioSettings
from ..somni.home import ExpertSet, HomePaths, base_model, find_experts
from ..somni.runtime import ensure_sleepai
from ..somni.store_reader import somni_today
from ..tools.catalog import ToolContext, abilities, available_tools, continued_request, ready_checker, relevant_tools
from .agent import LocalBackend, ServerBackend, ToolAgent
from .conversation import Conversation, with_history
from .learning import (
    Fact,
    asked_to_save,
    extract,
    fallback_call,
    first_sentences,
    home_place,
    is_question,
    personal_statement,
    slug,
    weather_call,
)
from .recall import PROFILE_TAG, classify, open_read_only_store, recording_orchestrator
from .types import ChatOutcome, Emitter, RecallItem

log = logging.getLogger(__name__)


def cited_node(used: str) -> str | None:
    """Graph node of what an answer cited: a memory, or a memory file (recall puts files in the
    context as pseudo-memories with ids like "file:111", and models cite those too)."""
    if not used:
        return None
    if used.startswith("file:") and used[5:].isdigit():
        return file_id(int(used[5:]))
    if used.startswith(LIVE_PREFIX):
        return None                                   # a live note, not a node in the graph
    return memory_id(used)


LIVE_PREFIX = "live:"


def systems_sentence(info: dict) -> str:
    """Somni's systems in one sentence (the model rarely thinks to call a tool for "how many memories?")."""
    from ..web_text import model_name
    parts = []
    if info.get("model"):
        parts.append(f"thinks with the {model_name(info['model'])} model")
    if "memories" in info:
        parts.append(f"has {info['memories']} memories and {info.get('learned_experts', 0)} learned experts")
    if "not_yet_learned_in_sleep" in info:
        parts.append(f"has {info['not_yet_learned_in_sleep']} conversations waiting to be learned in sleep")
    for key in ("voice", "listening", "sleep", "graphics_card"):
        if info.get(key):
            parts.append(f"{key.replace('_', ' ')}: {info[key]}")
    return "Right now Somni " + "; ".join(parts) + "."


def stored(memory_id: str) -> bool:
    return bool(memory_id) and not str(memory_id).startswith(LIVE_PREFIX)


@dataclass
class LiveNote:
    """Something true right now, put in front of the model like a memory (but never stored)."""
    key: str
    summary: str
    id: str = ""
    type: str = "fact"
    status: str = "active"
    strength: str = "verified"
    value: str = ""
    tags: list = field(default_factory=list)

    def __post_init__(self):
        self.id = self.id or LIVE_PREFIX + self.key


LEFTOVER_CITATION = re.compile(r"\s*\[memory:[^\]]*\]", re.I)
MEMORY_LINE = re.compile(r"\[(?:active|disputed|superseded), (?:verified|user-confirmed|inferred)\]\s*[\w/.-]+:\s*",
                         re.I)
ACK_PROMPT = ("The user just told you something and you saved it to memory: \"{fact}\" Reply with one short, "
              "warm sentence that shows you understood. Talk to the user directly. Do not quote the memory, list "
              "anything or ask a question.")
ACK_TOKENS = 40
# Where what a tool found out is kept: (key, tags). Not "user/...": those are recalled every time.
TOOL_FINDINGS = {"devices_find": ("home/devices-on-network", ["home"])}


def strip_citations(text: str) -> str:
    """Somni strips a valid [memory: key] citation; a small model sometimes invents one with a key
    that does not exist ("[memory: somni/remembered-fact: ...]"), or reads a memory line back
    ("[active, user-confirmed] user/home: ..."). Never show or speak those."""
    return MEMORY_LINE.sub("", LEFTOVER_CITATION.sub("", text)).strip()


def memory_messages(question: str, context: list | None, history: list[dict]) -> list[dict]:
    """Somni's prompt (its memory preamble in a system message, then the question) with the
    conversation so far in between."""
    from sleepai.models.llm import MEMORY_PREAMBLE
    messages = [{"role": "user", "content": question}]
    if context:
        lines = [f"- [{m.status}, {m.strength}] {m.key}: {m.summary}" for m in context]
        messages.insert(0, {"role": "system", "content": MEMORY_PREAMBLE + "\n" + "\n".join(lines)})
    return with_history(messages, history)


def conversational_hf_student(somni_hf):
    """Somni's HFStudent, hearing the last few turns (set .history before answering). Somni's own
    prompt builder is single-turn on purpose: training and evaluation use it unchanged."""
    class ConversationalHFStudent(somni_hf.HFStudent):
        history: list[dict] = []

        def _prompts(self, items):
            _, tok = somni_hf._load(self.base_model, self.adapter_dir)
            return [tok.apply_chat_template(memory_messages(q, context, self.history), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
                    for q, context in items]
    return ConversationalHFStudent


def conversational_llm_student():
    from sleepai.models.base import ABSTAIN, Answer, normalize
    from sleepai.models.llm import LLMStudent, chat, cited_memory

    class ConversationalLLMStudent(LLMStudent):
        history: list[dict] = []

        def answer(self, question: str, context=None) -> Answer:
            text = chat(self.url, memory_messages(question, context, self.history), model=self.model)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            text, used = cited_memory(text, context)
            value = ABSTAIN if "not sure" in text.lower() else normalize(text)
            caveat = bool(used) and any(m.id == used and m.status == "disputed" for m in context or [])
            return Answer(text, value, 1.0, used_memory=used, caveat=caveat)
    return ConversationalLLMStudent


class ProviderError(RuntimeError):
    """A provider cannot run with the current settings (message is shown to the user)."""


@dataclass
class ToolEnvironment:
    """What tools reach outside the model: connections, and a way to start a sleep."""
    connections: object                               # integrations.service.ConnectionsService
    request_sleep: Callable[[], None] | None = None
    conversation: Conversation = field(default_factory=Conversation)   # outlives model reloads
    devices: object = None                            # devices.hub.DeviceHub
    system_info: Callable | None = None               # () -> dict: Somni's own systems
    say_on: Callable | None = None                    # (device, message): speak it on that speaker


class _RecallingProvider:
    """Shared plumbing: the memory store, the instrumented orchestrator, tools, the ask flow."""

    name = "base"

    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings
        self.paths = HomePaths(settings.somni.home_path)
        self.read_only = settings.somni.read_only
        self.store = None
        self.student = None
        self.tool_env: ToolEnvironment | None = None
        self._emit: Emitter | None = None
        self._orchestrator = None
        self._own_conversation = Conversation()

    @property
    def conversation(self) -> Conversation:
        return self.tool_env.conversation if self.tool_env is not None else self._own_conversation

    # subclasses -------------------------------------------------
    def _make_student(self):
        raise NotImplementedError

    def _tool_backend(self):
        raise NotImplementedError

    def _before_answer(self, emit: Emitter) -> None:
        pass

    def _after_answer(self) -> None:
        pass

    # lifecycle --------------------------------------------------
    def load(self) -> None:
        ensure_sleepai(self.settings.somni.repo_path)
        db = self.paths.memory_db
        if self.read_only:
            if not db.is_file():
                raise ProviderError(f"no memory store at {db} (read-only mode never creates one)")
            self.store = open_read_only_store(db)
        else:
            from sleepai.memory.store import MemoryStore
            self.store = MemoryStore(db)
        self.student = self._make_student()
        self._orchestrator = recording_orchestrator(self.store, self.student, self.settings.somni.use_tags,
                                                    self._on_recall)

    def close(self) -> None:
        if self.store is not None:
            self.store.db.close()
        self.store = self.student = self._orchestrator = None

    def _on_recall(self, items: list[RecallItem]) -> None:
        if self._emit is not None:
            self._emit(EventType.RECALL, items=[i.to_wire() for i in items])

    def _today(self) -> int:
        row = self.store.db.execute("SELECT MAX(COALESCE((SELECT MAX(day) FROM episodes), 0), "
                                    "COALESCE((SELECT MAX(created_day) FROM memories), 0))").fetchone()
        return somni_today(int(row[0] or 0))

    # tools ------------------------------------------------------
    def _tool_context(self) -> ToolContext | None:
        if self.tool_env is None or not self.settings.tools.enabled:
            return None
        env = self.tool_env
        writable = not self.read_only
        return ToolContext(
            integration_config=env.connections.config,
            integration_ready=ready_checker(env.connections),
            integration_details=env.connections.details,
            allowed_domains=self.settings.tools.domains,
            remember=self._remember if writable else None,
            request_sleep=env.request_sleep if writable else None,
            search_memory=self._search_memory,
            learn_place=self._learn_place if writable else None,
            devices=env.devices, say_on=env.say_on, system_info=env.system_info)

    def _search_memory(self, query: str) -> list[dict]:
        found = self.store.search(query, k=6)
        if found:
            self._on_recall(classify(found, set()))          # light up what the search found
        return [{"key": m.key, "summary": m.summary} for m in found]

    def _remember(self, fact: str) -> str:
        """The remember tool: a memory now (user-confirmed: the user asked for it), not an episode
        that would only become one at the next sleep."""
        text = fact.strip()
        about = re.sub(r"^the user'?s?\s+", "", text, flags=re.I)
        return self._save_fact(Fact(" ".join(about.split()[:4]), text))

    def _save_fact(self, fact: Fact, strength: str = "user-confirmed", kind: str = "",
                   place: tuple[str, list[str]] | None = None) -> str | None:
        """Save (or update) a memory. Facts about the user are profile memories, recalled every time;
        a newer statement on the same topic replaces the older one ("call me Bob" after "call me sir")."""
        from sleepai.memory.lint import lint_memory
        from sleepai.memory.store import Memory
        day = self._today()
        if place:                                     # a fixed key: each update replaces the last
            key, tags = place
        elif kind:
            key, tags = f"{kind}/{slug(fact.topic, 48)}", [kind]
        elif fact.about_user:
            # Where the user lives has one key, however it was phrased, so a correction replaces it.
            topic = "home" if home_place([fact.sentence]) else fact.topic
            key, tags = f"user/{slug(topic)}", [PROFILE_TAG]
        else:
            key, tags = f"note/{slug(fact.topic)}", ["note"]
        episode = self.conversation.episode
        sources = ([f"episode:{episode}"] if episode else []) + [strength if not kind or place else "web"]
        memory = Memory(key=key, type="fact", summary=fact.sentence, value=fact.sentence, strength=strength,
                        sources=sources, tags=tags, created_day=day, last_verified_day=day)
        problems = lint_memory(memory)
        if problems:
            log.info("not saving %s: %s", key, "; ".join(problems))
            return None
        existing = self.store.active_by_key(key)
        if existing is None:
            return self.store.insert(memory).id
        if existing.summary == fact.sentence:
            return existing.id
        return self.store.supersede(existing.id, memory).id

    def _learn_from_user(self, question: str, history: list[dict]) -> list[tuple[str, str]]:
        """Before answering: what you asked it to remember, or said about yourself, becomes a memory
        now, so the answer already knows it ("you can call me sir" -> "Okay, Sir.")."""
        if self.read_only or not (asked_to_save(question) or personal_statement(question)):
            return []
        try:
            fact = extract(self._tool_backend().complete, question, history, required=asked_to_save(question))
        except Exception as exc:                              # learning must never break an answer
            log.warning("could not learn from %r: %s", question[:60], exc)
            return []
        if fact is not None and self._save_fact(fact):
            return [(fact.sentence, "remembered")]
        return []

    def _live_notes(self) -> list:
        """What is true right now (what it is connected to and can do): in every answer's context."""
        ctx = self._tool_context()
        if ctx is None:
            return []
        notes = []
        try:
            notes.append(LiveNote("somni/abilities-right-now", abilities(ctx)))
            if ctx.system_info is not None:
                notes.append(LiveNote("somni/systems-right-now", systems_sentence(ctx.system_info())))
        except Exception as exc:                          # a note must never break an answer
            log.warning("could not describe abilities: %s", exc)
        return notes

    def _learn_place(self, name: str) -> None:
        """Where the computer is, kept as an inferred memory, unless the user has said where they live
        (what they said always wins; saying it later replaces this)."""
        if self.store.active_by_key("user/home") is None:
            self._save_fact(Fact("home", f"The user lives in {name}."), strength="inferred")

    def _learn_from_tools(self, calls: list) -> list[tuple[str, str]]:
        """Facts a tool found out ("The user's home network has: ..."): kept as memories."""
        made = []
        for record in calls:
            sentence = record.result.get("learned") if record.ok and isinstance(record.result, dict) else None
            if sentence and not self.read_only:
                # One memory per kind of finding, replaced when it changes: a scan used to make a new
                # "the user's network has ..." memory each time (the order changed), filed as a fact
                # about the user and so put into every prompt.
                place = TOOL_FINDINGS.get(record.name, (f"home/{slug(record.name)}", ["home"]))
                saved = self._save_fact(Fact(record.name, sentence), strength="verified", place=place)
                if saved:
                    made.append((sentence, "learned"))
                    old = self.store.active_by_key(f"user/{slug('home devices')}")
                    if record.name == "devices_find" and old is not None and "network has" in old.summary:
                        self.store.update_fields(old.id, status="superseded", superseded_by=saved)
        return made

    def _learn_from_lookup(self, question: str, calls: list, answer: str) -> list[tuple[str, str]]:
        """After answering: what it looked up on the web for you is kept (as inferred)."""
        looked_up = any(r.ok and r.name in ("web_search", "web_read") for r in calls)
        if self.read_only or not looked_up or not answer or "not sure" in answer.lower():
            return []
        sentence = first_sentences(answer)
        if self._save_fact(Fact(question, sentence), strength="inferred", kind="learned"):
            return [(sentence, "learned")]
        return []

    @staticmethod
    def _announce(made: list[tuple[str, str]], emit: Emitter, start: int = 0) -> list[dict]:
        """Learned memories show like a tool call (chip in the chat, beam in the brain)."""
        chips = []
        for i, (sentence, label) in enumerate(made, start):
            call_id = f"learn_{i}"
            emit(EventType.TOOL_CALL, call_id=call_id, name="memory_remember", arguments={"fact": sentence})
            emit(EventType.TOOL_RESULT, call_id=call_id, name="memory_remember", ok=True,
                 summary=f"{label}: {sentence}")
            chips.append({"name": "memory_remember", "ok": True, "summary": f"{label}: {sentence}"})
        return chips

    def _try_tools(self, question: str, context: list, emit: Emitter, history: list[dict],
                   remembered: bool = False, forced: list[dict] | None = None):
        ctx = self._tool_context()
        if ctx is not None:
            ctx.question = continued_request(question, history)
            if remembered:
                ctx.remember = None                           # already saved: do not offer it again
        tools = relevant_tools(available_tools(ctx), question, history) if ctx else []
        names = {t.name for t in tools}
        if forced is None and "weather" in names and not personal_statement(question):
            call = weather_call(question)
            forced = [call] if call else None
        if not tools or (forced and not all(c["name"] in names for c in forced)):
            return None
        agent = ToolAgent(self._tool_backend(), ctx, emit, self.settings.tools.max_rounds)
        return agent.run(question, context, tools, history, forced)

    # answering --------------------------------------------------
    def ask(self, question: str, emit: Emitter) -> ChatOutcome:
        if self._orchestrator is None:
            raise ProviderError("the model is not loaded")
        self._emit = emit
        self._before_answer(emit)
        history = self.conversation.history()
        try:
            learned = self._learn_from_user(question, history)
            chips = self._announce(learned, emit)
            context = self._orchestrator.recall(question)     # after learning: the new memory is in it
            context = self._live_notes() + context
            result = self._try_tools(question, context, emit, history, remembered=bool(learned))
            if result is not None:
                outcome, calls = self._tool_outcome(question, context, result), result.calls
                self._keep_lesson(question, history, result)
            elif learned and not is_question(question):
                outcome, calls = self._acknowledge(question, context, history, learned[0][0], emit), []
            else:
                outcome, calls = self._answer(question, context, history, log=False), []
                retry = self._look_it_up(question, context, emit, history, outcome)
                if retry is not None:
                    outcome, calls = self._tool_outcome(question, context, retry), retry.calls
                elif not self.read_only:
                    outcome.episode_id = self._log_turn(question, context, outcome.text)
            later = self._learn_from_lookup(question, calls, outcome.text) + self._learn_from_tools(calls)
            chips += self._announce(later, emit, len(chips))
            outcome.tools = list(outcome.tools) + chips
        finally:
            self._after_answer()
            self._emit = None
        return outcome

    def _keep_lesson(self, question: str, history: list[dict], result) -> None:
        """Every tool turn, with whether it really worked, kept for sleep (engine/lessons.py)."""
        if self.read_only:
            return
        from ..tools.catalog import BY_NAME
        from . import lessons
        lessons.record(self.paths.home, question, history, result, {n for n, t in BY_NAME.items() if t.acts},
                       self._today())

    def _acknowledge(self, question: str, context: list, history: list[dict], fact: str,
                     emit: Emitter) -> ChatOutcome:
        """You told it something about yourself: a short, natural reply. (Somni's own answer path is
        trained to answer questions from memory; given a statement it says "I'm not sure" or reads
        the memory line back.)"""
        prompt = ACK_PROMPT.format(fact=fact)
        try:
            text = self._tool_backend().complete([{"role": "system", "content": prompt}, *history[-2:],
                                                  {"role": "user", "content": question}], ACK_TOKENS)
        except Exception as exc:                              # never fail an answer over politeness
            log.warning("acknowledgement failed: %s", exc)
            text = ""
        text = strip_citations(text).split("\n")[0].strip() or "Got it. I'll remember that."
        emit(EventType.TOKEN, text=text, replace=True)
        episode = None if self.read_only else self._log_turn(question, context, text)
        return ChatOutcome(text, None, False, episode, list(self._orchestrator.last_recall))

    def _look_it_up(self, question: str, context: list, emit: Emitter, history: list[dict], outcome: ChatOutcome):
        """Somni did not know: answer with the tool that can (the weather, a web search) instead of
        "I'm not sure", the way a person would look it up."""
        call = fallback_call(question) if "not sure" in outcome.text.lower() else None
        if call is None:
            return None
        emit(EventType.TOKEN, text="", replace=True)          # take back the "I'm not sure"
        result = self._try_tools(question, context, emit, history, forced=[call])
        if result is None:                                     # that tool is not connected
            emit(EventType.TOKEN, text=outcome.text, replace=True)
        return result

    def _answer(self, question: str, context: list, history: list[dict], log: bool = True) -> ChatOutcome:
        """Somni's own answer (experts and memory), with the conversation so far."""
        from sleepai.models.base import ABSTAIN
        self.student.history = history
        try:
            answer = self.student.answer(question, context)
        finally:
            self.student.history = []
        episode = None
        if not self.read_only:
            day = self._today()
            episode = self._log_turn(question, context, answer.text) if log else None
            if stored(answer.used_memory):    # only the memory the answer actually used is strengthened
                self.store.touch([answer.used_memory], day)
            if answer.value == ABSTAIN and not answer.used_memory:
                self.store.add_open_question(question, day=day)
        return ChatOutcome(strip_citations(answer.text), cited_node(answer.used_memory), answer.caveat, episode,
                           list(self._orchestrator.last_recall))

    def _log_turn(self, question: str, context: list, answer: str) -> int:
        turns = [{"role": "user", "content": question}]
        kept = [m.id for m in context if stored(m.id)]
        if kept:
            turns.append({"role": "memory", "content": ", ".join(kept)})
        turns.append({"role": "assistant", "content": answer})
        return self.conversation.log(self.store, turns, day=self._today())

    def _tool_outcome(self, question: str, context: list, result) -> ChatOutcome:
        from sleepai.models.llm import cited_memory
        text, used = cited_memory(result.text, context)
        text = strip_citations(text)
        episode = None
        if not self.read_only:
            # Logged like any conversation; tool output itself is not stored (it can hold private
            # mail headers and changes by the minute), only what was said.
            episode = self._log_turn(question, context, text)
            if stored(used):
                self.store.touch([used], self._today())
        return ChatOutcome(text, cited_node(used), False, episode, list(self._orchestrator.last_recall),
                           tools=[{"name": r.name, "ok": r.ok, "summary": r.summary()} for r in result.calls])

    def describe(self) -> dict:
        return {"provider": self.name, "read_only": self.read_only, "home": str(self.paths.home)}


class SomniLocalProvider(_RecallingProvider):
    """Somni's in-process student (Hugging Face model + stacked experts) with live telemetry."""

    name = "somni"

    def __init__(self, settings: StudioSettings) -> None:
        super().__init__(settings)
        self.model_name = base_model(self.paths)
        self.experts: ExpertSet = find_experts(self.paths, settings.somni.adapter_path)
        self.telemetry = None
        self.dtype = None
        self._model = self._tok = None

    def _make_student(self):
        from sleepai.models import hf as somni_hf

        from .precision import resolve_dtype, somni_load_dtype
        from .telemetry import Telemetry
        student = conversational_hf_student(somni_hf)(self.model_name,
                                                      max_new_tokens=self.settings.provider.max_new_tokens,
                                                      batch_size=1)
        student.adapter_dir = str(self.experts.directory) if self.experts.directory else None
        # HFStudent loads lazily through this module-level cache; loading here (same key) makes
        # the first question fast and gives the telemetry the exact model object wake uses.
        from . import gpu_memory
        self.dtype = resolve_dtype(self.settings.provider.dtype)
        gpu_memory.cap()
        with somni_load_dtype(self.dtype):
            self._model, self._tok = somni_hf._load(student.base_model, student.adapter_dir)
        # Every generate() on this model reads its prompt in pieces, also the framework's own answer
        # path (HFStudent.answer_many), not only the tool loop (agent.PREFILL_CHUNK explains why).
        from .agent import PREFILL_CHUNK
        self._model.generation_config.prefill_chunk_size = PREFILL_CHUNK
        self.telemetry = Telemetry(self._model, self._tok, modules_per_layer=len(self.experts.targets))
        self.telemetry.attach()
        return student

    def _tool_backend(self):
        return LocalBackend(self._model, self._tok, self.telemetry, self.settings.provider.max_new_tokens)

    def _before_answer(self, emit: Emitter) -> None:
        self.telemetry.begin(emit)

    def _after_answer(self) -> None:
        from . import gpu_memory
        self.telemetry.end()
        gpu_memory.release()

    def close(self) -> None:
        if self.telemetry is not None:
            self.telemetry.detach()
            self.telemetry = None
        self._model = self._tok = None
        super().close()
        try:
            from sleepai.models import hf as somni_hf
            somni_hf.clear_cache()
        except ImportError:
            pass

    def describe(self) -> dict:
        return {**super().describe(), "model": self.model_name, "experts": len(self.experts.experts),
                "dtype": str(self.dtype).replace("torch.", "") if self.dtype is not None else None,
                "experts_source": self.experts.source,
                "experts_dir": self.experts.directory.name if self.experts.directory else None}


class OpenAICompatProvider(_RecallingProvider):
    """Any OpenAI-compatible server (LM Studio, llama-server) through Somni's LLMStudent."""

    name = "openai"

    def _make_student(self):
        return conversational_llm_student()(self.settings.provider.url, self.settings.provider.model)

    def _tool_backend(self):
        return ServerBackend(self.settings.provider.url, self.settings.provider.model,
                             self.settings.provider.max_new_tokens)

    def ask(self, question: str, emit: Emitter) -> ChatOutcome:
        outcome = super().ask(question, emit)
        emit(EventType.TOKEN, text=outcome.text, replace=True)     # servers here are not streamed
        return outcome

    def describe(self) -> dict:
        return {**super().describe(), "model": self.settings.provider.model or "(server default)",
                "url": self.settings.provider.url}


PROVIDERS = {"somni": SomniLocalProvider, "openai": OpenAICompatProvider}


def make_provider(settings: StudioSettings):
    return PROVIDERS[settings.provider.kind](settings)
