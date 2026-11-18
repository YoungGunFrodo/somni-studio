"""Tool calling: the assistant decides whether a question needs a tool, calls it, and answers from
what came back.

    question ─► step: does the model start a tool call?
                 ├─ no  ─► (local model) Somni answers as usual, with its experts and memory
                 │         (server)      the server's reply is the answer
                 └─ yes ─► run the tool(s) ─► step again with the results ─► … ─► the answer

Local model: the decision runs the base model (experts off; they are trained on facts, not on
tool use) with the model's own chat template, which lists the tools. Generation stops after a few
tokens when the reply is clearly not a tool call, so ordinary questions pay only for reading the
prompt. Tool calls are parsed from the Hermes/Qwen format: <tool_call>{"name": ..., "arguments": ...}</tool_call>.

Server: OpenAI-compatible tool calling (LM Studio, llama-server with --jinja).
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from ..events import EventType
from ..tools.catalog import BY_NAME, CallRecord, Tool, ToolContext, run_tool
from .conversation import with_history
from .types import Emitter

log = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
TRAILING_JUNK_RE = re.compile(r"(?<=\})[^}]*$", re.S)           # an unclosed block: drop what follows the JSON
XML_FUNCTION_RE = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.S)
XML_PARAMETER_RE = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.S)
END_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")
DECIDE_TOKENS = 200
SERVER_TIMEOUT_S = 120

SYSTEM = ("You are Somni, a helpful assistant that lives on the user's own computer and remembers what you "
          "talk about. Use a tool whenever the user asks about or wants to change something a tool covers: "
          "the date or time; their smart-home devices (whether something is on or off, a temperature, turning "
          "things on or off); their email; remembering something they explicitly ask you to remember; going "
          "to sleep when they ask you to; the weather; looking something up when they ask or when you do not "
          "know a fact or it may have changed recently (news, prices, sports, opening hours). "
          "Never guess the state of a device: check it with a tool. "
          "When you need something you do not know (where the user is, a name, a fact), work it out before "
          "asking: look in your memories, then ask this computer (computer_info), then search the web. Ask the "
          "user only when none of those can tell you. When a tool says it needs something, get it and then call "
          "that tool again: finish what the user asked before answering. Never say something happened "
          "unless a tool result shows it did; if a tool failed, say plainly what failed. "
          "For anything else, just answer. After using a tool, reply in one or two short spoken sentences.")


@dataclass
class Step:
    calls: list[dict] = field(default_factory=list)     # [{"name": str, "arguments": dict}]
    text: str | None = None                             # the reply, when there was no tool call


class Backend(Protocol):
    def step(self, messages: list[dict], tools: list[dict], first: bool) -> Step | None:
        """One model turn. On the first turn a backend may return None for "no tool needed; answer
        the normal way" instead of a reply."""


def _value(raw: str):
    """A parameter value from the XML format: numbers, booleans and JSON stay typed."""
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    return value if isinstance(value, (int, float, bool, list, dict)) or value is None else text


def parse_tool_calls(text: str) -> list[dict]:
    """Tool calls in either format the Qwen family uses:
        JSON  <tool_call>{"name": "home_states", "arguments": {"search": "fan"}}</tool_call>    (Qwen3)
        XML   <tool_call><function=home_states><parameter=search>fan</parameter></function></tool_call>   (Qwen3.5)
    """
    calls = []
    for block in TOOL_BLOCK_RE.findall(text):
        block = block.strip()
        if block.startswith("{"):
            try:
                data = json.loads(TRAILING_JUNK_RE.sub("", block))
            except json.JSONDecodeError:
                continue
            name = data.get("name")
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(name, str):
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
            continue
        for name, body in XML_FUNCTION_RE.findall(block):
            calls.append({"name": name, "arguments": {k: _value(v) for k, v in XML_PARAMETER_RE.findall(body)}})
    return calls


def clean_reply(text: str) -> str:
    for marker in END_MARKERS:
        text = text.replace(marker, "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def could_be_tool_call(text: str) -> bool:
    t = text.lstrip()
    return t.startswith(TOOL_CALL_OPEN) or TOOL_CALL_OPEN.startswith(t)


# ---------------------------------------------------------------- local model
class LocalBackend:
    """The in-process model, through its chat template's tool support."""

    def __init__(self, model, tokenizer, telemetry=None, max_new_tokens: int = 256) -> None:
        self.model, self.tok, self.telemetry = model, tokenizer, telemetry
        self.max_new_tokens = max_new_tokens

    def _generate(self, prompt: str, max_new: int, stop_unless_tool: bool) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        start = enc.input_ids.shape[1]
        tok = self.tok

        class NotAToolCall(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tok.decode(input_ids[0, start:], skip_special_tokens=False)
                return not could_be_tool_call(text)

        stops = StoppingCriteriaList([NotAToolCall()]) if stop_unless_tool else None
        ectx = getattr(self.model, "_expert_ctx", None)
        if ectx is not None:
            ectx["off"], ectx["route"] = True, None          # the base model reads and writes tool calls
        try:
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id, stopping_criteria=stops)
        finally:
            if ectx is not None:
                ectx["off"] = False
        return self.tok.decode(out[0, start:], skip_special_tokens=False)

    def step(self, messages: list[dict], tools: list[dict], first: bool) -> Step | None:
        prompt = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        if first:
            if self.telemetry is not None:
                self.telemetry.mute_tokens = True             # the decision is not part of the answer
            try:
                text = self._generate(prompt, DECIDE_TOKENS, stop_unless_tool=True)
            finally:
                if self.telemetry is not None:
                    self.telemetry.mute_tokens = False
            calls = parse_tool_calls(text)
            return Step(calls=calls) if calls else None
        if self.telemetry is not None:                        # stream the reply unless it is a tool call
            self.telemetry.restart_text(hold=could_be_tool_call)
        text = self._generate(prompt, self.max_new_tokens, stop_unless_tool=False)
        calls = parse_tool_calls(text)
        return Step(calls=calls) if calls else Step(text=clean_reply(text))

    def complete(self, messages: list[dict], max_new: int) -> str:
        """A short plain reply from the base model, kept out of the live answer (no streaming)."""
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        if self.telemetry is not None:
            self.telemetry.mute_tokens = True
        try:
            return clean_reply(self._generate(prompt, max_new, stop_unless_tool=False))
        finally:
            if self.telemetry is not None:
                self.telemetry.mute_tokens = False


# ---------------------------------------------------------------- server
def _openai_messages(messages: list[dict]) -> list[dict]:
    """The OpenAI API wants tool-call arguments as a JSON string; chat templates (Qwen3.5) want a
    dict. Messages keep dicts; this makes the string form for servers."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{**c, "function": {**c["function"],
                                                        "arguments": json.dumps(c["function"]["arguments"])}}
                                     for c in m["tool_calls"]]}
        out.append(m)
    return out


class ServerBackend:
    """An OpenAI-compatible server's native tool calling."""

    def __init__(self, url: str, model: str = "", max_new_tokens: int = 256) -> None:
        self.url, self.model, self.max_new_tokens = url.rstrip("/"), model, max_new_tokens

    def step(self, messages: list[dict], tools: list[dict], first: bool) -> Step | None:
        payload = {"messages": _openai_messages(messages), "tools": tools, "tool_choice": "auto", "temperature": 0.2,
                   "max_tokens": self.max_new_tokens}
        if self.model:
            payload["model"] = self.model
        req = urllib.request.Request(  # noqa: S310 - the server the user configured
            self.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=SERVER_TIMEOUT_S) as r:  # noqa: S310 - the configured server
            message = json.loads(r.read())["choices"][0]["message"]
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args if isinstance(args, dict) else {}})
        if not calls:
            calls = parse_tool_calls(message.get("content") or "")    # some servers leave them in the text
        return Step(calls=calls) if calls else Step(text=clean_reply(message.get("content") or ""))

    def complete(self, messages: list[dict], max_new: int) -> str:
        payload = {"messages": messages, "temperature": 0.1, "max_tokens": max_new}
        if self.model:
            payload["model"] = self.model
        req = urllib.request.Request(  # noqa: S310 - the server the user configured
            self.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=SERVER_TIMEOUT_S) as r:  # noqa: S310 - the configured server
            return clean_reply(json.loads(r.read())["choices"][0]["message"].get("content") or "")


# ---------------------------------------------------------------- the loop
@dataclass
class AgentResult:
    text: str
    calls: list[CallRecord]


def system_message(context: list, devices: list[str] = (), states: list[dict] = (), notes: list[str] = ()) -> dict:
    """The tool prompt, the user's devices (names only), the state of devices the user named (read
    just now), and what recall found (the same lines Somni's own prompt uses)."""
    content = SYSTEM
    if devices:
        content += "\n\nThe user's smart-home devices: " + "; ".join(devices) + "."
    for note in notes:
        content += "\n\n" + note
    if states:
        content += ("\n\nState of the devices the user mentioned, read just now: "
                    + "; ".join(json.dumps(s, ensure_ascii=False) for s in states) + ".")
    if context:
        from sleepai.models.llm import MEMORY_PREAMBLE
        lines = [f"- [{m.status}, {m.strength}] {m.key}: {m.summary}" for m in context]
        content += "\n\n" + MEMORY_PREAMBLE + "\n" + "\n".join(lines)
    return {"role": "system", "content": content}


def _unfinished(records: list[CallRecord]) -> str | None:
    """A tool that failed and was not tried again, although another tool succeeded after it."""
    for i, record in enumerate(records):
        later = records[i + 1:]
        if (not record.ok and any(r.ok for r in later)
                and not any(r.name == record.name and r.ok for r in later)):
            return record.name
    return None


class ToolAgent:
    def __init__(self, backend: Backend, ctx: ToolContext, emit: Emitter, max_rounds: int = 3) -> None:
        self.backend, self.ctx, self.emit, self.max_rounds = backend, ctx, emit, max_rounds

    def _honest(self, step: Step, records: list[CallRecord], messages: list[dict], schemas: list[dict]) -> Step:
        """Every action failed: the answer must say so (measured: after two failed device calls a 4B
        model said "Music should now start playing")."""
        acted = [r for r in records if r.name in BY_NAME and BY_NAME[r.name].acts
                 and not r.error.startswith("the user ")]        # a guard refusing what was not asked: fine
        if not acted or any(r.ok for r in acted):
            return step
        failures = "; ".join(f"{r.name}: {r.error}" for r in acted)
        self.emit(EventType.TOKEN, text="", replace=True)
        messages.append({"role": "assistant", "content": step.text or ""})
        messages.append({"role": "user", "content": f"(That did not happen: every action failed ({failures}). "
                         "Tell me plainly, in one or two sentences, what you could not do and why. Do not say "
                         "it worked.)"})
        honest = self.backend.step(messages, [], first=False)
        if honest.calls or not honest.text:
            return Step(text=f"Sorry, sir, I couldn't do that: {acted[-1].error}.")
        return honest

    def run(self, question: str, context: list, tools: list[Tool], history: list[dict] = (),
            forced: list[dict] | None = None) -> AgentResult | None:
        """None when no tool was needed and the backend wants the normal answer path. `forced`: calls
        to make without asking the model first (it already said it does not know)."""
        schemas = [t.schema() for t in tools]
        clock = time.perf_counter()
        states = self.ctx.mentioned_devices(question)
        messages = with_history([system_message(context, self.ctx.device_names(), states, self.ctx.notes()),
                                 {"role": "user", "content": question}], list(history))
        timings = [("devices", time.perf_counter() - clock)]
        clock = time.perf_counter()
        step = Step(calls=forced) if forced else self.backend.step(messages, schemas, first=True)
        timings.append(("decide", time.perf_counter() - clock))
        if step is None:
            if not states:
                return None
            # No tool needed, but the answer is about a device: answer here, where its state is
            # known, not on the memory path that cannot see it.
            step = self.backend.step(messages, schemas, first=False)
        records: list[CallRecord] = []
        nudged = False
        for _round in range(self.max_rounds):
            unfinished = None if nudged or step.calls else _unfinished(records)
            if unfinished:
                # It gathered what a failed tool needed, then answered without using it (measured:
                # weather -> "needs a town" -> computer_info -> "the weather is not available").
                nudged = True
                self.emit(EventType.TOKEN, text="", replace=True)
                messages.append({"role": "assistant", "content": step.text or ""})
                messages.append({"role": "user", "content": f"(Not finished: you now have what {unfinished} "
                                 f"needed. Call {unfinished} again, then answer my question.)"})
                step = self.backend.step(messages, schemas, first=False)
                timings.append(("nudge", 0.0))
            if not step.calls:
                step = self._honest(step, records, messages, schemas)
                log.info("tool turn: %s", ", ".join(f"{name} {seconds:.1f}s" for name, seconds in timings))
                return AgentResult(step.text or "", records)
            ids = [f"call_{len(records) + i}" for i in range(len(step.calls))]
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for cid, c in zip(ids, step.calls, strict=True)]})
            for cid, call in zip(ids, step.calls, strict=True):
                self.emit(EventType.TOOL_CALL, call_id=cid, name=call["name"], arguments=call["arguments"])
                clock = time.perf_counter()
                record = run_tool(call["name"], call["arguments"], self.ctx)
                timings.append((call["name"], time.perf_counter() - clock))
                records.append(record)
                self.emit(EventType.TOOL_RESULT, call_id=cid, name=record.name, ok=record.ok,
                          summary=record.summary())
                messages.append({"role": "tool", "tool_call_id": cid, "name": call["name"],
                                 "content": record.for_model()})
            clock = time.perf_counter()
            step = self.backend.step(messages, schemas, first=False)
            timings.append(("answer" if not step.calls else "next step", time.perf_counter() - clock))
        if not step.calls:
            step = self._honest(step, records, messages, schemas)
        log.info("tool turn: %s", ", ".join(f"{name} {seconds:.1f}s" for name, seconds in timings))
        if step.calls:                                      # still asking for tools: stop here
            return AgentResult("I tried, but it took too many steps. Here is what I did: "
                               + "; ".join(r.summary() for r in records), records)
        return AgentResult(step.text or "", records)
