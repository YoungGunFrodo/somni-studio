"""Tool lessons: what each tool turn did, and whether it really worked, kept for sleep.

A small model learns to use tools from examples. The best examples are the house's own: a request,
the tool calls that answered it, and a check of the outcome (the light read back "off" after
"turn it off"; Home Assistant reported the devices changed). When Somni had to be corrected (it
only looked, it claimed something without a tool, it stopped one step short) the lesson keeps
both: the mistake, and the clean version without the correction, which is what to practise.

Lessons are written to <home>/tool_lessons.jsonl. Nothing trains on them yet: that is a sleep
(framework) change, to be made once these prove useful. Like memories and episodes, they are data,
never instructions.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

FILE = "tool_lessons.jsonl"
MAX_CONTENT = 1500                 # tool results are cut to this in a lesson


def verified(record) -> bool | None:
    """Did an action really happen? True / False when it can be checked, None when it cannot."""
    if not record.ok:
        return False
    r = record.result if isinstance(record.result, dict) else {}
    if record.name == "home_control" and isinstance(r.get("now"), dict):
        state = str(r["now"].get("state", "")).lower()
        action = str(record.arguments.get("action", ""))
        want = {"turn_off": "off", "turn_on": "on", "set": "on"}.get(action)
        return None if want is None or not state else state == want
    if record.name == "home_assist":
        return bool(r.get("changed")) and not r.get("failed")
    if record.name == "home_organize":
        return bool(r.get("organized"))
    return None


def outcome(calls: list, acted_names: set[str]) -> str:
    acts = [c for c in calls if c.name in acted_names]
    if not acts:
        return "looked" if calls else "no tool"
    checks = [verified(c) for c in acts]
    if any(v is True for v in checks) and not any(v is False for v in checks):
        return "done (checked)"
    if all(v is False for v in checks):
        return "failed"
    if any(v is False for v in checks):
        return "partly done"
    return "done (unchecked)"


def _cut(message: dict) -> dict:
    m = dict(message)
    if isinstance(m.get("content"), str) and len(m["content"]) > MAX_CONTENT:
        m["content"] = m["content"][:MAX_CONTENT] + "…"
    return m


def record(home: Path, question: str, history: list[dict], result, acted_names: set[str], day: int) -> dict | None:
    """Append one lesson. `result` is an AgentResult with .messages (from the user's request on) and
    .corrections (the mistakes Somni was corrected for)."""
    if result is None or not getattr(result, "messages", None):
        return None
    lesson = {
        "time": time.strftime("%Y-%m-%d %H:%M"), "day": day, "request": question,
        "before": [_cut(m) for m in list(history)[-4:]],
        "steps": [_cut(m) for m in result.messages],
        "answer": result.text,
        "outcome": outcome(result.calls, acted_names),
        "corrected": list(getattr(result, "corrections", [])),
    }
    try:
        path = Path(home) / FILE
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(lesson, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:                     # a lesson must never break an answer
        log.warning("could not keep a tool lesson: %s", exc)
        return None
    return lesson


def summary(home: Path, last: int = 200) -> dict:
    """How tool turns have been going (for 'check your systems' and the dream report later)."""
    path = Path(home) / FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-last:]
    except OSError:
        return {"lessons": 0}
    counts: dict[str, int] = {}
    corrected = 0
    for line in lines:
        try:
            lesson = json.loads(line)
        except ValueError:
            continue
        counts[lesson.get("outcome", "?")] = counts.get(lesson.get("outcome", "?"), 0) + 1
        corrected += bool(lesson.get("corrected"))
    return {"lessons": len(lines), "outcomes": counts, "needed_correcting": corrected}
