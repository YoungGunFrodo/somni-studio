"""Somni's own sleep, narrated: python -m somni_studio.somni.sleep_entry --home <home>

Runs exactly `sleepai.cli sleep`, after wrapping the pipeline's stage functions so each stage
announces itself on stdout as a `[studio] {json}` line the studio turns into light:

    formation   episodes -> memories              (start / end with how many were formed)
    training    a new expert learns them          (+ Somni's own "[train] step a/b" lines)
    dreaming    it quizzes itself on what it learned: which memories, and right or wrong
    evaluation  the gates (tests) run

Nothing about the sleep changes: the wrappers only print and pass every argument and result
through. If Somni's internals ever differ, the sleep still runs, just without the narration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MARK = "[studio] "
MAX_IDS = 300


def mark(**payload) -> None:
    print(MARK + json.dumps(payload, default=str), flush=True)


def _stage(module, name: str, phase: str, summary=None) -> None:
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        mark(phase=phase, state="start")
        result = original(*args, **kwargs)
        details = {}
        if summary is not None:
            try:
                details = summary(result)
            except Exception:               # narration must never break a sleep
                details = {}
        mark(phase=phase, state="end", **details)
        return result

    setattr(module, name, wrapper)


def instrument() -> None:
    from sleepai.sleep import dreaming, pipeline

    _stage(pipeline, "run_formation", "formation",
           lambda r: {"created": list(getattr(r, "created", []) or [])[:MAX_IDS]})
    _stage(pipeline, "run_consolidation", "training",
           lambda r: {"learned": list(getattr(r[1], "new_memories", []) or [])[:MAX_IDS]})
    _stage(pipeline, "run_dreaming", "dreaming",
           lambda r: {"attempts": r.attempts, "passed": r.passed, "rounds": r.rounds})
    _stage(pipeline, "evaluate", "evaluation", lambda ev: {"passed": bool(ev.passed)})

    # Dreaming detail. Questions are asked in a known order and every answer is checked with
    # `exact` in that same order, so the n-th check belongs to the n-th question's memory.
    asked_ids: list[str] = []
    checked: list[list] = []
    ask, check = dreaming._dream_questions, dreaming.exact

    def dream_questions(teacher, memories, n):
        asked = ask(teacher, memories, n)
        asked_ids[:] = [m.id for m, _q in asked]
        checked.clear()
        mark(phase="dreaming", state="asking", questions=len(asked),
             memories=list(dict.fromkeys(asked_ids))[:MAX_IDS])
        return asked

    def exact(answer, expected):
        ok = check(answer, expected)
        if len(checked) < len(asked_ids):
            checked.append([asked_ids[len(checked)], bool(ok)])
            if len(checked) == len(asked_ids):
                mark(phase="dreaming", state="checked", results=checked[:MAX_IDS],
                     right=sum(1 for _m, r in checked if r), questions=len(checked))
        return ok

    dreaming._dream_questions = dream_questions
    dreaming.exact = exact


PARENT_CHECK_S = 5


def watch_parent(pid: int) -> None:
    """Exit when the studio is gone. On Windows a force-closed parent does not take its children with
    it; an orphaned sleep would keep training on the GPU while a new studio runs on the same home.
    Exiting mid-sleep is safe: nothing half-trained is ever adopted."""
    import threading
    import time

    import psutil

    def watch():
        while psutil.pid_exists(pid):
            time.sleep(PARENT_CHECK_S)
        os._exit(3)
    threading.Thread(target=watch, daemon=True, name="parent-watch").start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--home", required=True)
    parser.add_argument("--parent", type=int, default=0, help="exit if this process (the studio) goes away")
    args = parser.parse_args(argv)
    if args.parent:
        watch_parent(args.parent)
    try:
        instrument()
    except Exception as exc:
        mark(phase="setup", state="no-narration", reason=f"{type(exc).__name__}: {exc}")
    transfer = Path(args.home) / "TRANSFER"
    if transfer.exists():
        return relearn(args.home, transfer)
    from sleepai.cli import main as somni_main
    return somni_main(["--home", args.home, "sleep"])


def relearn(home: str, marker: Path) -> int:
    """After a model switch: Somni's brain transfer. One sleep with a forced rebuild over the whole
    memory store, so the new model learns everything the old one knew (sleepai/transfer)."""
    from sleepai.cli import _system
    from sleepai.sleep.pipeline import run_sleep
    from sleepai.sleep.scheduler import today
    mark(phase="setup", state="relearn", to=json.loads(marker.read_text(encoding="utf-8")).get("to"))
    system = _system(argparse.Namespace(home=home))
    result = run_sleep(system, day=today(), force_rebuild=True, full_store=True)
    print(f"sleep {result['sleep_no']}: {result['status']} (relearned the memory store for the new model), "
          f"recommendation: {result.get('recommendation', '-')}", flush=True)
    if result["status"] != "paused for wake":
        marker.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
