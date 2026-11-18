"""Sleep on demand, and morning review.

`start()` runs Somni's own sleep (memory formation, a new expert, dreaming, evaluation gates, the
dream report) as a child process, through somni/sleep_entry.py, which narrates each stage. A
separate process keeps the studio responsive and gives training the whole GPU; the chat model is
unloaded first. Output lines are streamed to the UI as they come, and stage markers become
"sleep.phase" events (the brain shows formation, training, dreaming and testing).

Waking uses Somni's own signal, the `<home>/WAKE` file: the sleep stops at its next checkpoint.
`stop()` ends it immediately. A finished sleep leaves a candidate checkpoint "awaiting review":
`approve()` / `reject()` call Somni's morning review in-process (only the registry and the memory
store are touched, so no model is loaded).
"""
from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from .home import HomePaths, read_json

log = logging.getLogger(__name__)

LOG_LINES = 600


class SleepState(str, Enum):
    IDLE = "idle"
    SLEEPING = "sleeping"
    WAKING = "waking"            # WAKE was requested; Somni stops at its next checkpoint
    FINISHED = "finished"
    FAILED = "failed"


class SleepError(RuntimeError):
    pass


PROGRESS = re.compile(r"^(?P<label>[^:|]{0,80}):\s*\d{1,3}%\|")


def same_progress_bar(previous: str, line: str) -> bool:
    """True when `line` is the next update of the progress bar on `previous` (tqdm prints one
    line per update when its output is a pipe); the log then shows one line that moves."""
    a, b = PROGRESS.match(previous), PROGRESS.match(line)
    return bool(a and b and a.group("label") == b.group("label"))


Notify = Callable[[str, dict], None]          # (event kind: "status" | "log" | "phase", payload)

STUDIO_ROOT = Path(__file__).resolve().parents[2]
MARK = "[studio] "
TRAIN_STEP = re.compile(r"\[train\] step (\d+)/(\d+)")
PHASE_TEXT = {
    ("formation", "start"): "forming memories from recent conversations…",
    ("training", "start"): "training a new expert on them…",
    ("dreaming", "start"): "dreaming: quizzing itself on what it learned…",
    ("evaluation", "start"): "testing itself (it must not get worse at anything)…",
}


SLEEP_MODULE = "somni_studio.somni.sleep_entry"


def end_tree(pid: int, wait_s: float = 10.0) -> None:
    """End a process and everything it started. (A venv's python.exe on Windows is a launcher that
    runs the real interpreter as its child: ending only the launcher leaves the sleep running.)"""
    try:
        import psutil
        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), root]
    except Exception:                                   # already gone
        return
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.terminate()
    _, alive = psutil.wait_procs(procs, timeout=wait_s)
    for proc in alive:
        with contextlib.suppress(Exception):
            proc.kill()


def find_sleep(home: str, exclude: int | None = None):
    """A running Somni sleep process for `home` (lower-cased absolute path), or None."""
    try:
        import psutil
    except ImportError:
        return None
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if proc.info["pid"] == exclude or SLEEP_MODULE not in cmd or "--home" not in cmd:
            continue
        i = cmd.index("--home")
        if i + 1 < len(cmd) and str(Path(cmd[i + 1]).resolve()).lower() == home:
            return proc
    return None


class SleepRunner:
    def __init__(self, notify: Notify) -> None:
        self._notify = notify
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._state = SleepState.IDLE
        self._lines: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self._started = self._ended = None
        self._summary = ""
        self._paths: HomePaths | None = None
        self._home: Path | None = None
        self._stopping = False
        self._stop_reason = ""
        self._phase: dict = {}

    # ------------------------------------------------------------ status
    def status(self, paths: HomePaths | None = None) -> dict:
        with self._lock:
            out = {"state": self._state.value, "started_at": self._started, "ended_at": self._ended,
                   "summary": self._summary, "phase": dict(self._phase), "log": list(self._lines)[-200:]}
        leftover = self.leftover() if self._state is not SleepState.SLEEPING else None
        if leftover is not None:
            out.update(state=SleepState.SLEEPING.value, started_at=leftover.create_time(),
                       summary="still sleeping from before Somni Studio restarted (its progress is not shown)")
        paths = paths or self._paths
        if paths is not None:
            out["review"] = pending_review(paths)
            out["last_report"] = last_report(paths)
            from .bootstrap import pending_transfer
            out["relearn"] = pending_transfer(paths)
        return out

    def _set(self, state: SleepState, summary: str | None = None) -> None:
        with self._lock:
            self._state = state
            if summary is not None:
                self._summary = summary
        self._notify("status", {"state": state.value, "summary": self._summary})

    @property
    def running(self) -> bool:
        with self._lock:
            mine = self._proc is not None and self._proc.poll() is None
        return mine or self.leftover() is not None

    def leftover(self):
        """A sleep still running from before this studio started (a studio that was force-closed),
        on the same home: it is Somni's own, so it is shown as sleeping and can be stopped."""
        if self._paths is None and self._home is None:
            return None
        if self._proc is not None and self._proc.poll() is None:
            return None                              # our own sleep (and its child) is the one running
        home = str((self._paths.home if self._paths else self._home).resolve()).lower()
        return find_sleep(home, exclude=self._proc.pid if self._proc else None)

    def watch_home(self, home: Path) -> None:
        """The home whose leftover sleeps belong to this studio."""
        self._home = Path(home)

    # ------------------------------------------------------------ control
    def start(self, repo: Path, paths: HomePaths, python: str = sys.executable) -> None:
        if self.running:
            raise SleepError("already sleeping")
        if not paths.config.exists():
            raise SleepError(f"{paths.home} is not a Somni home (no config.json)")
        wake = paths.home / "WAKE"
        wake.unlink(missing_ok=True)
        path = os.pathsep.join([str(repo), str(STUDIO_ROOT), os.environ.get("PYTHONPATH", "")])
        env = {**os.environ, "PYTHONPATH": path, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen([python, "-u", "-m", "somni_studio.somni.sleep_entry", "--home", str(paths.home),
                                 "--parent", str(os.getpid())],
                                cwd=str(repo), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                creationflags=flags)
        with self._lock:
            self._proc, self._paths, self._stopping, self._phase = proc, paths, False, {}
            self._lines.clear()
            self._started, self._ended, self._summary = time.time(), None, ""
        self._set(SleepState.SLEEPING, "falling asleep…")
        threading.Thread(target=self._pump, args=(proc, paths), name="sleep-log", daemon=True).start()

    def wake(self) -> None:
        """Ask Somni to stop at its next checkpoint (its own WAKE signal)."""
        with self._lock:
            paths = self._paths
        if not self.running or paths is None:
            raise SleepError("not sleeping")
        (paths.home / "WAKE").write_text("woken from Somni Studio\n", encoding="utf-8")
        self._set(SleepState.WAKING, "waking up at the next checkpoint…")

    def stop(self, reason: str = "stopped before it finished; nothing was changed") -> None:
        """End the sleep now. Nothing half-trained is adopted: only finished candidates exist."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            leftover = self.leftover()
            if leftover is None:
                raise SleepError("not sleeping")
            end_tree(leftover.pid)                   # one from before a restart: safe to end, as above
            self._set(SleepState.IDLE, "stopped a sleep left running from before Somni Studio restarted")
            return
        with self._lock:
            self._stopping = True
            self._stop_reason = reason
        end_tree(proc.pid)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def shutdown(self) -> None:
        if self.running:
            self.stop()

    # ------------------------------------------------------------ narration
    def _on_marker(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or "phase" not in payload:
            return
        with self._lock:
            if payload.get("state") in ("start", "asking"):
                self._phase = {"phase": payload["phase"], "state": payload["state"]}
            elif payload.get("state") == "end" and self._phase.get("phase") == payload["phase"]:
                self._phase = {}
        self._notify("phase", payload)
        text = PHASE_TEXT.get((payload["phase"], payload.get("state")))
        if payload["phase"] == "dreaming" and payload.get("state") == "checked":
            text = f"dream: {payload.get('right', 0)} of {payload.get('questions', 0)} answers right"
        elif payload["phase"] == "formation" and payload.get("state") == "end":
            text = f"formed {len(payload.get('created') or [])} memories"
        if text:
            with self._lock:
                self._lines.append(f"— {text}")
                self._summary = text
            self._notify("log", {"line": f"— {text}", "replace": False})

    def _on_progress(self, step: int, total: int) -> None:
        with self._lock:
            phase = self._phase.get("phase", "training")
            self._phase = {**self._phase, "phase": phase, "step": step, "total": total}
        self._notify("phase", {"phase": phase, "state": "progress", "step": step, "total": total})

    # ------------------------------------------------------------ output
    def _pump(self, proc: subprocess.Popen, paths: HomePaths) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith(MARK):
                self._on_marker(line[len(MARK):])
                continue
            step = TRAIN_STEP.search(line)
            if step:
                self._on_progress(int(step.group(1)), int(step.group(2)))
            with self._lock:
                replace = bool(self._lines) and same_progress_bar(self._lines[-1], line)
                if replace:
                    self._lines[-1] = line
                else:
                    self._lines.append(line)
            self._notify("log", {"line": line, "replace": replace})
            if line.startswith("sleep ") and ":" in line:
                with self._lock:
                    self._summary = line
        code = proc.wait()
        (paths.home / "WAKE").unlink(missing_ok=True)
        with self._lock:
            self._ended = time.time()
            woken = self._state is SleepState.WAKING
            stopped = self._stopping
        if stopped:            # the user ended it: not a failure (Windows reports exit code 1 here)
            self._set(SleepState.IDLE, self._stop_reason or "stopped before it finished; nothing was changed")
        elif code == 0:
            self._set(SleepState.FINISHED, self._summary or ("woke up" if woken else "sleep finished"))
        else:
            last = self._lines[-1] if self._lines else ""
            self._set(SleepState.FAILED, f"sleep ended with exit code {code}" + (f": {last}" if last else ""))


# ---------------------------------------------------------------- review (Somni morning review)
def pending_review(paths: HomePaths) -> list[dict]:
    reg = read_json(paths.registry)
    out = []
    for cid, meta in (reg.get("candidates") or {}).items():
        if meta.get("status") == "awaiting review":
            out.append({"checkpoint": cid, "sleep_no": meta.get("sleep_no"), "passed": meta.get("passed"),
                        "memories": len(meta.get("memories") or []), "created_at": meta.get("created_at")})
    return out


def last_report(paths: HomePaths) -> dict | None:
    reports = sorted((paths.home / "sleeps").glob("sleep-*/report.json"))
    if not reports:
        return None
    data = read_json(reports[-1])
    formation = data.get("formation") or {}
    evaluation = data.get("evaluation") or {}
    return {
        "sleep_no": data.get("sleep_no"),
        "status": data.get("status"),
        "recommendation": data.get("recommendation"),
        "candidate": data.get("candidate"),
        "memories_formed": len(formation.get("created") or []),
        "learned": len((data.get("consolidation") or {}).get("new_memories") or []),
        "passed": evaluation.get("passed"),
        "gates": [{"name": g.get("name"), "passed": g.get("passed"), "current": g.get("current"),
                   "candidate": g.get("candidate"), "note": g.get("note", "")}
                  for g in evaluation.get("gates") or [] if isinstance(g, dict)],
        "report_md": str(reports[-1].with_name("report.md")),
    }


def _system(paths: HomePaths):
    """A Somni System for review only: registry + memory store, no models."""
    from sleepai.config import Config
    from sleepai.system import System
    cfg = Config.load(paths.config, home=str(paths.home))
    return System(cfg, teacher=None, trainer=None, student_factory=None)


def approve(paths: HomePaths, checkpoint: str) -> str:
    from sleepai.review import morning
    system = _system(paths)
    try:
        return morning.approve(system, checkpoint)
    finally:
        system.store.db.close()


def reject(paths: HomePaths, checkpoint: str) -> str:
    from sleepai.review import morning
    system = _system(paths)
    try:
        return morning.reject(system, checkpoint)
    finally:
        system.store.db.close()


def report_text(paths: HomePaths) -> str:
    info = last_report(paths)
    if not info:
        return ""
    path = Path(info["report_md"])
    return path.read_text(encoding="utf-8") if path.exists() else json.dumps(info, indent=2)
