"""Stepping aside for games: when another program needs the GPU, the assistant lets go of it.

Every few seconds the guard reads who is using the GPU (gpu_watch.py). When another program is
busy (above gpu.busy_percent for a few samples in a row: a game, a video export, a training run):

  * the chat model is unloaded, freeing its VRAM (after the answer in progress, if any);
  * a sleep in progress is stopped; Somni's training saves progress as it goes, so the next
    sleep continues instead of starting over.

When the GPU has been free for gpu.resume_after_s, whatever was paused comes back. While paused,
questions get a polite "paused" answer instead of loading the model onto a busy GPU.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .events import EventBus, EventType
from .gpu_watch import GpuMonitor, GpuPolicy, GpuUser
from .settings import GpuSettings

log = logging.getLogger(__name__)

INTERVAL_S = 3.0
BUSY_SAMPLES = 3                      # ~9 s of sustained use before pausing (no flapping on a blip)


@dataclass
class GuardHooks:
    """What the guard pauses and resumes (supplied by the app, so the guard stays testable)."""
    model_loaded: Callable[[], bool]
    unload_model: Callable[[], Awaitable[None]]
    load_model: Callable[[], None]
    sleeping: Callable[[], bool]
    stop_sleep: Callable[[str], Awaitable[None]]
    start_sleep: Callable[[], Awaitable[None]]


OWN_MODULES = ("somni_studio.somni.sleep_entry", "somni_studio.speech.stt_worker")


def _own_pids() -> set[int]:
    """This studio, its children (the sleep process, the speech worker), and any Somni sleep or speech
    worker left running by an earlier studio: Somni must never pause for itself."""
    pids = {os.getpid()}
    try:
        import psutil
        pids |= {c.pid for c in psutil.Process().children(recursive=True)}
        for proc in psutil.process_iter(["pid", "cmdline"]):
            if any(module in (proc.info.get("cmdline") or []) for module in OWN_MODULES):
                pids.add(proc.info["pid"])
    except Exception as exc:              # a process that just exited: this studio still counts
        log.debug("could not list Somni's processes: %s", exc)
    return pids


class GpuGuard:
    def __init__(self, bus: EventBus, settings: GpuSettings, hooks: GuardHooks,
                 monitor: GpuMonitor | None = None) -> None:
        self._bus, self._hooks = bus, hooks
        self._monitor = monitor if monitor is not None else GpuMonitor()
        self._settings = settings
        self._policy = self._make_policy(settings)
        self._paused_by: GpuUser | None = None
        self._paused_at: float | None = None
        self._resume: dict = {}               # what to bring back: {"model": bool, "sleep": bool}
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def _make_policy(self, settings: GpuSettings) -> GpuPolicy:
        return GpuPolicy(settings.busy_percent, BUSY_SAMPLES, settings.resume_after_s, settings.ignored,
                         own_pids=set())

    def configure(self, settings: GpuSettings) -> None:
        self._settings = settings
        self._policy = self._make_policy(settings)

    # ------------------------------------------------------------ status
    @property
    def paused(self) -> bool:
        return self._paused_by is not None

    def status(self) -> dict:
        by = self._paused_by
        return {"enabled": self._settings.auto_pause, "available": self._monitor.available, "paused": by is not None,
                "by": {"name": by.name, "pid": by.pid, "percent": by.percent} if by else None,
                "since": self._paused_at, "resume_after_s": self._settings.resume_after_s,
                "resumes": dict(self._resume)}

    def pause_message(self) -> str:
        by = self._paused_by
        return (f"I'm paused while {by.name if by else 'another program'} uses the graphics card. I'll be back "
                "when it's done (or press Resume).")

    # ------------------------------------------------------------ the loop
    async def tick(self, users: list[GpuUser] | None = None) -> None:
        """One check (users injected by tests; normally sampled)."""
        if not self._settings.auto_pause:
            if self.paused:
                await self.resume()
            return
        if users is None:
            users = await asyncio.to_thread(self._monitor.sample)
        self._policy.own_pids = await asyncio.to_thread(_own_pids)
        verdict = self._policy.update(users)
        if verdict.busy and not self.paused:
            await self._pause(verdict.by)
        elif not verdict.busy and self.paused:
            await self.resume()

    async def _pause(self, by: GpuUser) -> None:
        async with self._lock:
            if self.paused:
                return
            self._paused_by, self._paused_at = by, time.time()
            self._resume = {"model": self._hooks.model_loaded(), "sleep": self._hooks.sleeping()}
            log.info("pausing: %s is using the GPU (%.0f%%)", by.name, by.percent)
            self._publish()
            if self._resume["sleep"]:
                await self._hooks.stop_sleep(f"paused while {by.name} used the GPU; the next sleep continues")
            if self._resume["model"]:
                await self._hooks.unload_model()

    async def resume(self, reason: str = "the GPU is free again") -> None:
        async with self._lock:
            if not self.paused:
                return
            bring_back, self._resume = self._resume, {}
            self._paused_by = self._paused_at = None
            log.info("resuming: %s", reason)
            self._publish()
            if bring_back.get("model"):
                self._hooks.load_model()
            if bring_back.get("sleep"):
                try:
                    await self._hooks.start_sleep()
                except Exception:
                    log.exception("could not continue the sleep")

    def _publish(self) -> None:
        self._bus.publish(EventType.GPU_STATUS, **self.status())

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(INTERVAL_S)
            try:
                await self.tick()
            except Exception:
                log.exception("GPU guard check failed")

    def start(self) -> None:
        if self._task is None and self._monitor.available:
            self._task = asyncio.create_task(self._run(), name="gpu-guard")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._monitor.close()
