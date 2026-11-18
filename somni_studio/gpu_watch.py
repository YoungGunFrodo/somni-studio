"""Who is using the GPU right now (Windows), and the policy that steps aside for them.

Reads the same per-process numbers as Task Manager's GPU column (performance counters
"\\GPU Engine(*)\\Utilization Percentage") through the Windows PDH API in-process: about 1 ms per
sample, no PowerShell. Games and other graphics programs show on the 3D and compute engines
(measured: a Vulkan workload reads 100% "3D"). ROCm/HIP compute (this studio's own model) does
not appear in these counters at all, and this process is excluded anyway.

On other systems `GpuMonitor.available` is False and nothing is ever paused.
"""
from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

ENGINE_COUNTER = r"\GPU Engine(*)\Utilization Percentage"
INSTANCE = re.compile(r"pid_(\d+)_.*engtype_(\w+)", re.I)
WORK_ENGINES = ("3d", "compute", "graphics")      # where games and renderers show up
PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2
ERROR_SUCCESS = 0


class _Value(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_uint32), ("doubleValue", ctypes.c_double)]


class _Item(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _Value)]


@dataclass(frozen=True)
class GpuUser:
    pid: int
    name: str
    percent: float                   # 3D engine utilization, 0-100


@dataclass
class _Pdh:
    dll: object = None
    query: ctypes.c_void_p = field(default_factory=ctypes.c_void_p)
    counter: ctypes.c_void_p = field(default_factory=ctypes.c_void_p)


class GpuMonitor:
    """Per-process 3D-engine utilization. Call sample() every few seconds."""

    def __init__(self) -> None:
        self._pdh: _Pdh | None = None
        self._names: dict[int, str] = {}
        if sys.platform != "win32":
            return
        try:
            pdh = _Pdh(ctypes.WinDLL("pdh"))
            if pdh.dll.PdhOpenQueryW(None, None, ctypes.byref(pdh.query)) != ERROR_SUCCESS:
                return
            if pdh.dll.PdhAddEnglishCounterW(pdh.query, ENGINE_COUNTER, None, ctypes.byref(pdh.counter)) \
                    != ERROR_SUCCESS:
                return
            pdh.dll.PdhCollectQueryData(pdh.query)   # rate counters need two collections
            self._pdh = pdh
        except OSError as exc:
            log.warning("GPU counters unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._pdh is not None

    def _name(self, pid: int) -> str:
        if pid not in self._names:
            try:
                import psutil
                self._names[pid] = psutil.Process(pid).name()
            except Exception:
                self._names[pid] = f"pid {pid}"
        return self._names[pid]

    def sample(self) -> list[GpuUser]:
        """Processes using the 3D engine since the last sample, busiest first."""
        if self._pdh is None:
            return []
        dll, counter = self._pdh.dll, self._pdh.counter
        if dll.PdhCollectQueryData(self._pdh.query) != ERROR_SUCCESS:
            return []
        size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
        status = dll.PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
                                                  None)
        if (status & 0xFFFFFFFF) != PDH_MORE_DATA or not size.value:     # ctypes returns it signed
            return []
        buffer = (ctypes.c_byte * size.value)()
        if dll.PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
                                            buffer) != ERROR_SUCCESS:
            return []
        items = ctypes.cast(buffer, ctypes.POINTER(_Item))
        per_engine: dict[tuple[int, str], float] = defaultdict(float)
        for i in range(count.value):
            match = INSTANCE.search(items[i].szName or "")
            if not match or items[i].FmtValue.CStatus not in (0, 1):
                continue
            engine = match.group(2).lower()
            if engine.startswith(WORK_ENGINES):
                per_engine[(int(match.group(1)), engine)] += items[i].FmtValue.doubleValue
        per_pid: dict[int, float] = defaultdict(float)
        for (pid, _engine), pct in per_engine.items():          # a process's busiest engine, not the
            per_pid[pid] = max(per_pid[pid], pct)                # sum: one job can show on two engines
        users = [GpuUser(pid, self._name(pid), round(min(100.0, pct), 1)) for pid, pct in per_pid.items() if pct > 0.5]
        return sorted(users, key=lambda u: -u.percent)

    def close(self) -> None:
        if self._pdh is not None:
            self._pdh.dll.PdhCloseQuery(self._pdh.query)
            self._pdh = None


# ---------------------------------------------------------------- the policy
ALWAYS_IGNORED = {"dwm.exe", "system", "idle", "csrss.exe", "explorer.exe"}


@dataclass
class Verdict:
    busy: bool
    by: GpuUser | None = None


class GpuPolicy:
    """Decides when someone else needs the GPU: one other program above `busy_percent` for
    `busy_samples` samples in a row. Free again after `free_seconds` with nobody above it.
    Programs in `ignore` (and this studio's own processes) never count."""

    def __init__(self, busy_percent: float, busy_samples: int, free_seconds: float, ignore: set[str],
                 own_pids: set[int] | None = None) -> None:
        self.busy_percent, self.busy_samples, self.free_seconds = busy_percent, busy_samples, free_seconds
        self.ignore = {n.lower() for n in ignore} | ALWAYS_IGNORED
        self.own_pids = own_pids if own_pids is not None else {os.getpid()}
        self._streak = 0
        self._busy: GpuUser | None = None
        self._free_since: float | None = None

    def others(self, users: list[GpuUser]) -> list[GpuUser]:
        return [u for u in users if u.pid not in self.own_pids and u.name.lower() not in self.ignore]

    def update(self, users: list[GpuUser], now: float | None = None) -> Verdict:
        now = time.monotonic() if now is None else now
        heavy = [u for u in self.others(users) if u.percent >= self.busy_percent]
        if heavy:
            self._streak += 1
            self._free_since = None
            if self._streak >= self.busy_samples:
                self._busy = heavy[0]
        else:
            self._streak = 0
            if self._busy is not None:
                if self._free_since is None:
                    self._free_since = now
                if now - self._free_since >= self.free_seconds:
                    self._busy = None
                    self._free_since = None
        return Verdict(self._busy is not None, self._busy)
