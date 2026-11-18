"""Keeps the graph in step with the Somni home.

Every `poll_seconds` the service checks a cheap signature of the home (sizes and modification
times of memory.db, its journal, the checkpoint registry, config.json and the expert folders).
Only when that changes is the graph rebuilt (in a worker thread) and diffed against the previous
one; clients receive the diff, so a night of sleep that adds an expert and fifty memories shows
up as exactly that.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

from ..events import EventBus, EventType
from ..settings import StudioSettings
from ..somni.home import HomePaths, base_model, find_experts, model_shape, read_json
from ..somni.runtime import SomniNotFound, ensure_sleepai
from ..somni.store_reader import StoreReader
from .brain_graph import build_brain_graph
from .memory_graph import TierFn, bridge_link, build_memory_graph
from .model import Graph, diff

log = logging.getLogger(__name__)

Signature = tuple


def _stat(path: Path) -> tuple:
    try:
        st = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), st.st_mtime_ns, st.st_size)


def home_signature(paths: HomePaths) -> Signature:
    parts = [_stat(p) for p in paths.watched()]
    if paths.adapters.is_dir():
        parts += [_stat(d / "experts.json") for d in sorted(paths.adapters.iterdir()) if d.is_dir()]
    return tuple(parts)


def tier_function(settings: StudioSettings, paths: HomePaths) -> tuple[TierFn | None, str | None]:
    """Somni's own forgetting policy (sleepai.memory.salience) with the home's sleep settings."""
    try:
        ensure_sleepai(settings.somni.repo_path)
        from sleepai.config import Config
        from sleepai.memory import salience
    except (SomniNotFound, ImportError) as exc:
        return None, f"salience tiers unavailable: {exc}"
    cfg = read_json(paths.config)
    try:
        sleep_cfg = Config.load(None, **({"sleep": cfg["sleep"]} if "sleep" in cfg else {})).sleep
    except TypeError as exc:
        return None, f"salience tiers unavailable: bad sleep settings in {paths.config}: {exc}"
    return (lambda m, today: salience.tier(m, today, sleep_cfg)), None


def build_graph(settings: StudioSettings) -> Graph:
    paths = HomePaths(settings.somni.home_path)
    warnings: list[str] = []
    if not paths.home.is_dir():
        warnings.append(f"Somni home not found: {paths.home}")
    elif not paths.memory_db.is_file():
        warnings.append(f"no memory store yet: {paths.memory_db}")
    tier_of, problem = tier_function(settings, paths)
    if problem:
        warnings.append(problem)

    snap = StoreReader(paths.memory_db).snapshot(settings.graph.max_memories)
    shape = model_shape(base_model(paths))
    experts = find_experts(paths, settings.somni.adapter_path)
    if not shape.exact:
        warnings.append(f"config.json for {shape.name} not found; drawing {shape.layers} layers")
    if experts.source == "latest":
        warnings.append("no adopted checkpoint has experts; showing the newest trained expert folder")
    if snap.truncated:
        warnings.append(f"showing the {len(snap.memories)} most-used of {snap.totals['memories']} memories")

    from ..tools.catalog import TOOLS
    brain = build_brain_graph(shape, experts, tools=tuple((t.name, t.title) for t in TOOLS))
    memory = build_memory_graph(snap, tier_of)
    graph = Graph()
    graph.merge(brain)
    graph.merge(memory)
    graph.add_link(bridge_link())
    graph.meta = {"home": str(paths.home), "read_only": settings.somni.read_only,
                  "brain": brain.meta, "memory": memory.meta, "warnings": warnings}
    return graph


class GraphService:
    def __init__(self, bus: EventBus, settings: StudioSettings,
                 builder: Callable[[StudioSettings], Graph] = build_graph) -> None:
        self._bus = bus
        self._settings = settings
        self._builder = builder
        self._graph = Graph()
        self._revision = 0
        self._signature: Signature | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self.last_error: str | None = None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def paths(self) -> HomePaths:
        return HomePaths(self._settings.somni.home_path)

    def snapshot(self) -> dict:
        return self._graph.to_wire(self._revision)

    async def refresh(self, force: bool = False) -> bool:
        """Rebuild if the home changed (or always, with force). True when clients were notified."""
        async with self._lock:
            settings = self._settings
            signature = await asyncio.to_thread(home_signature, HomePaths(settings.somni.home_path))
            if not force and signature == self._signature:
                return False
            try:
                graph = await asyncio.to_thread(self._builder, settings)
            except Exception as exc:          # a half-written store must never kill the watcher
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("graph rebuild failed")
                return False
            self.last_error = None
            self._signature = signature
            if force or not self._graph.nodes:
                self._graph = graph
                self._revision += 1
                self._bus.publish(EventType.GRAPH_SNAPSHOT, revision=self._revision)
                return True
            change = diff(self._graph, graph)
            self._graph = graph
            if change.empty:
                return False
            self._revision += 1
            self._bus.publish(EventType.GRAPH_DIFF, **change.to_wire(self._revision, graph.meta))
            return True

    async def reconfigure(self, settings: StudioSettings) -> None:
        self._settings = settings
        await self.refresh(force=True)

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self._settings.graph.poll_seconds)
            try:
                await self.refresh()
            except Exception:
                log.exception("graph watcher iteration failed")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._watch(), name="graph-watcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
