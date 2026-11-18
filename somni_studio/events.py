"""Server -> browser events.

Producers (the graph watcher on the event loop, the model worker on its own thread) call
`EventBus.publish`; every WebSocket client has a bounded queue. A client that falls behind is
never allowed to slow the model down: its queue is cleared and it is told to resync (fetch a
fresh graph snapshot), which is always correct because events are hints and the REST API is the
source of truth.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)

CLIENT_QUEUE_SIZE = 1024


class EventType(str, Enum):
    HELLO = "hello"
    RESYNC = "resync"                 # client fell behind: refetch the graph
    GRAPH_SNAPSHOT = "graph.snapshot" # the graph was rebuilt from scratch (settings changed)
    GRAPH_DIFF = "graph.diff"         # nodes/links added, changed or removed
    ENGINE_STATUS = "engine.status"
    CHAT_START = "chat.start"
    RECALL = "recall"                 # memories / memory files retrieved for a question
    ROUTE = "route"                   # per-expert routing scores and on/off decisions
    ACTIVITY = "activity"             # per-layer activity while the model reads or writes
    TOKEN = "token"  # noqa: S105 - an event name, not a secret: one generated token
    CHAT_DONE = "chat.done"
    CHAT_ERROR = "chat.error"
    SPEECH = "speech"                 # one spoken sentence is ready (or failed)
    SLEEP_STATUS = "sleep.status"     # sleep started / waking / finished / failed
    SLEEP_LOG = "sleep.log"           # one line of sleep output
    SLEEP_PHASE = "sleep.phase"       # formation / training / dreaming / evaluation, with details
    TOOL_CALL = "tool.call"           # the assistant is using a tool
    TOOL_RESULT = "tool.result"       # ... and what came back
    GPU_STATUS = "gpu.status"         # paused for another program, or back
    CONNECTION = "connection"         # a connection's status changed


class EventBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def publish(self, kind: EventType, **payload: Any) -> None:
        """Thread-safe. Safe to call before the loop is attached (the event is dropped)."""
        event = {"type": kind.value, "ts": time.time(), **payload}
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._fanout(event)
        else:
            loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict) -> None:
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                while not q.empty():
                    q.get_nowait()
                q.put_nowait({"type": EventType.RESYNC.value, "ts": time.time()})
                log.warning("a client fell behind; it was told to resync")
