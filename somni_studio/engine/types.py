"""Shared engine types: what a provider returns and how it reports progress."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..events import EventBus, EventType


class Emitter:
    """Publishes events for one chat request (every event carries its request id)."""

    def __init__(self, bus: EventBus, request_id: str) -> None:
        self._bus, self.request_id = bus, request_id

    def __call__(self, kind: EventType, **payload: Any) -> None:
        self._bus.publish(kind, request_id=self.request_id, **payload)


@dataclass(frozen=True)
class RecallItem:
    node_id: str          # graph node to light up
    source: str           # search | tags | file
    rank: int
    label: str

    def to_wire(self) -> dict:
        return {"node_id": self.node_id, "source": self.source, "rank": self.rank, "label": self.label}


@dataclass
class ChatOutcome:
    text: str
    used_node: str | None = None          # the memory the answer cited, as a graph node id
    caveat: bool = False                  # answered from a disputed memory
    episode_id: int | None = None         # logged episode (None in read-only mode)
    recall: list[RecallItem] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)     # [{"name", "ok", "summary"}] calls made for it


class Provider(Protocol):
    """A way of answering: the in-process Somni model, or an OpenAI-compatible server."""

    name: str

    def load(self) -> None: ...

    def ask(self, question: str, emit: Emitter) -> ChatOutcome: ...

    def close(self) -> None: ...

    def describe(self) -> dict: ...
