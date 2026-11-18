"""A provider that answers instantly and emits the same events the real one does."""
from __future__ import annotations

import threading

from somni_studio.engine.types import ChatOutcome, RecallItem
from somni_studio.events import EventType


class FakeProvider:
    name = "fake"
    instances: list[FakeProvider] = []

    def __init__(self, settings, fail_load: bool = False, fail_ask: bool = False, gate: threading.Event | None = None):
        self.settings = settings
        self.fail_load, self.fail_ask, self.gate = fail_load, fail_ask, gate
        self.loaded = self.closed = False
        self.questions: list[str] = []
        FakeProvider.instances.append(self)

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("no GPU here")
        self.loaded = True

    def ask(self, question, emit):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_ask:
            raise ValueError("model exploded")
        self.questions.append(question)
        recall = [RecallItem("mem:m1", "search", 0, "normans/named"), RecallItem("file:1", "file", 1, "file/1")]
        emit(EventType.RECALL, items=[r.to_wire() for r in recall])
        emit(EventType.ACTIVITY, phase="read", layers=[{"node_id": "brain:layer:0", "value": 0.5}])
        for piece in ("Nor", "mandy"):
            emit(EventType.TOKEN, text=piece)
        return ChatOutcome("Normandy", "mem:m1", False, None, recall)

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict:
        return {"provider": "fake", "model": "fake-model"}


def factory(**kwargs):
    return lambda settings: FakeProvider(settings, **kwargs)
