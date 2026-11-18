"""What every integration declares."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FieldKind(str, Enum):
    TEXT = "text"
    URL = "url"
    INT = "int"
    CHOICE = "choice"                # one of Field.options
    SECRET = "secret"  # noqa: S105 - a field kind; secrets are stored locally, never sent to the browser


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: FieldKind = FieldKind.TEXT
    placeholder: str = ""
    help: str = ""
    default: Any = ""
    required: bool = True
    options: tuple[str, ...] = ()
    hidden: bool = False             # stored, never shown or sent to the browser (sign-in data)

    def to_wire(self) -> dict:
        return {**asdict(self), "kind": self.kind.value, "options": list(self.options)}


class Invocation(str, Enum):
    AUTOMATIC = "automatic"      # the assistant uses it on every question
    USER = "user"                # you trigger it from the UI
    PLANNED = "planned"          # wired up, but the assistant cannot call tools by itself yet


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    invocation: Invocation = Invocation.PLANNED


class State(str, Enum):
    NOT_CONFIGURED = "not_configured"
    CHECKING = "checking"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class Status:
    state: State
    message: str = ""
    details: dict = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)

    def to_wire(self) -> dict:
        return {"state": self.state.value, "message": self.message, "details": self.details,
                "checked_at": self.checked_at}


class CheckFailed(RuntimeError):
    """A connection check failed; the message is shown to the user."""


class Integration:
    id: str = ""
    title: str = ""
    category: str = ""
    description: str = ""
    fields: tuple[Field, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    examples: tuple[str, ...] = ()             # things to say once it is connected
    setup: tuple[str, ...] = ()                # how to set it up, step by step (shown on its card)
    derived_from: str | None = None            # status comes from another connection (no own fields)

    def check(self, config: dict) -> Status:
        """Blocking connection test (run it in a thread). Raises CheckFailed with a readable message."""
        raise NotImplementedError

    def to_wire(self) -> dict:
        return {"id": self.id, "title": self.title, "category": self.category, "description": self.description,
                "fields": [f.to_wire() for f in self.fields if not f.hidden], "examples": list(self.examples),
                "setup": list(self.setup), "derived_from": self.derived_from,
                "tools": [{"name": t.name, "title": t.title, "description": t.description,
                           "invocation": t.invocation.value} for t in self.tools]}
