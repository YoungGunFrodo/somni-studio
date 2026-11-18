"""Stored connections, their live status, and the catalogue of tools."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from .base import CheckFailed, FieldKind, Integration, Invocation, State, Status, ToolSpec
from .custom_api import CustomApi
from .email_imap import EmailImap
from .google_home import GoogleHome
from .home_assistant import HomeAssistant
from .web import WebLookup

log = logging.getLogger(__name__)

INTEGRATIONS: tuple[Integration, ...] = (HomeAssistant(), GoogleHome(), WebLookup(), CustomApi(), EmailImap())

# Abilities that run on every conversation (not tools the model chooses).
ALWAYS_ON: tuple[tuple[str, ToolSpec], ...] = (
    ("Memory", ToolSpec("memory.recall", "Recall memories",
                        "Search the memory store and memory files before every answer (Somni recall).",
                        Invocation.AUTOMATIC)),
    ("Memory", ToolSpec("memory.learn", "Remember conversations",
                        "Every conversation is logged as an episode; sleep turns it into memories and weights.",
                        Invocation.AUTOMATIC)),
    ("Voice", ToolSpec("speech.say", "Speak answers", "Read answers aloud with on-device text-to-speech.",
                       Invocation.AUTOMATIC)),
    ("Voice", ToolSpec("speech.listen", "Listen", "Hear you through the microphone (push to talk or hands-free).",
                       Invocation.USER)),
)
HOME_ID = "home_assistant"
GROUPS = {"clock": "Assistant", "memory": "Memory", "sleep": "Sleep", "home": "Devices", "devices": "Devices",
          "device": "Devices", "speaker": "Devices", "email": "Email",
          "web": "Web lookup", "weather": "Web lookup", "google": "Google Home", "api": "Custom API",
          "computer": "Assistant"}


class UnknownIntegration(KeyError):
    pass


def _by_id(integration_id: str) -> Integration:
    for integ in INTEGRATIONS:
        if integ.id == integration_id:
            return integ
    raise UnknownIntegration(integration_id)


class ConnectionStore:
    """data/connections.json: {integration id: {field: value}}. Local only; gitignored."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):      # best effort; on Windows data/ keeps the user's own ACL
            os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def get(self, integration_id: str) -> dict:
        with self._lock:
            return dict(self._read().get(integration_id) or {})

    def put(self, integration_id: str, values: dict) -> dict:
        with self._lock:
            data = self._read()
            data[integration_id] = values
            self._write(data)
            return dict(values)

    def delete(self, integration_id: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(integration_id, None) is not None:
                self._write(data)


def clean_values(integ: Integration, values: dict, previous: dict) -> dict:
    """Validate submitted values. An empty secret keeps the stored one (the browser never has it)."""
    if not isinstance(values, dict):
        raise ValueError("values must be an object")
    known = {f.key: f for f in integ.fields}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown field {sorted(unknown)[0]}")
    out = {}
    for key, f in known.items():
        value = values.get(key, previous.get(key, f.default))
        if f.kind is FieldKind.SECRET and (value is None or value == ""):
            value = previous.get(key, "")
        if f.kind is FieldKind.CHOICE and value not in f.options:
            raise ValueError(f"{f.label} must be one of {', '.join(f.options)}")
        if f.kind is FieldKind.INT and value not in ("", None):
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{f.label} must be a number") from None
        if isinstance(value, str):
            value = value.strip()
        if f.required and value in ("", None):
            raise ValueError(f"{f.label} is required")
        out[key] = value
    return out


def public_values(integ: Integration, values: dict) -> dict:
    """Stored values safe to show: secrets become a flag."""
    out = {}
    for f in integ.fields:
        if f.kind is FieldKind.SECRET:
            out[f.key] = ""
            out[f.key + "_set"] = bool(values.get(f.key))
        else:
            out[f.key] = values.get(f.key, f.default)
    return out


class ConnectionsService:
    def __init__(self, store: ConnectionStore, notify: Callable[[dict], None]) -> None:
        self.store = store
        self._notify = notify
        self._status: dict[str, Status] = {}
        self._lock = threading.Lock()

    def status_of(self, integ: Integration) -> Status:
        if isinstance(integ, GoogleHome):
            return GoogleHome.from_home_assistant(self.status_of(_by_id(integ.derived_from)))
        with self._lock:
            known = self._status.get(integ.id)
        if known is not None:
            return known
        return Status(State.CHECKING if self.store.get(integ.id) else State.NOT_CONFIGURED)

    def list(self) -> list[dict]:
        out = []
        for integ in INTEGRATIONS:
            values = self.store.get(integ.id)
            out.append({**integ.to_wire(), "configured": bool(values), "values": public_values(integ, values),
                        "status": self.status_of(integ).to_wire()})
        return out

    def details(self, integration_id: str) -> dict:
        """What the last check found (e.g. Home Assistant's service domains)."""
        return dict(self.status_of(_by_id(integration_id)).details)

    def check(self, integration_id: str) -> Status:
        """Blocking: test the stored connection and remember the result."""
        integ = _by_id(integration_id)
        if integ.derived_from:
            return self.check(integ.derived_from) and self.status_of(integ)
        values = self.store.get(integ.id)
        if not values:
            status = Status(State.NOT_CONFIGURED)
        else:
            try:
                status = integ.check(values)
            except CheckFailed as exc:
                status = Status(State.ERROR, str(exc))
            except Exception as exc:          # a broken integration must not take the server down
                log.exception("checking %s failed", integ.id)
                status = Status(State.ERROR, f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._status[integ.id] = status
        self._notify({"id": integ.id, "status": status.to_wire()})
        for other in INTEGRATIONS:                  # cards whose status follows this one
            if other.derived_from == integ.id:
                self._notify({"id": other.id, "status": self.status_of(other).to_wire()})
        return status

    def check_all(self) -> None:
        for integ in INTEGRATIONS:
            if self.store.get(integ.id):
                self.check(integ.id)

    on_change: Callable[[str, str], None] | None = None   # (thing, sentence): becomes a memory

    def save(self, integration_id: str, values: dict) -> Status:
        integ = _by_id(integration_id)
        self.store.put(integ.id, clean_values(integ, values, self.store.get(integ.id)))
        status = self.check(integ.id)
        if status.state is State.CONNECTED:
            self._changed(integ, f"Somni is connected to {self._described(integ)}.")
        return status

    def _described(self, integ: Integration) -> str:
        config = self.store.get(integ.id)
        if integ.id == "web":
            return f"web lookup, searching with {config.get('provider') or 'wikipedia'}, and weather"
        if integ.id == "custom_api":
            return f"the user's {config.get('name') or 'custom'} API ({config.get('about') or 'no description'})"
        return integ.title

    def _changed(self, integ: Integration, sentence: str) -> None:
        if self.on_change is not None:
            try:
                self.on_change(integ.id, sentence)
            except Exception as exc:              # a memory must never break a connection
                log.warning("could not note %s: %s", integ.id, exc)

    def remove(self, integration_id: str) -> None:
        integ = _by_id(integration_id)
        self._changed(integ, f"Somni is no longer connected to {integ.title}.")
        self.store.delete(integ.id)
        with self._lock:
            self._status.pop(integ.id, None)
        self._notify({"id": integ.id, "status": Status(State.NOT_CONFIGURED).to_wire()})

    def integrations(self) -> tuple[Integration, ...]:
        return INTEGRATIONS

    def config(self, integration_id: str) -> dict:
        return self.store.get(_by_id(integration_id).id)

    def tools(self, speech_ready: bool, listen_ready: bool, tools_enabled: bool, writable: bool,
              device_links: list[dict] = ()) -> list[dict]:
        """Everything the assistant can do: abilities that always run, and the tools it calls itself."""
        from ..tools.catalog import TOOLS
        out = []
        for group, tool in ALWAYS_ON:
            ready, needs = True, None
            if tool.name == "speech.say" and not speech_ready:
                ready, needs = False, "the voice model (scripts/fetch_voice.py)"
            if tool.name == "speech.listen" and not listen_ready:
                ready, needs = False, "a Whisper model (scripts/fetch_listen.py)"
            if tool.name == "memory.learn" and not writable:
                ready, needs = False, "a writable home (somni.read_only off)"
            out.append({"name": tool.name, "title": tool.title, "description": tool.description, "group": group,
                        "invocation": tool.invocation.value, "ready": ready, "needs": needs})
        titles = {i.id: i.title for i in INTEGRATIONS}
        for tool in TOOLS:
            ready, needs = tools_enabled, None if tools_enabled else "tools switched on (Settings → tools.enabled)"
            if ready and tool.needs and self.status_of(_by_id(tool.needs)).state is not State.CONNECTED:
                ready, needs = False, f"connect {titles[tool.needs]}"
            if ready and tool.capability and tool.capability not in (self.details(tool.needs).get("services") or []):
                ready, needs = False, "Home Assistant's Google Assistant SDK integration"
            if ready and tool.name in ("memory_remember", "sleep_start") and not writable:
                ready, needs = False, "a writable home (somni.read_only off)"
            home_connected = self.status_of(_by_id(HOME_ID)).state is State.CONNECTED
            if ready and tool.name in ("home_states", "home_control") and not (device_links or home_connected):
                ready, needs = False, "a connected device (Connect tab, or ask Somni to find your devices)"
            if ready and tool.name == "speaker_say":
                if not any(link["driver"] == "cast" for link in device_links):
                    ready, needs = False, "a connected speaker or display (Nest, Chromecast)"
                elif not speech_ready:
                    ready, needs = False, "the voice model (scripts/fetch_voice.py)"
            out.append({"name": tool.name, "title": tool.title, "description": tool.description.split(" (")[0],
                        "group": GROUPS.get(tool.name.split("_")[0], "Assistant"), "acts": tool.acts,
                        "invocation": "assistant", "ready": ready, "needs": needs})
        return out
