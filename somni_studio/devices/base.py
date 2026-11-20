"""Smart devices Somni talks to directly, over your own network: no hub, no cloud, no account.

A driver knows one family of devices (WiZ, LIFX, Hue, ...). It can find them, connect to them
(some need you to press a button first), list what they control, and switch them.

A *link* is one connection Somni keeps: a single bulb or plug, or a bridge with many lights behind
it. A *device* is one thing you can switch: a light, a plug, a room of lights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


class DeviceError(RuntimeError):
    """Something went wrong; the message is written for the user (and the model) to read."""


class NeedsUser(DeviceError):
    """You have to do something first, e.g. press the button on the Hue bridge."""


POWER_ACTIONS = ("turn_on", "turn_off", "toggle", "set")
MEDIA_ACTIONS = ("play", "pause", "stop", "volume_up", "volume_down", "launch")


@dataclass
class Command:
    action: str                          # one of POWER_ACTIONS or MEDIA_ACTIONS
    brightness: int | None = None        # 1-100
    volume: int | None = None            # 0-100
    app: str = ""                        # for launch: "Netflix"
    color: str = ""                      # "red", "warm white", "#ff8800", "2700K" (colors.parse)


@dataclass
class Found:
    """Something that answered a network scan."""
    driver: str | None                  # None: recognised, but Somni cannot control it yet
    address: str
    name: str
    model: str = ""
    ident: str = ""                      # stable id (MAC, serial): finds it again if its address changes
    note: str = ""                       # what you need to do first, or why it is not supported
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.driver or 'other'}:{(self.ident or self.address).lower()}"

    def to_wire(self) -> dict:
        out = asdict(self)
        out.pop("extra")
        return {**out, "key": self.key, "supported": self.driver is not None}


@dataclass
class Device:
    id: str                              # "<link key>" or "<link key>/<sub>"
    name: str
    kind: str                            # light | switch | speaker | tv
    on: bool | None = None
    brightness: int | None = None        # 0-100, dimmable devices only
    reachable: bool = True
    link: str = ""
    model: str = ""
    volume: int | None = None            # 0-100, speakers and displays
    detail: str = ""                     # what it is doing ("YouTube", "screensaver")
    color: str = ""                      # what color it shows ("warm white (2700K)", "blue")

    @property
    def state(self) -> str:
        if not self.reachable:
            return "unreachable"
        return {True: "on", False: "off"}.get(self.on, "unknown")

    def row(self) -> dict:
        """The shape the model sees (the same as Home Assistant devices)."""
        row = {"entity_id": self.id, "name": self.name, "state": self.state, "type": self.kind}
        if self.brightness is not None and self.on:
            row["brightness_pct"] = self.brightness
        if self.volume is not None:
            row["volume_pct"] = self.volume
        if self.kind == "speaker":
            row["note"] = "to start music use play_music; turn_on and play only resume what was already playing"
        if self.detail:
            row["playing"] = self.detail
        if self.color and self.on:
            row["color"] = self.color
        return row

    def to_wire(self) -> dict:
        return {**asdict(self), "state": self.state}


class Driver:
    id = ""
    title = ""
    kinds: tuple[str, ...] = ("light",)
    enable_hint = ""                     # shown when nothing was found: a setting to switch on, say

    def discover(self, timeout: float, seen: dict) -> list[Found]:
        """Look for devices. `seen` holds what the generic SSDP / mDNS scan heard, by address."""
        return []

    def connect(self, found: Found) -> dict:
        """Extra link data to store (Hue's key, ...). Raises NeedsUser if a button must be pressed."""
        return {}

    def devices(self, link: dict) -> list[Device]:
        raise NotImplementedError

    def control(self, link: dict, sub: str, command: Command) -> None:
        """Do one command. Lights and plugs only switch (see switch); media drivers override this."""
        if command.action not in ("turn_on", "turn_off", "set"):
            raise DeviceError(f"{self.title} devices cannot {command.action.replace('_', ' ')}")
        if command.color:
            raise DeviceError(f"Somni cannot change the color of {self.title} lights yet")
        on = {"turn_on": True, "turn_off": False}.get(command.action)
        if command.action == "set" and command.brightness is None:
            on = True
        self.switch(link, sub, on, command.brightness)

    def switch(self, link: dict, sub: str, on: bool | None, brightness: int | None) -> None:
        """Switch one device. on=None leaves power alone; brightness is 1-100 or None."""
        raise DeviceError(f"{self.title} devices cannot be switched")

    def play(self, link: dict, sub: str, url: str, title: str, content_type: str = "audio/wav",
             live: bool = False) -> None:
        """Play audio from a URL on the device (the assistant's voice, a radio stream)."""
        raise DeviceError(f"{self.title} devices cannot play sound")

    # helpers -----------------------------------------------------------
    @staticmethod
    def device_id(link: dict, sub: str = "") -> str:
        return f"{link['id']}/{sub}" if sub else link["id"]
