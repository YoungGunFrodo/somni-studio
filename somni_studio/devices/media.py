"""Speakers, displays and TVs on your network.

  Google Cast  Nest Hub, Nest Mini/Audio, Chromecast, Google TV, many TVs and speakers. Volume,
               pause / play / stop, and the assistant speaking out loud on them. No setup.
  Roku         Roku sticks and Roku TVs (External Control Protocol): apps, play / pause, Home,
               volume and power on Roku TVs. No setup ("Control by mobile apps" must not be off).

The Cast protocol is small (a TLS socket carrying length-prefixed protobuf messages with JSON
inside), so it is spoken here directly instead of pulling in a library.
"""
from __future__ import annotations

import contextlib
import json
import socket
import ssl
import struct
import time
import urllib.parse
import xml.etree.ElementTree as ET  # noqa: S405 - replies from devices on the LAN, no entities

from . import net
from .base import Command, Device, DeviceError, Driver, Found

# ---------------------------------------------------------------- Cast protocol
NS_CONNECTION = "urn:x-cast:com.google.cast.tp.connection"
NS_HEARTBEAT = "urn:x-cast:com.google.cast.tp.heartbeat"
NS_RECEIVER = "urn:x-cast:com.google.cast.receiver"
NS_MEDIA = "urn:x-cast:com.google.cast.media"
DEFAULT_MEDIA_RECEIVER = "CC1AD845"
CONNECT_S = 2.5                  # a cast device that does not accept within this is not answering
SENDER = "sender-somni"


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def encode_message(source: str, destination: str, namespace: str, payload: dict) -> bytes:
    """A CastMessage (protocol v2, string payload) as protobuf."""
    def text(field: int, value: str) -> bytes:
        raw = value.encode()
        return _varint(field << 3 | 2) + _varint(len(raw)) + raw
    body = (_varint(1 << 3) + _varint(0) + text(2, source) + text(3, destination) + text(4, namespace)
            + _varint(5 << 3) + _varint(0) + text(6, json.dumps(payload)))
    return struct.pack(">I", len(body)) + body


def decode_message(body: bytes) -> dict:
    fields, pos = {}, 0
    while pos < len(body):
        key, pos = _read_varint(body, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            fields[number], pos = _read_varint(body, pos)
        elif wire == 2:
            size, pos = _read_varint(body, pos)
            fields[number], pos = body[pos:pos + size], pos + size
        else:
            raise DeviceError("the cast device sent something unexpected")
    payload = fields.get(6, b"{}")
    return {"source": fields.get(2, b"").decode(), "namespace": fields.get(4, b"").decode(),
            "payload": json.loads(payload.decode() or "{}")}


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


class CastChannel:
    """One short session with a cast device."""

    def __init__(self, address: str, timeout: float = 4.0) -> None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False               # cast devices use self-signed certificates
        context.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection((net.require_lan(address), 8009), timeout=CONNECT_S)
        self.sock = context.wrap_socket(raw)
        self.timeout, self.request_id = timeout, 0
        self.connect("receiver-0")

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def send(self, destination: str, namespace: str, payload: dict) -> int:
        self.request_id += 1
        payload = {**payload, "requestId": self.request_id}
        self.sock.sendall(encode_message(SENDER, destination, namespace, payload))
        return self.request_id

    def connect(self, destination: str) -> None:
        self.sock.sendall(encode_message(SENDER, destination, NS_CONNECTION, {"type": "CONNECT"}))

    def _recv(self) -> dict:
        size = struct.unpack(">I", self._exactly(4))[0]
        return decode_message(self._exactly(size))

    def _exactly(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise DeviceError("the cast device closed the connection")
            buf += chunk
        return buf

    def wait(self, accept, timeout: float | None = None) -> dict:
        """Read messages until accept(message) is true (answering heartbeats meanwhile)."""
        end = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < end:
            self.sock.settimeout(max(0.1, end - time.monotonic()))
            try:
                message = self._recv()
            except TimeoutError:
                break
            if message["namespace"] == NS_HEARTBEAT and message["payload"].get("type") == "PING":
                self.sock.sendall(encode_message(SENDER, message["source"], NS_HEARTBEAT, {"type": "PONG"}))
                continue
            if accept(message):
                return message
        raise DeviceError("the cast device did not answer in time")

    def request(self, destination: str, namespace: str, payload: dict) -> dict:
        rid = self.send(destination, namespace, payload)
        return self.wait(lambda m: m["payload"].get("requestId") == rid)["payload"]

    def status(self) -> dict:
        return self.request("receiver-0", NS_RECEIVER, {"type": "GET_STATUS"}).get("status", {})


def _running_app(status: dict) -> dict | None:
    for app in status.get("applications") or []:
        if not app.get("isIdleScreen"):
            return app
    return None


class Cast(Driver):
    id, title = "cast", "Google Cast"
    kinds = ("speaker",)

    def discover(self, timeout, seen):
        out = []
        for addr, info in seen.items():
            mdns = info.get("mdns", {})
            if "_googlecast._tcp.local" not in mdns.get("services", {}):
                continue
            txt = mdns.get("txt", {})
            out.append(Found(self.id, addr, txt.get("fn") or "Cast device", txt.get("md") or "Google Cast",
                             str(txt.get("id") or addr).lower()))
        return out

    def devices(self, link):
        with CastChannel(link["address"]) as ch:
            status = ch.status()
        app = _running_app(status)
        volume = status.get("volume", {})
        level = None if volume.get("level") is None else round(volume["level"] * 100)
        detail = ""
        if app:
            detail = app.get("displayName", "")
            if app.get("statusText") and app["statusText"] != detail:
                detail += f": {app['statusText']}"
        return [Device(self.device_id(link), link["name"], "speaker", app is not None, None, True, link["id"],
                       link.get("model", ""), 0 if volume.get("muted") else level, detail[:80])]

    def control(self, link, sub, command: Command):
        with CastChannel(link["address"]) as ch:
            status = ch.status()
            app = _running_app(status)
            action = command.action
            if action == "set" and command.volume is None:
                raise DeviceError("say what volume to set, from 0 to 100")
            if command.volume is not None or action in ("volume_up", "volume_down"):
                level = (status.get("volume") or {}).get("level") or 0
                if command.volume is not None:
                    level = command.volume / 100
                else:
                    level += 0.1 if action == "volume_up" else -0.1
                ch.request("receiver-0", NS_RECEIVER, {"type": "SET_VOLUME",
                                                        "volume": {"level": max(0.0, min(1.0, level))}})
                return
            if action in ("stop", "turn_off"):
                if app:
                    ch.request("receiver-0", NS_RECEIVER, {"type": "STOP", "sessionId": app["sessionId"]})
                return
            if action in ("pause", "play", "turn_on", "toggle"):
                if not app:
                    raise DeviceError("nothing is playing on it")
                ch.connect(app["transportId"])
                media = ch.request(app["transportId"], NS_MEDIA, {"type": "GET_STATUS"}).get("status") or []
                if not media:
                    raise DeviceError(f"{app.get('displayName', 'that app')} has nothing to pause or play")
                kind = {"pause": "PAUSE", "play": "PLAY", "turn_on": "PLAY"}.get(action)
                if kind is None:                                     # toggle
                    kind = "PAUSE" if media[0].get("playerState") == "PLAYING" else "PLAY"
                ch.send(app["transportId"], NS_MEDIA, {"type": kind, "mediaSessionId": media[0]["mediaSessionId"]})
                return
            raise DeviceError(f"a cast device cannot {action.replace('_', ' ')}")

    def play(self, link, sub, url, title, content_type="audio/wav", live=False):
        """Play audio from a URL on it (the Default Media Receiver fetches it: a clip from this computer,
        or a radio stream)."""
        with CastChannel(link["address"], timeout=10) as ch:
            ch.send("receiver-0", NS_RECEIVER, {"type": "LAUNCH", "appId": DEFAULT_MEDIA_RECEIVER})

            def launched(message):
                apps = message["payload"].get("status", {}).get("applications") or []
                return any(a.get("appId") == DEFAULT_MEDIA_RECEIVER and a.get("transportId") for a in apps)
            status = ch.wait(launched, 12)["payload"]["status"]
            app = next(a for a in status["applications"] if a.get("appId") == DEFAULT_MEDIA_RECEIVER)
            ch.connect(app["transportId"])
            ch.send(app["transportId"], NS_MEDIA, {
                "type": "LOAD", "autoplay": True,
                "media": {"contentId": url, "contentType": content_type,
                          "streamType": "LIVE" if live else "BUFFERED",
                          "metadata": {"metadataType": 0, "title": title}}})
            loaded = ("MEDIA_STATUS", "LOAD_FAILED")
            ch.wait(lambda m: m["namespace"] == NS_MEDIA and m["payload"].get("type") in loaded, 10)


# ---------------------------------------------------------------- Roku
class Roku(Driver):
    id, title = "roku", "Roku"
    kinds = ("tv",)
    PORT = 8060
    KEYS = {"play": "Play", "pause": "Play", "stop": "Home", "volume_up": "VolumeUp", "volume_down": "VolumeDown"}

    def _url(self, link_or_address, path: str) -> str:
        address = link_or_address if isinstance(link_or_address, str) else link_or_address["address"]
        return f"http://{address}:{self.PORT}{path}"

    def _xml(self, address, path: str):
        return ET.fromstring(net.http_get(self._url(address, path), timeout=3))  # noqa: S314 - LAN device

    def _post(self, link, path: str) -> None:
        net.http_get(self._url(link, path), method="POST", body=b"")

    def discover(self, timeout, seen):
        out = []
        for addr, info in seen.items():
            ssdp = info.get("ssdp", {})
            if "roku:ecp" not in ssdp.get("st", []) and "Roku" not in ssdp.get("server", ""):
                continue
            try:
                about = self._xml(addr, "/query/device-info")
            except (OSError, ET.ParseError):
                continue
            name = about.findtext("user-device-name") or about.findtext("friendly-device-name") or "Roku"
            out.append(Found(self.id, addr, name, about.findtext("friendly-model-name") or "Roku",
                             (about.findtext("serial-number") or addr).lower(),
                             extra={"tv": about.findtext("is-tv") == "true"}))
        return out

    def connect(self, found):
        return {"tv": found.extra.get("tv", False)}

    def devices(self, link):
        about = self._xml(link, "/query/device-info")
        detail = ""
        with contextlib.suppress(OSError, ET.ParseError):
            active = self._xml(link, "/query/active-app")
            app = active.findtext("app") or ""
            saver = active.find("screensaver")
            detail = "screensaver" if saver is not None else ("Home" if app == "Roku" else app)
        on = about.findtext("power-mode") in ("PowerOn", None)
        return [Device(self.device_id(link), link["name"], "tv", on, None, True, link["id"], link.get("model", ""),
                       None, detail)]

    def apps(self, link) -> dict[str, str]:
        return {(a.text or "").strip(): a.get("id") for a in self._xml(link, "/query/apps").findall("app")}

    def control(self, link, sub, command: Command):
        action = command.action
        if action == "launch":
            apps = self.apps(link)
            want = command.app.lower().strip()
            match = (next((i for n, i in apps.items() if n.lower() == want), None)
                     or next((i for n, i in apps.items() if want and want in n.lower()), None))
            if match is None:
                raise DeviceError(f"no app called {command.app!r} on the Roku (it has {', '.join(list(apps)[:12])})")
            self._post(link, f"/launch/{urllib.parse.quote(match)}")
            return
        if action in ("turn_on", "turn_off", "toggle"):
            if not link.get("tv"):
                if action == "turn_on":
                    self._post(link, "/keypress/Home")
                    return
                raise DeviceError("a Roku stick cannot switch itself or the TV off; I can go to its Home screen")
            key = {"turn_on": "PowerOn", "turn_off": "PowerOff", "toggle": "Power"}[action]
            self._post(link, f"/keypress/{key}")
            return
        if action == "set" and command.volume is not None:
            raise DeviceError("Roku can only turn the volume up or down a step at a time")
        key = self.KEYS.get(action)
        if key is None:
            raise DeviceError(f"a Roku cannot {action.replace('_', ' ')}")
        steps = 3 if action.startswith("volume") else 1
        for _ in range(steps):
            self._post(link, f"/keypress/{key}")
