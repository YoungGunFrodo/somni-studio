"""Philips Hue bulbs over Bluetooth, straight from this PC: no bridge, no app, no account.

Every Hue bulb sold since 2019 also speaks Bluetooth Low Energy. A new bulb, fresh from the box,
accepts pairing as soon as it is powered; Windows remembers the pairing, so after that Somni just
connects. Range is the PC's Bluetooth: about a room, through a wall or two.

The bulb's light-control service (as Philips' own app uses it, and as documented by the open-source
projects that control these bulbs):
  932c32bd-0002-...  power        1 byte, 0 off / 1 on
  932c32bd-0003-...  brightness   1 byte, 1-254
  932c32bd-0004-...  white        2 bytes little-endian, mireds (153 = 6500K ... 454 = 2200K)
  932c32bd-0005-...  color        4 bytes: x, y as little-endian uint16 scaled to 0-65535
White-only bulbs simply have no color characteristic (and "white ambiance" ones no color).

Bluetooth is asynchronous (bleak); the rest of Somni is not. One background thread runs an event
loop and keeps each bulb's connection open, so a command takes a moment rather than a reconnect.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import threading
import time

from . import colors
from .base import Command, Device, DeviceError, Driver, Found, NeedsUser

log = logging.getLogger(__name__)


def _uuid(n: int) -> str:
    return f"932c32bd-{n:04x}-47a2-835a-a8d455b859dd"


POWER, BRIGHTNESS, WHITE, COLOR = _uuid(2), _uuid(3), _uuid(4), _uuid(5)
LIGHT_SERVICE = _uuid(0)
HUE_ADVERT = "0000fe0f-0000-1000-8000-00805f9b34fb"        # Signify (Philips Hue) service, in adverts
MODEL = "00002a24-0000-1000-8000-00805f9b34fb"             # standard Device Information: model number
NAME = "97fe6561-0003-4f62-86e9-b71ee2da3d22"             # the bulb's name, as set in the Hue app
CONNECT_S = 12.0
COMMAND_S = 8.0
RETRY_AFTER_S = 45.0          # a bulb that just failed to connect is reported unreachable, not retried
MIREDS_MIN, MIREDS_MAX = 153, 454


def available() -> bool:
    try:
        import bleak  # noqa: F401
    except ImportError:
        return False
    return True


class _Loop:
    """One event loop thread for every Bluetooth conversation, with each bulb's connection kept open."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self.clients: dict[str, object] = {}
        self.failed_at: dict[str, float] = {}

    def run(self, coro, timeout: float):
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(target=self._loop.run_forever, name="bluetooth", daemon=True).start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise


LOOP = _Loop()


def _bleak_errors() -> tuple[type[BaseException], ...]:
    from bleak.exc import BleakError
    return (BleakError, OSError, asyncio.TimeoutError, TimeoutError)


async def _client(address: str, pair: bool = False):
    """The open connection to a bulb, connecting (and pairing) if needed."""
    from bleak import BleakClient
    client = LOOP.clients.get(address)
    if client is not None and client.is_connected:
        return client
    client = BleakClient(address, timeout=CONNECT_S,
                         disconnected_callback=lambda _c: LOOP.clients.pop(address, None))
    await client.connect()
    if pair:
        with contextlib.suppress(NotImplementedError):   # this platform pairs by itself when needed
            await client.pair()
    LOOP.clients[address] = client
    return client


def _has(client, uuid: str) -> bool:
    return client.services.get_characteristic(uuid) is not None


async def _read_state(address: str) -> dict:
    client = await _client(address)
    state = {"on": (await client.read_gatt_char(POWER))[0] == 1,
             "brightness": round((await client.read_gatt_char(BRIGHTNESS))[0] * 100 / 254),
             "white": None, "xy": None, "can_color": _has(client, COLOR), "can_white": _has(client, WHITE)}
    if state["can_white"]:
        data = await client.read_gatt_char(WHITE)
        state["white"] = struct.unpack_from("<H", data)[0] if len(data) >= 2 else None
    if state["can_color"]:
        data = await client.read_gatt_char(COLOR)
        if len(data) >= 4:
            x, y = struct.unpack_from("<HH", data)
            state["xy"] = (x / 65535, y / 65535)
    return state


async def _write(address: str, uuid: str, payload: bytes) -> None:
    client = await _client(address)
    if not _has(client, uuid):
        raise DeviceError("this bulb cannot do that (it is a white-only model)")
    await client.write_gatt_char(uuid, payload, response=True)


async def _identify(address: str) -> dict:
    """Connect for the first time: pair, then read the model and name."""
    client = await _client(address, pair=True)
    if not _has(client, POWER):
        raise DeviceError("that is not a Hue light (it has no light controls)")
    out = {}
    for key, uuid in (("model", MODEL), ("name", NAME)):
        try:
            out[key] = (await client.read_gatt_char(uuid)).decode("utf-8", "replace").strip("\0 ")
        except Exception as exc:                     # optional: name and model are niceties
            log.debug("no %s from %s: %s", key, address, exc)
    out["color"] = _has(client, COLOR)
    return out


def _color_name(state: dict) -> str:
    """What a bulb shows, in words. Hue bulbs keep both values; the one last set is not reported, so a
    color bulb whose point is near the white line counts as white."""
    xy, white = state.get("xy"), state.get("white")
    if xy and not (0.29 < xy[0] < 0.50 and 0.29 < xy[1] < 0.43):
        return colors.describe_xy(*xy)
    if white:
        return colors.describe_kelvin(round(1_000_000 / white))
    return ""


class HueBluetooth(Driver):
    id = "hue_ble"
    title = "Philips Hue (Bluetooth)"
    kinds = ("light",)
    enable_hint = ("Hue bulbs without a bridge connect over this PC's Bluetooth: switch the lamp on near the PC "
                   "(within a room or so). A new bulb pairs straight away; one already set up in the Hue app "
                   "needs 'Make visible' there first (Settings → Voice assistants).")

    def discover(self, timeout, seen):
        if not available():
            return []
        from bleak import BleakScanner

        async def scan():
            return await BleakScanner.discover(timeout=max(timeout, 4.0), return_adv=True)
        try:
            heard = LOOP.run(scan(), max(timeout, 4.0) + 5)
        except _bleak_errors() as exc:
            log.info("Bluetooth scan failed: %s", exc)
            return []
        found = []
        for device, adv in heard.values():
            uuids = [u.lower() for u in adv.service_uuids or []]
            name = adv.local_name or device.name or ""
            if HUE_ADVERT in uuids or LIGHT_SERVICE in uuids or name.lower().startswith("hue "):
                found.append(Found(self.id, device.address, name or "Hue bulb", "Philips Hue (Bluetooth)",
                                   ident=device.address.lower(), extra={"rssi": adv.rssi}))
        return found

    def connect(self, found: Found) -> dict:
        try:
            info = LOOP.run(_identify(found.address), CONNECT_S + 10)
        except DeviceError:
            raise
        except _bleak_errors() as exc:
            raise NeedsUser("the bulb would not pair. Switch it off and on at the wall and try again within a "
                            "minute; if it was set up in the Hue app before, turn on 'Make visible' there "
                            f"(Settings → Voice assistants). ({exc})") from None
        extra = {"color": info.get("color", False)}
        if info.get("model"):
            extra["model"] = f"Philips Hue {info['model']} (Bluetooth)"
        if info.get("name") and info["name"].lower() not in ("", "hue light", "hue lamp"):
            extra["name"] = info["name"]
        return extra

    def devices(self, link):
        address = link["address"]
        if time.monotonic() - LOOP.failed_at.get(address, -RETRY_AFTER_S) < RETRY_AFTER_S:
            raise DeviceError(f"{link['name']} is out of Bluetooth range or switched off")
        try:
            state = LOOP.run(_read_state(address), CONNECT_S + COMMAND_S)
        except _bleak_errors() as exc:
            LOOP.failed_at[address] = time.monotonic()
            raise DeviceError(f"{link['name']} is out of Bluetooth range or switched off ({exc})") from None
        LOOP.failed_at.pop(address, None)
        return [Device(link["id"], link["name"], "light", state["on"], state["brightness"], True, link["id"],
                       link.get("model", ""), color=_color_name(state))]

    def control(self, link, sub, command: Command) -> None:
        if command.action not in ("turn_on", "turn_off", "set"):
            raise DeviceError(f"Hue lights cannot {command.action.replace('_', ' ')}")
        writes: list[tuple[str, bytes]] = []
        if command.action == "turn_off":
            writes.append((POWER, b"\x00"))
        else:
            writes.append((POWER, b"\x01"))
            if command.brightness is not None:
                writes.append((BRIGHTNESS, bytes([max(1, min(254, round(command.brightness * 254 / 100)))])))
            if command.color:
                try:
                    color = colors.parse(command.color)
                except ValueError as exc:
                    raise DeviceError(str(exc)) from None
                if color.is_white:
                    m = max(MIREDS_MIN, min(MIREDS_MAX, colors.mireds(color.kelvin)))
                    writes.append((WHITE, struct.pack("<H", m)))
                else:
                    if not link.get("color", True):
                        raise DeviceError(f"{link['name']} is a white-only bulb; it can do warm to cool white")
                    x, y = color.xy
                    writes.append((COLOR, struct.pack("<HH", round(x * 65535), round(y * 65535))))

        async def apply():
            for uuid, payload in writes:
                await _write(link["address"], uuid, payload)
        try:
            LOOP.run(apply(), CONNECT_S + COMMAND_S)
        except DeviceError:
            raise
        except _bleak_errors() as exc:
            LOOP.failed_at[link["address"]] = time.monotonic()
            raise DeviceError(f"{link['name']} is out of Bluetooth range or switched off ({exc})") from None
        LOOP.failed_at.pop(link["address"], None)

    def switch(self, link, sub, on, brightness):
        action = "turn_off" if on is False else "turn_on"
        self.control(link, sub, Command(action, brightness))
