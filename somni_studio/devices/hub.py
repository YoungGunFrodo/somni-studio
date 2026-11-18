"""The devices Somni controls directly: scanning, connecting, reading and switching them.

Connections are kept in data/devices.json (a Hue bridge's key is the only secret). Devices often get
a new network address from the router; each connection remembers the device's own id, so when it
stops answering at the old address, a quick scan finds it again.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import net
from .base import Command, Device, DeviceError, Driver, Found, NeedsUser
from .lights import Govee, Hue, Kasa, Lifx, Shelly, Wiz, Yeelight
from .media import Cast, Roku

log = logging.getLogger(__name__)

DRIVERS: tuple[Driver, ...] = (Wiz(), Lifx(), Kasa(), Hue(), Yeelight(), Govee(), Shelly(), Cast(), Roku())
BY_ID = {d.id: d for d in DRIVERS}
SCAN_S = 3.0
SCAN_FRESH_S = 600              # a scan this recent is reused when connecting by name
STATE_FRESH_S = 20              # device states this recent are reused for names
NOT_DEVICES = ("InternetGatewayDevice", "WANDevice", "WANConnectionDevice", "MediaServer")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class DeviceHub:
    on_change = None             # (thing, sentence) -> None: what is connected becomes a memory

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._found: dict[str, Found] = {}
        self._scanned_at = 0.0
        self._devices: list[Device] = []
        self._devices_at = 0.0
        self._links: list[dict] = self._read()

    # ------------------------------------------------------------ storage
    def _read(self) -> list[dict]:
        try:
            return list(json.loads(self.path.read_text(encoding="utf-8")).get("links", []))
        except (OSError, ValueError):
            return []

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"links": self._links}, indent=2), encoding="utf-8")
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    @property
    def links(self) -> list[dict]:
        with self._lock:
            return [dict(link) for link in self._links]

    def _link(self, link_id: str) -> dict | None:
        return next((link for link in self._links if link["id"] == link_id), None)

    # ------------------------------------------------------------ scanning
    def scan(self, timeout: float = SCAN_S) -> list[Found]:
        """Everything on the network that answers: devices Somni can control, and others it recognises."""
        direct = [d for d in DRIVERS if d.id in ("wiz", "lifx", "kasa", "yeelight", "govee")]
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="scan") as pool:
            ssdp_job, mdns_job = pool.submit(net.ssdp, timeout), pool.submit(net.mdns, timeout)
            direct_jobs = [pool.submit(self._safe_discover, d, timeout, {}) for d in direct]
            seen: dict[str, dict] = {}
            for addr, headers in ssdp_job.result().items():
                seen.setdefault(addr, {})["ssdp"] = headers
            for addr, info in mdns_job.result().items():
                seen.setdefault(addr, {})["mdns"] = info
            found = [f for job in direct_jobs for f in job.result()]
            found += [f for d in DRIVERS if d not in direct for f in self._safe_discover(d, timeout, seen)]
            claimed = {f.address for f in found}
            mine = set(net.local_ipv4())
            others = [a for a in seen if a not in claimed and a not in mine]
            found += [f for f in pool.map(lambda a: self._identify(a, seen[a]), others) if f]
        with self._lock:
            self._found = {f.key: f for f in found}
            self._scanned_at = time.monotonic()
            moved = self._follow_moves(found)
        if moved:
            self._write()
        return found

    @staticmethod
    def _safe_discover(driver: Driver, timeout: float, seen: dict) -> list[Found]:
        try:
            return driver.discover(timeout, seen)
        except Exception as exc:                      # one brand failing must not hide the others
            log.warning("%s discovery failed: %s", driver.title, exc)
            return []

    @staticmethod
    def _identify(address: str, info: dict) -> Found | None:
        """Name something no driver claimed (a TV, a printer, a phone): Somni cannot control it yet."""
        ssdp, mdns = info.get("ssdp", {}), info.get("mdns", {})
        if any(kind in st for st in ssdp.get("st", []) for kind in NOT_DEVICES):
            return None                                # the router and media servers: nothing to control
        about = net.describe(ssdp["location"]) if ssdp.get("location") else {}
        txt = mdns.get("txt", {})
        name = about.get("friendlyName") or txt.get("fn") or next(iter(mdns.get("services", {}).values()), "")
        if not name:
            return None
        model = " ".join(filter(None, [about.get("manufacturer"), about.get("modelName") or txt.get("md")]))
        services = mdns.get("services", {})
        why = ("a Matter device: not supported yet" if any("matter" in s for s in services)
               else "a HomeKit device: not supported yet" if "_hap._tcp.local" in services
               else "Somni has no driver for this yet")
        return Found(None, address, name, model, address, note=why)

    def _follow_moves(self, found: list[Found]) -> bool:
        """A connected device answering from a new address (the router gave it another one)."""
        moved = False
        for f in found:
            link = self._link(f.key)
            if link is not None and link["address"] != f.address:
                log.info("%s moved from %s to %s", link["name"], link["address"], f.address)
                link["address"], moved = f.address, True
        return moved

    def found(self) -> list[Found]:
        with self._lock:
            return list(self._found.values())

    # ------------------------------------------------------------ connecting
    def connect(self, target: str) -> dict:
        """Connect what the last scan found: by name, address or key; "all" connects everything that
        needs no button press. Scans first if the last scan is old or does not know the target."""
        if time.monotonic() - self._scanned_at > SCAN_FRESH_S or not self._match_found(target):
            self.scan()
        wanted = self._match_found(target)
        if not wanted:
            names = ", ".join(f.name for f in self.found() if f.driver) or "nothing I can control"
            raise DeviceError(f"I could not find {target!r} on the network. I can see: {names}.")
        report: dict[str, list] = {"connected": [], "already": [], "waiting": [], "unsupported": []}
        for f in wanted:
            if f.driver is None:
                report["unsupported"].append(f"{f.name} ({f.note})")
            elif self._link(f.key) is not None:
                report["already"].append(f.name)
            else:
                try:
                    self._add(f)
                    report["connected"].append(f.name)
                except NeedsUser as exc:
                    report["waiting"].append(f"{f.name}: {exc}")
        self._devices_at = 0.0                         # names changed
        return {k: v for k, v in report.items() if v}

    def lookup(self, name: str) -> list[Found]:
        """Things on the network (connected or not) whose name matches; scans if the last scan is old."""
        if time.monotonic() - self._scanned_at > SCAN_FRESH_S:
            self.scan()
        return self._match_found(name) if name.strip().lower() not in ("all", "everything") else []

    def _match_found(self, target: str) -> list[Found]:
        found = self.found()
        t = target.strip().lower()
        if t in ("all", "everything", "all devices", "every device"):
            return [f for f in found if f.driver and not f.note]
        exact = [f for f in found if t in (f.key, f.address, f.name.lower())]
        if exact:
            return exact
        words = _words(t)
        return [f for f in found if words and all(w in _words(f"{f.name} {f.model}") for w in words)]

    def _add(self, found: Found) -> None:
        driver = BY_ID[found.driver]
        extra = driver.connect(found)
        link = {"id": found.key, "driver": found.driver, "address": found.address, "ident": found.ident,
                "name": found.name, "model": found.model, **extra}
        with self._lock:
            self._links = [x for x in self._links if x["id"] != link["id"]] + [link]
            self._write()
        self._changed(link, f"Somni is connected to the {link['name']} ({link.get('model') or found.driver}), "
                            f"a device on the user's home network.")

    def _changed(self, link: dict, sentence: str) -> None:
        if self.on_change is not None:
            try:
                self.on_change(f"device {link['id']}", sentence)
            except Exception as exc:              # a memory must never break a connection
                log.warning("could not note %s: %s", link["name"], exc)

    def forget(self, link_id: str) -> bool:
        with self._lock:
            gone = self._link(link_id)
            self._links = [x for x in self._links if x["id"] != link_id]
            if gone is None:
                return False
            self._write()
            self._devices_at = 0.0
        self._changed(gone, f"Somni is no longer connected to the {gone['name']}.")
        return True

    # ------------------------------------------------------------ using
    def devices(self, max_age: float = 0.0) -> list[Device]:
        """Every connected device with its current state (read in parallel)."""
        if max_age and time.monotonic() - self._devices_at < max_age:
            return list(self._devices)
        links = self.links
        if not links:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(links)), thread_name_prefix="devices") as pool:
            results = list(pool.map(self._read_link, links))
        devices = [d for group in results for d in group]
        self._devices, self._devices_at = devices, time.monotonic()
        return devices

    def _read_link(self, link: dict) -> list[Device]:
        try:
            return BY_ID[link["driver"]].devices(link)
        except Exception as exc:                       # an unplugged lamp is a state, not an error
            log.info("%s did not answer: %s", link["name"], exc)
            known = [d for d in self._devices if d.link == link["id"]]
            if known:
                return [Device(d.id, d.name, d.kind, None, None, False, d.link, d.model) for d in known]
            return [Device(link["id"], link["name"], BY_ID[link["driver"]].kinds[0], None, None, False, link["id"],
                           link.get("model", ""))]

    def names(self) -> list[str]:
        """'Kitchen display (cast:...)' for the prompt: names only, from a recent read."""
        return [f"{d.name} ({d.id})" for d in self.devices(max_age=STATE_FRESH_S)] if self._links else []

    def find(self, query: str) -> list[Device]:
        devices = self.devices(max_age=STATE_FRESH_S)
        q = query.strip().lower()
        exact = [d for d in devices if q in (d.id.lower(), d.name.lower())]
        if exact:
            return exact
        words = [w for w in _words(q) if w not in ("the", "my", "light", "lights", "lamp")] or _words(q)
        return [d for d in devices if words and all(w in _words(d.name) for w in words)]

    def control(self, device: Device, command: Command) -> Device:
        link_id, _, sub = device.id.partition("/")
        with self._lock:
            link = self._link(device.link or link_id)
        if link is None:
            raise DeviceError(f"{device.name} is not connected")
        if command.action == "toggle":
            command = Command("turn_off" if device.on else "turn_on", command.brightness, command.volume, command.app)
        driver = BY_ID[link["driver"]]
        try:
            driver.control(link, sub, command)
        except (OSError, TimeoutError):
            if not self._refind(link):
                raise DeviceError(f"{device.name} is not answering (is it switched on at the wall?)") from None
            driver.control(self._link(link["id"]), sub, command)
        time.sleep(0.3)                                # let it settle before reading it back
        fresh = [d for d in self._read_link(self._link(link["id"]) or link) if d.id == device.id]
        self._devices_at = 0.0
        return fresh[0] if fresh else device

    def play(self, device: Device, url: str, title: str = "Somni", content_type: str = "audio/wav",
             live: bool = False) -> None:
        link_id, _, sub = device.id.partition("/")
        link = self._link(device.link or link_id)
        if link is None:
            raise DeviceError(f"{device.name} is not connected")
        try:
            BY_ID[link["driver"]].play(link, sub, url, title, content_type, live)
        except (OSError, TimeoutError):
            raise DeviceError(f"{device.name} is not answering (try unplugging it for ten seconds)") from None

    def _refind(self, link: dict) -> bool:
        """It stopped answering: maybe it has a new address. Scan for just its kind."""
        old = link["address"]
        self.scan(timeout=2.0)
        current = self._link(link["id"])
        return current is not None and current["address"] != old
