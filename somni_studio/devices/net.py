"""Talking on the local network: broadcasts, SSDP and a small mDNS query.

Everything here stays on your LAN: replies are only accepted from private addresses, and devices
are only ever contacted at private addresses.
"""
from __future__ import annotations

import contextlib
import ipaddress
import re
import socket
import struct
import time
import urllib.request
import xml.etree.ElementTree as ET  # noqa: S405 - parses device descriptions from the LAN, no entities

SSDP_ADDR = ("239.255.255.250", 1900)
MDNS_ADDR = ("224.0.0.251", 5353)
MDNS_SERVICES = ("_hue._tcp.local", "_shelly._tcp.local", "_http._tcp.local", "_googlecast._tcp.local",
                 "_sonos._tcp.local", "_hap._tcp.local", "_matter._tcp.local", "_matterc._udp.local",
                 "_airplay._tcp.local", "_elg._tcp.local", "_wled._tcp.local", "_esphomelib._tcp.local")


def is_lan(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.is_private and not ip.is_loopback and not ip.is_multicast


def require_lan(address: str) -> str:
    if not is_lan(address):
        raise ValueError(f"{address} is not an address on your home network")
    return address


def local_ipv4() -> list[str]:
    found = set()
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))          # no packet is sent: this only picks the LAN interface
        found.add(s.getsockname()[0])
    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    return sorted(a for a in found if is_lan(a))


def broadcasts() -> list[str]:
    """The /24 broadcast of each interface (home networks), plus the all-ones broadcast."""
    out = {"255.255.255.255"}
    for ip in local_ipv4():
        out.add(str(ipaddress.ip_network(f"{ip}/24", strict=False).broadcast_address))
    return sorted(out)


def _send(s: socket.socket, payload: bytes, targets: list[tuple[str, int]]) -> None:
    for target in targets:
        if ipaddress.ip_address(target[0]).is_multicast:
            for ip in local_ipv4():              # every interface: the default may be a VPN or VM adapter
                with contextlib.suppress(OSError):
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                    s.sendto(payload, target)
            continue
        with contextlib.suppress(OSError):
            s.sendto(payload, target)


def udp_query(payload: bytes, targets: list[tuple[str, int]], timeout: float, bind_port: int = 0,
              reply_filter=None, resend: bool = False) -> list[tuple[bytes, str]]:
    """Send one datagram to each target; collect replies (bytes, sender) from LAN addresses until timeout.
    resend: send again halfway (multicast over Wi-Fi drops packets; a device missed once answers the second)."""
    replies: list[tuple[bytes, str]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        s.bind(("", bind_port))
        _send(s, payload, targets)
        start = time.monotonic()
        end, again = start + timeout, (start + timeout / 2 if resend else None)
        while (left := end - time.monotonic()) > 0:
            if again is not None and time.monotonic() >= again:
                _send(s, payload, targets)
                again = None
            s.settimeout(min(left, again - time.monotonic()) if again is not None else left)
            try:
                data, (addr, _port) = s.recvfrom(65535)
            except TimeoutError:
                break
            except OSError:                  # Windows: ICMP "port unreachable" from a unicast target
                continue
            if is_lan(addr) and (reply_filter is None or reply_filter(data)):
                replies.append((data, addr))
    return replies


def udp_request(payload: bytes, address: str, port: int, timeout: float = 2.0) -> bytes:
    """One request, one reply, from that device only."""
    require_lan(address)
    for data, addr in udp_query(payload, [(address, port)], timeout):
        if addr == address:
            return data
    raise TimeoutError(f"no answer from {address}")


def http_get(url: str, timeout: float = 3.0, method: str = "GET", body: bytes | None = None) -> bytes:
    """HTTP to a device on the LAN (never anywhere else)."""
    host = urllib.request.urlparse(url).hostname or ""
    require_lan(host)
    req = urllib.request.Request(url, data=body, method=method,  # noqa: S310 - LAN-only, checked above
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read(1_000_000)


# ---------------------------------------------------------------- SSDP (UPnP)
def ssdp(timeout: float) -> dict[str, dict]:
    """address -> headers of its SSDP replies (lower-case keys), for everything that answers."""
    msg = (b"M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\n"
           b"MX: 2\r\nST: ssdp:all\r\n\r\n")
    out: dict[str, dict] = {}
    for data, addr in udp_query(msg, [SSDP_ADDR], timeout, resend=True):
        headers = out.setdefault(addr, {"st": []})
        for line in data.decode("utf-8", "replace").split("\r\n")[1:]:
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if key == "st":
                headers["st"].append(value)
            else:
                headers.setdefault(key, value)
    return out


def describe(location: str) -> dict:
    """friendlyName / manufacturer / modelName from a UPnP device description."""
    try:
        root = ET.fromstring(http_get(location, timeout=2))  # noqa: S314
    except Exception:                       # a device that answers badly is just left unnamed
        return {}
    out = {}
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in ("friendlyName", "manufacturer", "modelName") and tag not in out and node.text:
            out[tag] = node.text.strip()
    return out


# ---------------------------------------------------------------- mDNS (Bonjour)
def _qname(name: str) -> bytes:
    return b"".join(bytes([len(p)]) + p.encode() for p in name.split(".")) + b"\0"


def _read_name(data: bytes, pos: int, depth: int = 0) -> tuple[str, int]:
    labels, end = [], None
    while depth < 20:
        n = data[pos]
        if n == 0:
            pos += 1
            break
        if n & 0xC0 == 0xC0:                 # compression pointer
            ptr = struct.unpack_from(">H", data, pos)[0] & 0x3FFF
            end = end or pos + 2
            pos, depth = ptr, depth + 1
            continue
        labels.append(data[pos + 1:pos + 1 + n].decode("utf-8", "replace"))
        pos += 1 + n
    return ".".join(labels), (end or pos)


def parse_mdns(data: bytes) -> list[tuple[str, int, object]]:
    """(name, type, value) for every record: PTR -> name, TXT -> dict, A -> address, SRV -> port."""
    _id, _flags, qd, an, ns, ar = struct.unpack_from(">6H", data, 0)
    pos, records = 12, []
    for _ in range(qd):
        _, pos = _read_name(data, pos)
        pos += 4
    for _ in range(an + ns + ar):
        name, pos = _read_name(data, pos)
        rtype, _cls, _ttl, length = struct.unpack_from(">HHIH", data, pos)
        pos += 10
        rdata = data[pos:pos + length]
        if rtype == 12:
            records.append((name, rtype, _read_name(data, pos)[0]))
        elif rtype == 16:
            txt, i = {}, 0
            while i < len(rdata):
                n = rdata[i]
                k, _, v = rdata[i + 1:i + 1 + n].decode("utf-8", "replace").partition("=")
                txt[k] = v
                i += 1 + n
            records.append((name, rtype, txt))
        elif rtype == 1 and length == 4:
            records.append((name, rtype, socket.inet_ntoa(rdata)))
        elif rtype == 33 and length >= 6:                     # SRV: the service's port
            records.append((name, rtype, struct.unpack_from(">H", rdata, 4)[0]))
        pos += length
    return records


def mdns(timeout: float, services: tuple[str, ...] = MDNS_SERVICES) -> dict[str, dict]:
    """address -> {"services": {service: instance}, "txt": {...}, "ports": {service: port}}."""
    questions = b"".join(_qname(s) + struct.pack(">HH", 12, 0x8001) for s in services)  # PTR, unicast reply
    query = struct.pack(">6H", 0, 0, len(services), 0, 0, 0) + questions
    out: dict[str, dict] = {}
    for data, addr in udp_query(query, [MDNS_ADDR], timeout, resend=True):
        try:
            records = parse_mdns(data)
        except (struct.error, IndexError):
            continue
        entry = out.setdefault(addr, {"services": {}, "txt": {}, "ports": {}})
        for name, rtype, value in records:
            if rtype == 12 and name in services:
                entry["services"][name] = re.sub(r"\._.*$", "", str(value))
            elif rtype == 16:
                entry["txt"].update(value)
            elif rtype == 33:
                service = next((s for s in services if name.endswith(s)), name)
                entry["ports"][service] = value
    return out
