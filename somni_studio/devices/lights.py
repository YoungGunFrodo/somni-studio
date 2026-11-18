"""Smart lights and plugs that can be controlled on your own network, without their cloud.

  WiZ (Philips WiZ, many store brands)  UDP 38899, no setup
  LIFX                                  UDP 56700, no setup
  TP-Link Kasa (older firmware)          TCP 9999, no setup  (Tapo and newer Kasa need a TP-Link login: not here)
  Philips Hue                           bridge HTTP API; press the bridge's button once
  Yeelight                              TCP 55443; switch on "LAN Control" in the Yeelight app
  Govee                                 UDP 4003; switch on "LAN Control" in the Govee Home app
  Shelly                                HTTP; no setup (unless you gave it a password)
"""
from __future__ import annotations

import json
import socket
import struct

from . import net
from .base import Device, DeviceError, Driver, Found, NeedsUser


def _pct(value: int | None, low: int = 1) -> int | None:
    return None if value is None else max(low, min(100, int(value)))


# ---------------------------------------------------------------- WiZ
class Wiz(Driver):
    id, title = "wiz", "WiZ"
    PORT = 38899

    def discover(self, timeout, seen):
        msg = json.dumps({"method": "getPilot", "params": {}}).encode()
        out = {}
        for data, addr in net.udp_query(msg, [(b, self.PORT) for b in net.broadcasts()], timeout):
            try:
                result = json.loads(data).get("result") or {}
            except ValueError:
                continue
            mac = str(result.get("mac", "")).lower()
            if mac:
                out[mac] = Found(self.id, addr, f"WiZ light {mac[-4:]}", "WiZ", mac)
        return list(out.values())

    def _call(self, link, method, params=None):
        reply = net.udp_request(json.dumps({"method": method, "params": params or {}}).encode(),
                                link["address"], self.PORT)
        data = json.loads(reply)
        if "error" in data:
            raise DeviceError(f"the WiZ light said: {data['error'].get('message', data['error'])}")
        return data.get("result") or {}

    def devices(self, link):
        r = self._call(link, "getPilot")
        return [Device(self.device_id(link), link["name"], "light", bool(r.get("state")), r.get("dimming"),
                       link=link["id"], model="WiZ")]

    def switch(self, link, sub, on, brightness):
        params = {}
        if on is not None:
            params["state"] = on
        if brightness is not None:
            params.update(state=True, dimming=_pct(brightness, 10))
        self._call(link, "setPilot", params)


# ---------------------------------------------------------------- LIFX
class Lifx(Driver):
    id, title = "lifx", "LIFX"
    PORT = 56700
    SOURCE = 0x534F4D4E            # "SOMN"
    GET, STATE, SET_POWER, SET_COLOR = 101, 107, 117, 102

    @classmethod
    def packet(cls, kind: int, payload: bytes = b"", target: bytes = b"\0" * 8, tagged: bool = False) -> bytes:
        protocol = 1024 | 0x1000 | (0x2000 if tagged else 0)
        return (struct.pack("<HHI", 36 + len(payload), protocol, cls.SOURCE)
                + struct.pack("<8s6sBB", target, b"\0" * 6, 0x01, 0)
                + struct.pack("<QHH", 0, kind, 0) + payload)

    @staticmethod
    def parse_state(data: bytes) -> dict | None:
        if len(data) < 36 + 52 or struct.unpack_from("<H", data, 32)[0] != Lifx.STATE:
            return None
        hue, sat, bri, kelvin, _, power = struct.unpack_from("<HHHHhH", data, 36)
        label = data[48:80].split(b"\0", 1)[0].decode("utf-8", "replace")
        return {"mac": data[8:14].hex(), "hsbk": (hue, sat, bri, kelvin), "on": power > 0, "label": label}

    def discover(self, timeout, seen):
        out = {}
        for data, addr in net.udp_query(self.packet(self.GET, tagged=True),
                                        [(b, self.PORT) for b in net.broadcasts()], timeout):
            state = self.parse_state(data)
            if state:
                out[state["mac"]] = Found(self.id, addr, state["label"] or f"LIFX {state['mac'][-4:]}", "LIFX",
                                          state["mac"])
        return list(out.values())

    def _state(self, link):
        target = bytes.fromhex(link["ident"]) + b"\0\0"
        state = self.parse_state(net.udp_request(self.packet(self.GET, target=target), link["address"], self.PORT))
        if state is None:
            raise DeviceError("the LIFX light gave an answer I do not understand")
        return state, target

    def devices(self, link):
        state, _ = self._state(link)
        return [Device(self.device_id(link), link["name"], "light", state["on"],
                       round(state["hsbk"][2] * 100 / 65535), link=link["id"], model="LIFX")]

    def switch(self, link, sub, on, brightness):
        state, target = self._state(link)
        if brightness is not None:
            hue, sat, _, kelvin = state["hsbk"]
            payload = struct.pack("<BHHHHI", 0, hue, sat, round(_pct(brightness) * 65535 / 100), kelvin, 300)
            net.udp_query(self.packet(self.SET_COLOR, payload, target), [(link["address"], self.PORT)], 0.3)
            on = True
        if on is not None:
            payload = struct.pack("<HI", 65535 if on else 0, 300)
            net.udp_query(self.packet(self.SET_POWER, payload, target), [(link["address"], self.PORT)], 0.3)


# ---------------------------------------------------------------- TP-Link Kasa (legacy protocol)
class Kasa(Driver):
    id, title = "kasa", "TP-Link Kasa"
    kinds = ("switch", "light")
    PORT = 9999
    BULB = "smartlife.iot.smartbulb.lightingservice"
    DIMMER = "smartlife.iot.dimmer"

    @staticmethod
    def encrypt(data: bytes) -> bytes:
        key, out = 171, bytearray()
        for c in data:
            key ^= c
            out.append(key)
        return bytes(out)

    @staticmethod
    def decrypt(data: bytes) -> bytes:
        key, out = 171, bytearray()
        for c in data:
            out.append(key ^ c)
            key = c
        return bytes(out)

    def discover(self, timeout, seen):
        msg = self.encrypt(b'{"system":{"get_sysinfo":{}}}')
        out = {}
        for data, addr in net.udp_query(msg, [(b, self.PORT) for b in net.broadcasts()], timeout):
            try:
                info = json.loads(self.decrypt(data))["system"]["get_sysinfo"]
            except (ValueError, KeyError, TypeError):
                continue
            ident = str(info.get("deviceId") or info.get("mac") or info.get("mic_mac") or addr).lower()
            out[ident] = Found(self.id, addr, info.get("alias") or "Kasa device", info.get("model", "Kasa"), ident)
        return list(out.values())

    def _send(self, link, request: dict) -> dict:
        body = self.encrypt(json.dumps(request).encode())
        net.require_lan(link["address"])
        with socket.create_connection((link["address"], self.PORT), timeout=3) as s:
            s.sendall(struct.pack(">I", len(body)) + body)
            size = struct.unpack(">I", self._read(s, 4))[0]
            return json.loads(self.decrypt(self._read(s, size)))

    @staticmethod
    def _read(s, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise DeviceError("the Kasa device closed the connection")
            buf += chunk
        return buf

    def devices(self, link):
        info = self._send(link, {"system": {"get_sysinfo": {}}})["system"]["get_sysinfo"]
        model = info.get("model", "Kasa")
        if info.get("children"):                              # power strips: one device per outlet
            return [Device(self.device_id(link, c["id"]), c.get("alias") or f"{link['name']} {i + 1}", "switch",
                           bool(c.get("state")), link=link["id"], model=model)
                    for i, c in enumerate(info["children"])]
        if "light_state" in info:
            ls = info["light_state"]
            return [Device(self.device_id(link), link["name"], "light", bool(ls.get("on_off")),
                           ls.get("brightness") if ls.get("on_off") else ls.get("dft_on_state", {}).get("brightness"),
                           link=link["id"], model=model)]
        dimmable = bool(info.get("brightness") is not None and "HS220" in model)
        return [Device(self.device_id(link), link["name"], "light" if dimmable else "switch",
                       bool(info.get("relay_state")), info.get("brightness") if dimmable else None,
                       link=link["id"], model=model)]

    def switch(self, link, sub, on, brightness):
        context = {"context": {"child_ids": [sub]}} if sub else {}
        info = self._send(link, {"system": {"get_sysinfo": {}}})["system"]["get_sysinfo"]
        if "light_state" in info:
            state = {"transition_period": 300}
            if on is not None:
                state["on_off"] = int(on)
            if brightness is not None:
                state.update(on_off=1, brightness=_pct(brightness))
            self._send(link, {self.BULB: {"transition_light_state": state}})
            return
        if brightness is not None and info.get("brightness") is not None:
            self._send(link, {self.DIMMER: {"set_brightness": {"brightness": _pct(brightness)}}})
            on = True
        if on is not None:
            self._send(link, {**context, "system": {"set_relay_state": {"state": int(on)}}})


# ---------------------------------------------------------------- Philips Hue (bridge)
class Hue(Driver):
    id, title = "hue", "Philips Hue"
    APP = "somni_studio#home"

    def discover(self, timeout, seen):
        out = []
        for addr, info in seen.items():
            headers = info.get("ssdp", {})
            services = info.get("mdns", {}).get("services", {})
            if not ("hue-bridgeid" in headers or "IpBridge" in headers.get("server", "")
                    or "_hue._tcp.local" in services):
                continue
            try:
                config = json.loads(net.http_get(f"http://{addr}/api/config", timeout=2))
            except (OSError, ValueError):
                continue
            out.append(Found(self.id, addr, config.get("name") or "Hue Bridge", config.get("modelid", "Hue Bridge"),
                             str(config.get("bridgeid", addr)).lower(),
                             note="press the round button on top of the Hue bridge, then connect"))
        return out

    def connect(self, found):
        reply = json.loads(net.http_get(f"http://{found.address}/api", method="POST",
                                        body=json.dumps({"devicetype": self.APP}).encode()))
        item = reply[0] if isinstance(reply, list) and reply else {}
        if "success" in item:
            return {"key": item["success"]["username"]}
        if (item.get("error") or {}).get("type") == 101:
            raise NeedsUser("Press the round link button on top of your Hue bridge, then ask me again within "
                            "30 seconds.")
        raise DeviceError(f"the Hue bridge said: {(item.get('error') or {}).get('description', reply)}")

    def _get(self, link, path):
        data = json.loads(net.http_get(f"http://{link['address']}/api/{link['key']}{path}"))
        if isinstance(data, list) and data and "error" in data[0]:
            raise DeviceError(f"the Hue bridge said: {data[0]['error'].get('description')}")
        return data

    def devices(self, link):
        out = []
        for light_id, light in self._get(link, "/lights").items():
            state = light.get("state", {})
            dimmable = "bri" in state
            out.append(Device(self.device_id(link, f"light/{light_id}"), light.get("name", f"Hue light {light_id}"),
                              "light", bool(state.get("on")),
                              round(state["bri"] * 100 / 254) if dimmable else None, bool(state.get("reachable", True)),
                              link["id"], light.get("modelid", "Hue")))
        for group_id, group in self._get(link, "/groups").items():
            if group.get("type") not in ("Room", "Zone"):
                continue
            action = group.get("action", {})
            out.append(Device(self.device_id(link, f"group/{group_id}"), group.get("name", f"Room {group_id}"),
                              "light", bool(group.get("state", {}).get("any_on")),
                              round(action["bri"] * 100 / 254) if "bri" in action else None, True, link["id"],
                              "Hue room"))
        return out

    def switch(self, link, sub, on, brightness):
        kind, _, number = sub.partition("/")
        body = {}
        if on is not None:
            body["on"] = on
        if brightness is not None:
            body.update(on=True, bri=max(1, round(_pct(brightness) * 254 / 100)))
        path = f"/lights/{number}/state" if kind == "light" else f"/groups/{number}/action"
        net.http_get(f"http://{link['address']}/api/{link['key']}{path}", method="PUT", body=json.dumps(body).encode())


# ---------------------------------------------------------------- Yeelight
class Yeelight(Driver):
    id, title = "yeelight", "Yeelight"
    PORT = 55443
    enable_hint = "Yeelight lights answer only with \"LAN Control\" switched on in the Yeelight app."

    def discover(self, timeout, seen):
        msg = (b'M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1982\r\nMAN: "ssdp:discover"\r\n'
               b"ST: wifi_bulb\r\n")
        out = {}
        for data, addr in net.udp_query(msg, [("239.255.255.250", 1982)], timeout):
            headers = {}
            for line in data.decode("utf-8", "replace").split("\r\n"):
                key, sep, value = line.partition(":")
                if sep:
                    headers[key.strip().lower()] = value.strip()
            ident = headers.get("id", addr).lower()
            out[ident] = Found(self.id, addr, headers.get("name") or f"Yeelight {ident[-4:]}",
                               f"Yeelight {headers.get('model', '')}".strip(), ident)
        return list(out.values())

    def _call(self, link, method, params):
        net.require_lan(link["address"])
        with socket.create_connection((link["address"], self.PORT), timeout=3) as s:
            s.sendall((json.dumps({"id": 1, "method": method, "params": params}) + "\r\n").encode())
            reply = json.loads(s.makefile().readline() or "{}")
        if "error" in reply:
            raise DeviceError(f"the Yeelight said: {reply['error'].get('message')}")
        return reply.get("result") or []

    def devices(self, link):
        power, bright = (self._call(link, "get_prop", ["power", "bright"]) + ["", ""])[:2]
        return [Device(self.device_id(link), link["name"], "light", power == "on",
                       int(bright) if str(bright).isdigit() else None, link=link["id"], model=link.get("model", ""))]

    def switch(self, link, sub, on, brightness):
        if brightness is not None:
            self._call(link, "set_power", ["on", "smooth", 300])
            self._call(link, "set_bright", [_pct(brightness), "smooth", 300])
        elif on is not None:
            self._call(link, "set_power", ["on" if on else "off", "smooth", 300])


# ---------------------------------------------------------------- Govee
class Govee(Driver):
    id, title = "govee", "Govee"
    enable_hint = "Govee lights answer only with \"LAN Control\" switched on in the Govee Home app."
    SCAN, LISTEN, CONTROL = 4001, 4002, 4003

    def _ask(self, targets, msg: dict, timeout: float):
        try:
            return net.udp_query(json.dumps({"msg": msg}).encode(), targets, timeout, bind_port=self.LISTEN)
        except OSError:                       # port 4002 taken (the Govee app on this PC): skip Govee
            return []

    def discover(self, timeout, seen):
        out = {}
        for data, addr in self._ask([("239.255.255.250", self.SCAN)],
                                    {"cmd": "scan", "data": {"account_topic": "reserve"}}, timeout):
            try:
                info = json.loads(data)["msg"]["data"]
            except (ValueError, KeyError, TypeError):
                continue
            ident = str(info.get("device", addr)).lower()
            out[ident] = Found(self.id, info.get("ip", addr), f"Govee {info.get('sku', 'light')}",
                               f"Govee {info.get('sku', '')}".strip(), ident)
        return list(out.values())

    def devices(self, link):
        for data, addr in self._ask([(link["address"], self.CONTROL)], {"cmd": "devStatus", "data": {}}, 2):
            if addr != link["address"]:
                continue
            status = json.loads(data)["msg"]["data"]
            return [Device(self.device_id(link), link["name"], "light", bool(status.get("onOff")),
                           status.get("brightness"), link=link["id"], model=link.get("model", ""))]
        raise DeviceError("the Govee light did not answer")

    def switch(self, link, sub, on, brightness):
        target = [(net.require_lan(link["address"]), self.CONTROL)]
        if on is not None or brightness is not None:
            net.udp_query(json.dumps({"msg": {"cmd": "turn", "data": {"value": int(on is not False)}}}).encode(),
                          target, 0.2)
        if brightness is not None:
            net.udp_query(json.dumps({"msg": {"cmd": "brightness", "data": {"value": _pct(brightness)}}}).encode(),
                          target, 0.2)


# ---------------------------------------------------------------- Shelly
class Shelly(Driver):
    id, title = "shelly", "Shelly"
    kinds = ("switch", "light")

    def discover(self, timeout, seen):
        out = []
        for addr, info in seen.items():
            names = " ".join(info.get("mdns", {}).get("services", {}).values()).lower()
            if "shelly" not in names:
                continue
            try:
                about = json.loads(net.http_get(f"http://{addr}/shelly", timeout=2))
            except (OSError, ValueError):
                continue
            locked = about.get("auth") or about.get("auth_en")
            out.append(Found(self.id, addr, about.get("name") or about.get("id") or about.get("type") or "Shelly",
                             about.get("model") or about.get("type", "Shelly"),
                             str(about.get("mac", addr)).lower(), extra={"gen": about.get("gen", 1)},
                             note="this Shelly has a password; turn off its login to use it here" if locked else ""))
        return out

    def connect(self, found):
        if found.note:
            raise NeedsUser("This Shelly is password protected. Turn off \"Restrict login\" in its web page, "
                            "then ask me again.")
        return {"gen": found.extra.get("gen", 1)}

    def _json(self, link, path):
        return json.loads(net.http_get(f"http://{link['address']}{path}"))

    def devices(self, link):
        out = []
        if link.get("gen", 1) >= 2:
            status = self._json(link, "/rpc/Shelly.GetStatus")
            for key, value in status.items():
                kind, _, index = key.partition(":")
                if kind in ("switch", "light") and isinstance(value, dict):
                    name = link["name"] if index == "0" else f"{link['name']} {int(index) + 1}"
                    out.append(Device(self.device_id(link, key), name, kind, bool(value.get("output")),
                                      value.get("brightness"), link=link["id"], model=link.get("model", "")))
            return out
        status = self._json(link, "/status")
        for i, relay in enumerate(status.get("relays", [])):
            out.append(Device(self.device_id(link, f"relay:{i}"), link["name"] if i == 0 else f"{link['name']} {i + 1}",
                              "switch", bool(relay.get("ison")), link=link["id"], model=link.get("model", "")))
        for i, light in enumerate(status.get("lights", [])):
            out.append(Device(self.device_id(link, f"light:{i}"), link["name"] if i == 0 else f"{link['name']} {i + 1}",
                              "light", bool(light.get("ison")), light.get("brightness"), link=link["id"],
                              model=link.get("model", "")))
        return out

    def switch(self, link, sub, on, brightness):
        kind, _, index = sub.partition(":")
        base = f"http://{link['address']}"
        if link.get("gen", 1) >= 2:
            method = "Light.Set" if kind == "light" else "Switch.Set"
            query = f"id={index}&on={'true' if on is not False else 'false'}"
            if brightness is not None and kind == "light":
                query += f"&brightness={_pct(brightness)}"
            net.http_get(f"{base}/rpc/{method}?{query}")
            return
        path = "light" if kind == "light" else "relay"
        query = f"turn={'on' if on is not False else 'off'}"
        if brightness is not None and kind == "light":
            query += f"&brightness={_pct(brightness)}"
        net.http_get(f"{base}/{path}/{index}?{query}")
