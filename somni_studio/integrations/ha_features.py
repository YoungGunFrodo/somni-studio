"""More of Home Assistant, for Somni's tools: its own assistant, rooms, routines, devices it has
discovered, and playing sound on its speakers.

Everything goes through Home Assistant's documented APIs with Somni's sign-in:
- its assistant ("Assist"): POST /api/conversation/process. It already understands rooms, groups,
  "all the lights", timers and scenes, so plain words cover most of what Home Assistant can do;
- rooms ("areas"): read with a template (POST /api/template), as Home Assistant has no REST list;
- routines ("automations"): /api/config/automation/config/<id>, the same API its own editor uses;
- discovered devices ("config flows"): listed over its websocket API, added over REST.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid

from .base import CheckFailed
from .home_assistant import TIMEOUT_S, token_for

AREAS_TTL_S = 60
_areas_cache: dict = {}               # url -> (fetched at, {entity_id: room})


def _call(config: dict, method: str, path: str, body: dict | None = None, raw: bool = False):
    url = (config.get("url") or "").rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise CheckFailed("the Home Assistant address must start with http:// or https://")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method,  # noqa: S310 - scheme checked above
                                 headers={"Authorization": f"Bearer {token_for(config)}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S * 2) as r:  # noqa: S310 - scheme checked above
            text = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise CheckFailed(f"Home Assistant said {exc.code}: {detail or exc.reason}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise CheckFailed(f"cannot reach Home Assistant: {getattr(exc, 'reason', exc)}") from None
    if raw:
        return text
    return json.loads(text) if text.strip() else None


# ---------------------------------------------------------------- its assistant
def assist(config: dict, text: str) -> dict:
    """Hand a request in plain words to Home Assistant's assistant; return what it did and said."""
    reply = _call(config, "POST", "/api/conversation/process", {"text": text, "language": "en"}) or {}
    response = reply.get("response") or {}
    speech = ((response.get("speech") or {}).get("plain") or {}).get("speech", "")
    kind = response.get("response_type", "")
    data = response.get("data") or {}
    out = {"asked": text, "home_assistant_said": speech, "result": kind}
    done = [t.get("name") for t in data.get("success", []) if t.get("name")]
    failed = [t.get("name") for t in data.get("failed", []) if t.get("name")]
    if done:
        out["changed"] = done
    if failed:
        out["failed"] = failed
        out["note"] = ("These did NOT change: " + ", ".join(failed) + ". Home Assistant could not reach or control "
                       "them. Tell the user plainly; do not say it worked.")
        devices_done = [t.get("name") for t in data.get("success", []) if t.get("type") == "entity"]
        if kind == "action_done":                     # its speech says "Turned off the lights" regardless
            out["result"] = "partly done" if devices_done else "failed"
            out["home_assistant_said"] = f"(it said '{speech}', but the devices did not change)"
    if kind == "error":
        code = data.get("code", "")
        out["note"] = ("Home Assistant did not understand or could not do it"
                       + (f" ({code})" if code else "") + "; try other words or a specific device")
    return out


# ---------------------------------------------------------------- rooms
AREAS_TEMPLATE = (
    "{% set ns = namespace(rows=[]) %}"
    "{% for a in areas() %}"
    "{% set ids = area_entities(a) %}"
    "{% for d in area_devices(a) %}{% set ids = ids + device_entities(d) %}{% endfor %}"
    "{% set ns.rows = ns.rows + [{'room': area_name(a), 'entities': ids | unique | list}] %}"
    "{% endfor %}"
    "{{ ns.rows | tojson }}")


def rooms(config: dict) -> dict[str, str]:
    """entity_id -> the room it is in (Home Assistant's areas). Cached for a minute."""
    url = config.get("url") or ""
    cached = _areas_cache.get(url)
    if cached and time.monotonic() - cached[0] < AREAS_TTL_S:
        return cached[1]
    text = _call(config, "POST", "/api/template", {"template": AREAS_TEMPLATE}, raw=True)
    try:
        rows = json.loads(text)
    except ValueError:
        rows = []
    out = {}
    for row in rows if isinstance(rows, list) else []:
        for entity in row.get("entities") or []:
            out.setdefault(entity, row.get("room", ""))
    _areas_cache[url] = (time.monotonic(), out)
    return out


def room_names(config: dict) -> list[str]:
    return sorted(set(rooms(config).values()))


# ---------------------------------------------------------------- routines
DAYS = {"mon": "mon", "tue": "tue", "wed": "wed", "thu": "thu", "fri": "fri", "sat": "sat", "sun": "sun"}
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"]
WEEKEND = ["sat", "sun"]
TIME = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", re.I)
SOMNI_TAG = "Made by Somni"


def _days(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", (text or "").lower())
    if not words or {"every", "day", "daily", "everyday"} >= set(words):
        return []
    out = []
    for w in words:
        if w in ("weekday", "weekdays"):
            out += WEEKDAYS
        elif w in ("weekend", "weekends"):
            out += WEEKEND
        elif w[:3] in DAYS:
            out.append(DAYS[w[:3]])
    return list(dict.fromkeys(out))


def _clock(text: str) -> str:
    m = TIME.match(text or "")
    if not m:
        raise CheckFailed(f"'{text}' is not a time; use 22:00, 7:30am, sunrise or sunset")
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if half == "pm" and hour < 12:
        hour += 12
    if half == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        raise CheckFailed(f"'{text}' is not a time of day")
    return f"{hour:02d}:{minute:02d}:00"


def routine_config(name: str, do: str, at: str = "", days: str = "", offset_minutes: int = 0,
                   device: str = "", becomes: str = "") -> dict:
    """A Home Assistant automation: when <at time / sunrise / sunset / a device changes>, on <days>,
    have Home Assistant's assistant do <do> (plain words, so any request it understands works)."""
    if not do.strip():
        raise CheckFailed("say what the routine should do, e.g. 'turn off all the lights'")
    when = at.strip().lower()
    if device:
        trigger = {"trigger": "state", "entity_id": device}
        if becomes:
            trigger["to"] = becomes
    elif when in ("sunrise", "sunset"):
        trigger = {"trigger": "sun", "event": when}
        if offset_minutes:
            sign = "-" if offset_minutes < 0 else ""
            trigger["offset"] = f"{sign}00:{abs(int(offset_minutes)):02d}:00"
    elif when:
        trigger = {"trigger": "time", "at": _clock(when)}
    else:
        raise CheckFailed("say when: a time (22:00), sunrise, sunset, or a device and what it becomes")
    conditions = [{"condition": "time", "weekday": _days(days)}] if _days(days) else []
    return {"alias": name.strip()[:80] or do.strip()[:60], "description": SOMNI_TAG,
            "triggers": [trigger], "conditions": conditions,
            "actions": [{"action": "conversation.process", "data": {"text": do.strip(), "language": "en"}}],
            "mode": "single"}


def save_routine(config: dict, body: dict) -> dict:
    routine_id = uuid.uuid4().hex[:13]
    _call(config, "POST", f"/api/config/automation/config/{routine_id}", body)
    return {"created": body["alias"], "id": routine_id}


def routines(config: dict) -> list[dict]:
    """The automations Home Assistant has (ours and the user's own)."""
    rows = _call(config, "GET", "/api/states") or []
    out = []
    for s in rows:
        if not str(s.get("entity_id", "")).startswith("automation."):
            continue
        attrs = s.get("attributes") or {}
        out.append({"name": attrs.get("friendly_name") or s["entity_id"], "id": attrs.get("id", ""),
                    "on": s.get("state") == "on", "last_ran": attrs.get("last_triggered")})
    return out


def delete_routine(config: dict, name: str) -> dict:
    words = name.lower().strip()
    found = [r for r in routines(config) if r["id"] and (words == r["id"] or words in r["name"].lower())]
    if len(found) != 1:
        names = ", ".join(r["name"] for r in found) or ", ".join(r["name"] for r in routines(config)) or "none"
        raise CheckFailed(f"'{name}' matches {len(found)} routines ({names})")
    _call(config, "DELETE", f"/api/config/automation/config/{found[0]['id']}")
    return {"deleted": found[0]["name"]}


# ---------------------------------------------------------------- discovered devices
def _ws(config: dict, message: dict) -> object:
    """One request over Home Assistant's websocket API (some things have no REST endpoint)."""
    from websockets.sync.client import connect
    url = (config.get("url") or "").rstrip("/")
    ws_url = re.sub(r"^http", "ws", url) + "/api/websocket"
    try:
        with connect(ws_url, open_timeout=TIMEOUT_S, close_timeout=2) as ws:
            json.loads(ws.recv(timeout=TIMEOUT_S))                          # auth_required
            ws.send(json.dumps({"type": "auth", "access_token": token_for(config)}))
            if json.loads(ws.recv(timeout=TIMEOUT_S)).get("type") != "auth_ok":
                raise CheckFailed("Home Assistant refused Somni's sign-in; sign in again")
            ws.send(json.dumps({"id": 1, **message}))
            reply = json.loads(ws.recv(timeout=TIMEOUT_S * 2))
    except (OSError, TimeoutError) as exc:
        raise CheckFailed(f"cannot reach Home Assistant: {exc}") from None
    if not reply.get("success"):
        raise CheckFailed(f"Home Assistant said: {(reply.get('error') or {}).get('message', 'no')}")
    return reply.get("result")


def _flow_name(flow: dict) -> str:
    placeholders = (flow.get("context") or {}).get("title_placeholders") or {}
    name = placeholders.get("name") or placeholders.get("title") or ""
    handler = flow.get("handler", "")
    return f"{name} ({handler})" if name and handler and handler not in name.lower() else name or handler


def discovered(config: dict) -> list[dict]:
    """Devices and services Home Assistant found by itself and is waiting for someone to add."""
    flows = _ws(config, {"type": "config_entries/flow/progress"}) or []
    return [{"name": _flow_name(f), "flow_id": f.get("flow_id"), "kind": f.get("handler", "")}
            for f in flows if (f.get("context") or {}).get("source") not in ("user", "reauth", "reconfigure")]


def add_discovered(config: dict, name: str) -> dict:
    """Add discovered devices: one by name, or 'all'. Ones that need more (a login, a code) are
    reported so the user can finish them in Home Assistant."""
    waiting = discovered(config)
    if not waiting:
        return {"added": [], "note": "Home Assistant has nothing waiting to be added"}
    words = name.lower().strip()
    pick = waiting if words in ("", "all", "everything", "all of them") else \
        [w for w in waiting if words in w["name"].lower() or words in w["kind"].lower()]
    if not pick:
        raise CheckFailed(f"nothing waiting matches '{name}'; waiting: {', '.join(w['name'] for w in waiting)}")
    added, needs = [], []
    for item in pick:
        step = _call(config, "POST", f"/api/config/config_entries/flow/{item['flow_id']}", {}) or {}
        if step.get("type") == "create_entry":
            added.append(item["name"])
        elif step.get("type") == "abort":
            needs.append(f"{item['name']}: {step.get('reason', 'stopped')}")
        else:
            fields = [f.get("name") for f in step.get("data_schema") or [] if f.get("name")]
            needs.append(f"{item['name']}: needs {', '.join(fields) or 'more information'}")
    out = {"added": added}
    if needs:
        out["needs_user"] = needs
        out["note"] = ("finish these in Home Assistant (Settings → Devices & services); "
                       "Somni never types passwords or codes for the user")
    return out


def known_macs(config: dict) -> set[str]:
    """The network (MAC) addresses of every device Home Assistant has, as 12 lowercase hex digits."""
    macs = set()
    for device in _ws(config, {"type": "config/device_registry/list"}) or []:
        for kind, value in device.get("connections") or []:
            if kind == "mac":
                macs.add(re.sub(r"[^0-9a-f]", "", str(value).lower()))
    return macs


def missed_wiz(config: dict) -> list[dict]:
    """WiZ bulbs answering on the network that Home Assistant has not added (its discovery listens for
    a bulb joining the network, so one set up before, or while it was off, can be missed)."""
    from ..devices.lights import Wiz
    bulbs = Wiz().discover(3.0, {})
    if not bulbs:
        return []
    known = known_macs(config)
    return [{"name": f"WiZ bulb {f.ident[-4:]} at {f.address}", "host": f.address, "kind": "wiz"}
            for f in bulbs if re.sub(r"[^0-9a-f]", "", f.ident.lower()) not in known]


def add_by_address(config: dict, handler: str, host: str) -> dict:
    """Add a device Home Assistant connects to by its address (WiZ, ...): its 'Add integration' form."""
    step = _call(config, "POST", "/api/config/config_entries/flow", {"handler": handler}) or {}
    for _ in range(3):
        if step.get("type") != "form":
            break
        fields = [x.get("name") for x in step.get("data_schema") or []]
        if "host" not in fields:
            break
        step = _call(config, "POST", f"/api/config/config_entries/flow/{step['flow_id']}", {"host": host}) or {}
        if step.get("errors"):
            raise CheckFailed(f"Home Assistant could not add {host}: {step['errors']}")
    if step.get("type") == "create_entry":
        return {"added": step.get("title") or host}
    raise CheckFailed(f"Home Assistant did not add {host} ({step.get('reason') or step.get('step_id') or 'stopped'})")


# ---------------------------------------------------------------- names and rooms
NOT_MAIN = ("sensor", "update", "event", "button", "number", "select")


def _entity(config: dict, target: str) -> str:
    """The entity a name or id means. A made-up id (a small model guesses ids) is searched by its words."""
    from .home_assistant import HomeAssistant
    target = target.strip()
    if re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", target):
        try:
            _ws(config, {"type": "config/entity_registry/get", "entity_id": target})
            return target
        except CheckFailed:
            target = target.split(".", 1)[1].replace("_", " ")
    found = [r for r in HomeAssistant().states(config, target) if r["entity_id"].split(".", 1)[0] not in NOT_MAIN]
    if len(found) != 1:
        names = ", ".join(f"{r['name']} ({r['entity_id']})" for r in found[:8])
        raise CheckFailed(f"'{target}' matches {len(found)} devices" + (f": {names}" if names else ""))
    return found[0]["entity_id"]


NEW_WORDS = re.compile(r"\b(new|newest|latest|other|second|last|wiz|bulb|light|lamp|that one|it)\b", re.I)


def _add_if_missing(config: dict, target: str) -> str | None:
    """'The new bulb' that is not in Home Assistant yet: add it, and return its light's entity id."""
    from .home_assistant import HomeAssistant
    missed = missed_wiz(config)
    if not missed:
        return None
    t = target.lower()
    pick = [b for b in missed if b["host"] in t or b["name"].split()[2] in t]
    if not pick and len(missed) == 1 and NEW_WORDS.search(t):
        pick = missed
    if len(pick) != 1:
        return None
    title = add_by_address(config, "wiz", pick[0]["host"])["added"]
    for _ in range(10):                                # Home Assistant makes its entities a moment later
        rows = [r for r in HomeAssistant().states(config, title) if r["entity_id"].startswith("light.")]
        if len(rows) == 1:
            return rows[0]["entity_id"]
        time.sleep(1)
    raise CheckFailed(f"added {title} to Home Assistant, but its light has not appeared yet; try again")


def _area_id(config: dict, room: str) -> str:
    """The id of a room, made if it doesn't exist yet."""
    want = room.strip()
    for area in _ws(config, {"type": "config/area_registry/list"}) or []:
        if area.get("name", "").lower() == want.lower():
            return area["area_id"]
    return (_ws(config, {"type": "config/area_registry/create", "name": want[:1].upper() + want[1:]}) or {})["area_id"]


def organize(config: dict, target: str, name: str = "", room: str = "") -> dict:
    """Rename a device and/or put it in a room. The device (not just one of its entities) is changed,
    so its other parts (a bulb's signal sensor, say) follow."""
    if not name.strip() and not room.strip():
        raise CheckFailed("say what to call it or which room it is in")
    added = None
    try:
        entity = _entity(config, target)
    except CheckFailed:
        added = entity = _add_if_missing(config, target)
        if entity is None:
            raise
    entry = _ws(config, {"type": "config/entity_registry/get", "entity_id": entity}) or {}
    change: dict = {}
    if room.strip():
        change["area_id"] = _area_id(config, room)
    if entry.get("device_id"):
        if name.strip():
            change["name_by_user"] = name.strip()[:60]
        _ws(config, {"type": "config/device_registry/update", "device_id": entry["device_id"], **change})
    else:
        if name.strip():
            change["name"] = name.strip()[:60]
        _ws(config, {"type": "config/entity_registry/update", "entity_id": entity, **change})
    _areas_cache.clear()
    out = {"organized": entity}
    if added:
        out["added_to_home_assistant"] = True
    if name.strip():
        out["name"] = name.strip()
    if room.strip():
        out["room"] = room.strip()
    return out


# ---------------------------------------------------------------- sound on its speakers
def speakers(config: dict) -> list[dict]:
    rows = _call(config, "GET", "/api/states") or []
    return [{"entity_id": s["entity_id"], "name": (s.get("attributes") or {}).get("friendly_name") or s["entity_id"],
             "state": s.get("state")} for s in rows if str(s.get("entity_id", "")).startswith("media_player.")]


def play_url(config: dict, entity_id: str, url: str, content_type: str = "music") -> None:
    _call(config, "POST", "/api/services/media_player/play_media",
          {"entity_id": entity_id, "media_content_id": url, "media_content_type": content_type})
