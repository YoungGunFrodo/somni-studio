"""The tools, what they need, and how they run."""
from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..integrations import ha_features
from ..integrations.base import CheckFailed, State
from ..integrations.custom_api import CustomApi
from ..integrations.email_imap import EmailImap
from ..integrations.google_home import SDK_DOMAIN
from ..integrations.home_assistant import HomeAssistant
from ..integrations.web import WebLookup

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 3000          # what the model gets back from one call, at most


class ToolFailed(RuntimeError):
    """A tool could not do what was asked; the message goes back to the model and the user."""


@dataclass
class ToolContext:
    """What tools may touch. Built per question by the provider."""
    integration_config: Callable[[str], dict]          # integration id -> stored config (with secrets)
    integration_ready: Callable[[str], bool]
    allowed_domains: frozenset[str]
    remember: Callable[[str], int] | None = None        # log a statement for the next sleep
    request_sleep: Callable[[], None] | None = None     # sleep once this answer is done
    question: str = ""                                   # what the user said (for the guards below)
    integration_details: Callable[[str], dict] = lambda _id: {}   # what the last connection check found
    search_memory: Callable[[str], list[dict]] | None = None       # the memory store, by words
    learn_place: Callable[[str], None] | None = None      # keep where the computer is, as a memory
    system_info: Callable[[], dict] | None = None         # Somni's own systems (model, memories, voice, ...)

    def notes(self) -> list[str]:
        """Extra things the model should know about what is connected (never secrets)."""
        if not self.integration_ready(API.id):
            return []
        return [API.note(self.integration_config(API.id))]
    devices: Any = None                                  # devices.hub.DeviceHub: devices Somni controls directly
    say_on: Callable[[Any, str], None] | None = None     # (device, message): speak it on that speaker

    def device_names(self) -> list[str]:
        """'Kitchen light (light.kitchen)' for every smart-home device, so the model knows what
        exists. Names only: states are read with a tool, never guessed from a stale list."""
        names = _device_names(self.integration_config(HOME.id)) if self.integration_ready(HOME.id) else []
        if self.devices is not None:
            try:
                names = self.devices.names() + names
            except Exception as exc:              # no device list is better than no answer
                log.warning("could not list devices: %s", exc)
        return names[:MAX_DEVICE_NAMES]

    def mentioned_devices(self, question: str) -> list[dict]:
        """Devices the user named, with their state read now: the model answers "what volume is the
        kitchen display at?" from this instead of guessing (measured: a 4B model said 50%; it was 40)."""
        if self.devices is None or not self.devices.links:
            return []
        words = set(re.findall(r"[a-z0-9]+", question.lower()))
        try:
            devices = self.devices.devices(max_age=3.0)
        except Exception as exc:
            log.warning("could not read devices: %s", exc)
            return []
        return [d.row() for d in devices if words & _distinct_words(d.name)]

    def home_ready(self) -> bool:
        return self.integration_ready(HOME.id) or bool(self.devices is not None and self.devices.links)


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str                                     # what the model reads
    parameters: dict                                      # JSON schema of the arguments
    run: Callable[[dict, ToolContext], dict]
    needs: str | None = None                             # integration id that must be connected
    acts: bool = False                                   # changes something in the world
    capability: str | None = None                        # ... and a service it must have (from its check)

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    def available(self, ctx: ToolContext) -> bool:
        if self.needs and not ctx.integration_ready(self.needs):
            return False
        if self.capability and self.capability not in (ctx.integration_details(self.needs).get("services") or []):
            return False
        if self.name == "memory_remember" and ctx.remember is None:
            return False
        if self.name == "memory_search" and ctx.search_memory is None:
            return False
        if self.name in ("home_states", "home_control") and not ctx.home_ready():
            return False
        if self.name in ("devices_find", "device_connect") and ctx.devices is None:
            return False
        if self.name == "speaker_say" and (ctx.say_on is None or ctx.devices is None or not any(
                link["driver"] == "cast" for link in ctx.devices.links)):
            return False
        if self.name == "play_music" and not (_has_direct_speakers(ctx) or ctx.integration_ready(HOME.id)):
            return False
        return not (self.name == "sleep_start" and ctx.request_sleep is None)


def _schema(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def validate(tool: Tool, args: Any) -> dict:
    if not isinstance(args, dict):
        raise ToolFailed("arguments must be an object")
    props = tool.parameters.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolFailed(f"unknown argument {sorted(unknown)[0]}")
    for key in tool.parameters.get("required", []):
        if args.get(key) in (None, ""):
            raise ToolFailed(f"missing argument {key}")
    out = {}
    for key, value in args.items():
        spec = props[key]
        expected = TYPES.get(spec.get("type", "string"), str)
        if spec.get("type") == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) and spec.get("type") != "boolean" or not isinstance(value, expected):
            raise ToolFailed(f"{key} must be {spec.get('type', 'string')}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolFailed(f"{key} must be one of {', '.join(map(str, spec['enum']))}")
        out[key] = value
    return out


# ---------------------------------------------------------------- built-in tools
def _now(_args: dict, _ctx: ToolContext) -> dict:
    now = dt.datetime.now().astimezone()
    return {"date": now.strftime("%A %d %B %Y"), "time": now.strftime("%H:%M"),
            "timezone": now.tzname() or ""}


# Guards in code, not just in the prompt: a small model sometimes reaches for a tool that was not
# asked for (measured: "write a poem about rain" -> it tried to save a fact about rain). Saving to
# memory shapes what the assistant learns, and sleeping takes over the GPU, so both run only when
# the user's own words ask for them.
ASKED_TO_REMEMBER = re.compile(r"\b(remember|don'?t forget|do not forget|note (that|this|down)|make a note|"
                               r"keep in mind|memori[sz]e|memory|memories|save (this|that|it)|"
                               r"write (it|this|that) down|store (this|that|it))\b", re.I)
ASKED_TO_SLEEP = re.compile(r"\b(sleep|nap|go to bed|bedtime|rest now|dream)\b", re.I)
# Device control: measured, a 4B model asked only "is the fan on?" checked it AND switched it on.
# Asking about the house must never change it, so control runs only for an action request.
ACTION_WORDS = re.compile(r"\b(turn|switch|set|dim|brighten|toggle|start|stop|put|make|open|close|raise|lower|"
                          r"increase|decrease|change|play|pause|resume|activate|run|power|shut|kill|enable|disable|"
                          r"lock|unlock|arm|disarm|launch|watch|mute|unmute|louder|quieter|volume|skip)\b",
                          re.I)
QUESTION_START = re.compile(r"^\s*(is|are|was|were|does|do|did|has|have|what|which|how|who|when|where|why|"
                            r"tell me|can you tell|could you tell|check)\b", re.I)
ON_OFF = re.compile(r"\b(on|off|up|down)\b", re.I)


ASKED_TO_CONNECT = re.compile(r"\b(connect|pair|add|set ?up|link|hook (it |them )?up|install|yes|yeah|yep|sure|"
                              r"ok|okay|please|do it|go ahead|all of them|both)\b", re.I)
ASKED_TO_SAY = re.compile(r"\b(say|tell|announce|broadcast|speak|call out|let \w+ know|shout|read (it|this) out)\b",
                          re.I)
KIND_DOMAINS = {"light": "light", "switch": "switch", "speaker": "media_player", "tv": "media_player"}
GENERIC_NAME_WORDS = {"the", "a", "my", "light", "lights", "lamp", "bulb", "plug", "switch", "display", "speaker",
                      "hub", "stick", "streaming", "tv", "mini", "google", "nest", "smart", "room"}


def _distinct_words(name: str) -> set[str]:
    """'Kitchen speaker 2' -> {'kitchen'}; 'Roku Streaming Stick' -> {'roku'}."""
    words = {w for w in re.findall(r"[a-z0-9]+", name.lower()) if not w.isdigit()}
    return (words - GENERIC_NAME_WORDS) or words


def asked_to_act(question: str) -> bool:
    """'Turn on the lights', 'lights off', 'can you dim the kitchen?' yes; 'is the fan on?' no."""
    if ACTION_WORDS.search(question):
        return True
    return bool(ON_OFF.search(question)) and not QUESTION_START.search(question)


def continued_request(question: str, history: list[dict]) -> str:
    """What the guards should read: a follow-up that is not a question carries on the last request
    (measured: "make them purple" ... "they are still blue" was refused as "only asking about it")."""
    if asked_to_act(question) or QUESTION_START.search(question) or question.rstrip().endswith("?"):
        return question
    last = next((m.get("content", "") for m in reversed(list(history)) if m.get("role") == "user"), "")
    return f"{last} / {question}" if last and asked_to_act(last) else question


def _remember(args: dict, ctx: ToolContext) -> dict:
    if not ASKED_TO_REMEMBER.search(ctx.question):
        raise ToolFailed("the user did not ask you to remember anything; just answer them")
    text = args["fact"].strip()[:1000]
    ctx.remember(text)
    return {"saved": True}


def _memory_search(args: dict, ctx: ToolContext) -> dict:
    found = ctx.search_memory(args["query"].strip()[:200])
    return {"memories": found, "found": len(found)}


def _sleep(_args: dict, ctx: ToolContext) -> dict:
    if not ASKED_TO_SLEEP.search(ctx.question):
        raise ToolFailed("the user did not ask you to sleep; just answer them")
    ctx.request_sleep()
    return {"sleeping": True}


# ---------------------------------------------------------------- smart home
HOME = HomeAssistant()
EMAIL = EmailImap()
DEVICE_LIST_TTL_S = 60
MAX_DEVICE_NAMES = 60
_device_cache: dict = {"key": None, "at": 0.0, "names": []}


def _device_names(config: dict) -> list[str]:
    import time
    key = (config.get("url"), config.get("token"))
    now = time.monotonic()
    if _device_cache["key"] == key and now - _device_cache["at"] < DEVICE_LIST_TTL_S:
        return _device_cache["names"]
    try:
        rows = HOME.states(config, limit=1000)
    except Exception as exc:                 # no device list is better than no answer
        log.warning("could not list devices: %s", exc)
        return []
    useful = [r for r in rows if r["entity_id"].split(".", 1)[0] not in ("sun", "zone", "person", "update")]
    names = [f"{r['name']} ({r['entity_id']}" + (f", {r['room']})" if r.get("room") else ")")
             for r in useful[:MAX_DEVICE_NAMES]]
    _device_cache.update(key=key, at=now, names=names)
    return names


def _home_states(args: dict, ctx: ToolContext) -> dict:
    search, domain = args.get("search", ""), args.get("domain", "")
    rows = []
    if ctx.devices is not None and ctx.devices.links:
        found = ctx.devices.find(search) if search else ctx.devices.devices()
        typed = [d for d in found if not domain or domain in (d.kind, KIND_DOMAINS.get(d.kind))]
        rows += [d.row() for d in (typed or (found if search else []))]   # the type is only a hint
    if ctx.integration_ready(HOME.id):
        rows += HOME.states(ctx.integration_config(HOME.id), search, domain)
    if not rows and search and ctx.devices is not None:
        connected = {link["id"] for link in ctx.devices.links}
        waiting = [f.name for f in ctx.devices.lookup(search) if f.driver and f.key not in connected]
        if waiting:
            return {"devices": [], "count": 0, "not_connected": waiting,
                    "note": "these are on the network but not connected yet: offer to connect them"}
    return {"devices": rows, "count": len(rows)}


def _direct_device(ctx: ToolContext, name: str):
    """A device Somni controls directly, by id or name (None: not one of those)."""
    if ctx.devices is None or not ctx.devices.links:
        return None
    found = ctx.devices.find(name)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        names = ", ".join(f"{d.name} ({d.id})" for d in found[:8])
        raise ToolFailed(f"'{name}' matches {len(found)} devices: {names}")
    if ":" in name:
        raise ToolFailed(f"there is no device {name}")
    return None


def _control_direct(device, args: dict, ctx: ToolContext) -> dict:
    from ..devices.base import Command, DeviceError
    domain = KIND_DOMAINS.get(device.kind, device.kind)
    if domain not in ctx.allowed_domains:
        raise ToolFailed(f"controlling {domain} devices from chat is switched off (Settings → tools.allowed_domains)")
    command = Command(args["action"], args.get("brightness_pct"), args.get("volume_pct"), args.get("app", ""),
                      args.get("color", ""))
    try:
        now = ctx.devices.control(device, command)
    except DeviceError as exc:
        raise ToolFailed(str(exc)) from None
    return {"done": f"{args['action']} {device.name}", "now": now.row()}


def _home_control(args: dict, ctx: ToolContext) -> dict:
    if not asked_to_act(ctx.question):
        raise ToolFailed("the user only asked about the device; do not change it, just tell them its state")
    entity = args["entity_id"].strip()
    direct = _direct_device(ctx, entity) if ":" in entity or "." not in entity else None
    if direct is not None:
        return _control_direct(direct, args, ctx)
    if not ctx.integration_ready(HOME.id):
        raise ToolFailed(f"there is no device called '{entity}'; ask the user which device they mean")
    if args["action"] not in ("turn_on", "turn_off", "toggle", "set"):
        raise ToolFailed(f"Home Assistant devices here cannot {args['action'].replace('_', ' ')}")
    config = ctx.integration_config(HOME.id)
    if "." not in entity:                       # a name was given: find the device it means
        found = HOME.states(config, entity)
        exact = [d for d in found if d["name"].lower() == entity.lower()]
        found = exact or found                  # "Office light" is not also "Office light 2"
        if len(found) != 1:
            names = ", ".join(f"{d['name']} ({d['entity_id']})" for d in found[:8])
            raise ToolFailed(f"'{entity}' matches {len(found)} devices" + (f": {names}" if names else ""))
        entity = found[0]["entity_id"]
    domain = entity.split(".", 1)[0]
    if domain not in ctx.allowed_domains:
        raise ToolFailed(f"controlling {domain} devices from chat is switched off (Settings → tools.allowed_domains)")
    data = {}
    if args.get("brightness_pct") is not None:
        data["brightness_pct"] = max(0, min(100, int(args["brightness_pct"])))
    if args.get("temperature") is not None:
        data["temperature"] = float(args["temperature"])
    scene = (args.get("scene") or "").strip()
    if args.get("color") or scene:
        if domain != "light":
            raise ToolFailed("only lights have a color or a scene")
        scenes = HOME.attributes(config, entity).get("effect_list") or []
        if args.get("color") and not scene and _scene_named(scenes, args["color"]):
            scene = args["color"]                  # "cozy" is one of the light's scenes, not a color
        elif args.get("color"):
            data.update(_ha_color(args["color"]))
        if scene:
            match = _scene_named(scenes, scene)
            if match is None:
                raise ToolFailed(f"this light has no scene '{scene}'" + (f"; its scenes are: {', '.join(scenes)}"
                                                                         if scenes else ""))
            data["effect"] = match
    action = args["action"]
    if action == "set" and domain == "climate":
        service = "set_temperature"
    elif action == "set":
        service = "turn_on"
    else:
        service = action
    state = HOME.call(config, entity, service, data)
    return {"done": f"{service} {entity}", "now": state}


def _scene_named(scenes: list[str], wanted: str) -> str | None:
    """'cozy' -> 'Cozy', 'the fireplace setting' -> 'Fireplace'."""
    words = set(re.findall(r"[a-z]+", wanted.lower())) - {"the", "a", "setting", "scene", "mode", "color", "colour"}
    for scene in scenes:
        if set(re.findall(r"[a-z]+", scene.lower())) == words or scene.lower() == wanted.strip().lower():
            return scene
    return None


def _ha_color(text: str) -> dict:
    """A color in words -> Home Assistant's light.turn_on data."""
    from ..devices import colors
    try:
        color = colors.parse(text)
    except ValueError as exc:
        raise ToolFailed(str(exc)) from None
    if color.is_white:
        return {"color_temp_kelvin": color.kelvin}
    return {"rgb_color": list(color.rgb)}


def _devices_find(_args: dict, ctx: ToolContext) -> dict:
    connected = {link["id"] for link in ctx.devices.links}
    in_home = set()
    if ctx.integration_ready(HOME.id):
        try:
            in_home = ha_features.known_macs(ctx.integration_config(HOME.id))
        except CheckFailed as exc:
            log.info("could not read Home Assistant's devices: %s", exc)
    found, groups = [], {"connected": [], "in Home Assistant": [], "new, not in Home Assistant yet": [],
                         "not connected yet": [], "cannot control": []}
    for f in ctx.devices.scan():
        mac = re.sub(r"[^0-9a-f]", "", (f.ident or "").lower())
        if f.key in connected:
            status = "already connected"
            groups["connected"].append(f.name)
        elif mac and mac in in_home:
            status = "already in Home Assistant (control and name it there; do not connect it again)"
            groups["in Home Assistant"].append(f.name)
        elif f.driver == "wiz" and ctx.integration_ready(HOME.id):
            status = "NEW: not in Home Assistant yet (home_organize adds it and names it in one step)"
            groups["new, not in Home Assistant yet"].append(f.name)
        elif f.driver:
            status = "found, NOT connected yet (device_connect can connect it)"
            groups["not connected yet"].append(f.name + (f" ({f.note})" if f.note else ""))
        else:
            status = f"cannot be controlled: {f.note}"
            groups["cannot control"].append(f.name)
        found.append({"name": f.name, "model": f.model, "status": status})
    summary = ". ".join(f"{k.capitalize()}: {', '.join(v)}" for k, v in groups.items() if v)
    out = {"summary": summary or "Nothing answered on the network.", "found": found, "count": len(found)}
    if ctx.integration_config(HOME.id).get("url") and not ctx.integration_ready(HOME.id):
        out["home_assistant"] = ("NOT ANSWERING right now, so which devices are in it is unknown and nothing can be "
                                 "named, put in a room or added to it. Tell the user Home Assistant is not reachable.")
    new = groups["new, not in Home Assistant yet"]
    if new and (ASKED_TO_ORGANIZE.search(ctx.question) or ASKED_TO_CONNECT.search(ctx.question)):
        if len(new) == 1:
            out["then"] = {"tool": "home_organize",
                           "say": f"The new bulb is {new[0]}. Use device '{new[0]}' with the name and room the "
                                  "user gave (it is added to Home Assistant automatically)."}
        else:
            out["then"] = {"tool": "home_discovered", "say": f"There are {len(new)} new bulbs ({', '.join(new)}): "
                                                             "add them all with action add, name all."}
    if found:
        out["learned"] = ("The user's home network has: "
                          + ", ".join(sorted(f"{f['name']} ({f['model']})" if f["model"] else f["name"]
                                             for f in found)) + ".")
    return out


def _device_connect(args: dict, ctx: ToolContext) -> dict:
    from ..devices.base import DeviceError
    if not ASKED_TO_CONNECT.search(ctx.question):
        raise ToolFailed("the user did not ask to connect anything; ask them first")
    try:
        return ctx.devices.connect(args["name"])
    except DeviceError as exc:
        raise ToolFailed(str(exc)) from None


def _one_speaker(ctx: ToolContext, name: str):
    found = [d for d in ctx.devices.find(name) if d.kind == "speaker"] if name else []
    if len(found) == 1:
        return found[0]
    speakers = [d.name for d in ctx.devices.devices(max_age=60) if d.kind == "speaker"]
    if not name and len(speakers) == 1:
        return next(d for d in ctx.devices.devices(max_age=60) if d.kind == "speaker")
    raise ToolFailed(f"'{name}' matches {len(found)} speakers; the speakers are: {', '.join(speakers) or 'none'}")


def _play_music(args: dict, ctx: ToolContext) -> dict:
    from ..devices.base import DeviceError
    from ..devices.radio import find_station
    from ..integrations.base import CheckFailed
    if not asked_to_act(ctx.question):
        raise ToolFailed("the user did not ask for anything to be played; just answer them")
    speaker = _music_speaker(ctx, args.get("device", "").strip())
    try:
        station = find_station(args.get("what", ""))
        if isinstance(speaker, dict):                     # a Home Assistant speaker
            ha_features.play_url(ctx.integration_config(HOME.id), speaker["entity_id"], station["url"])
            return {"playing": station["name"], "on": speaker["name"]}
        ctx.devices.play(speaker, station["url"], station["name"], station["content_type"], live=True)
    except (CheckFailed, DeviceError) as exc:
        raise ToolFailed(str(exc)) from None
    return {"playing": station["name"], "on": speaker.name}


def _has_direct_speakers(ctx: ToolContext) -> bool:
    return ctx.devices is not None and any(link["driver"] == "cast" for link in ctx.devices.links)


def _music_speaker(ctx: ToolContext, name: str):
    """A speaker Somni plays on itself (preferred), else one of Home Assistant's (a dict)."""
    if _has_direct_speakers(ctx):
        try:
            return _one_speaker(ctx, name)
        except ToolFailed:
            if not ctx.integration_ready(HOME.id):
                raise
    if not ctx.integration_ready(HOME.id):
        raise ToolFailed("no speakers are connected")
    found = ha_features.speakers(ctx.integration_config(HOME.id))
    words = _distinct_words(name) if name else set()
    match = [s for s in found if words and words <= set(re.findall(r"[a-z0-9]+", s["name"].lower()))]
    if len(match) == 1:
        return match[0]
    if not name and len(found) == 1:
        return found[0]
    raise ToolFailed(f"'{name}' matches {len(match)} speakers; the speakers are: "
                     f"{', '.join(s['name'] for s in found) or 'none'}")


def _speaker_say(args: dict, ctx: ToolContext) -> dict:
    from ..devices.base import DeviceError
    if not ASKED_TO_SAY.search(ctx.question):
        raise ToolFailed("the user did not ask for anything to be said out loud; just answer them")
    speaker = _one_speaker(ctx, args["device"].strip())
    try:
        ctx.say_on(speaker, args["message"].strip()[:500])
    except DeviceError as exc:
        raise ToolFailed(str(exc)) from None
    return {"said": args["message"].strip()[:500], "on": speaker.name}


def _google(args: dict, ctx: ToolContext) -> dict:
    if not asked_to_act(ctx.question):
        raise ToolFailed("the user did not ask for anything to be done; just answer them")
    return HOME.google_command(ctx.integration_config(HOME.id), args["command"].strip()[:300])


def _home_assist(args: dict, ctx: ToolContext) -> dict:
    request = args["request"].strip()[:300]
    if asked_to_act(request) and not asked_to_act(ctx.question):
        raise ToolFailed("the user only asked a question; ask Home Assistant a question, do not change anything")
    try:
        out = ha_features.assist(ctx.integration_config(HOME.id), request)
    except CheckFailed as exc:
        raise ToolFailed(str(exc)) from None
    if out.get("result") == "failed":
        raise ToolFailed(out["note"])
    return out


ASKED_ROUTINE = re.compile(r"\b(every|each|daily|nightly|weekdays?|weekends?|routines?|automations?|automate|"
                           r"schedule|whenever|when|sunrise|sunset|at \d|mornings?|nights?|evenings?)\b", re.I)
ASKED_TO_REMOVE = re.compile(r"\b(delete|remove|cancel|stop|get rid|drop|undo|turn off|disable)\b", re.I)


def _home_routine(args: dict, ctx: ToolContext) -> dict:
    config = ctx.integration_config(HOME.id)
    action = args.get("action", "list")
    try:
        if action == "list":
            found = ha_features.routines(config)
            return {"routines": found, "count": len(found)}
        if action == "delete":
            if not ASKED_TO_REMOVE.search(ctx.question):
                raise ToolFailed("the user did not ask to remove a routine; ask them first")
            return ha_features.delete_routine(config, args.get("name", ""))
        if not ASKED_ROUTINE.search(ctx.question):
            raise ToolFailed("the user did not ask for a routine; just answer them")
        body = ha_features.routine_config(args.get("name", ""), args.get("do", ""), args.get("at", ""),
                                          args.get("days", ""), int(args.get("offset_minutes") or 0),
                                          args.get("device", ""), args.get("becomes", ""))
        made = ha_features.save_routine(config, body)
        trigger = body["triggers"][0]
        made["when"] = trigger.get("at") or trigger.get("event") or \
            f"{trigger.get('entity_id')} becomes {trigger.get('to', 'anything')}"
        made["days"] = body["conditions"][0]["weekday"] if body["conditions"] else "every day"
        made["does"] = args.get("do", "")
        return made
    except CheckFailed as exc:
        raise ToolFailed(str(exc)) from None


def _home_discovered(args: dict, ctx: ToolContext) -> dict:
    config = ctx.integration_config(HOME.id)
    try:
        missed = ha_features.missed_wiz(config)
        if args.get("action", "list") == "add":
            if not ASKED_TO_CONNECT.search(ctx.question):
                raise ToolFailed("the user did not ask to add anything; ask them first")
            name = args.get("name", "all")
            words = name.lower().strip()
            everything = words in ("", "all", "everything", "all of them")
            out: dict = {"added": []}
            for bulb in missed:
                if everything or words in bulb["name"].lower() or "wiz" in words or "bulb" in words:
                    out["added"].append(ha_features.add_by_address(config, "wiz", bulb["host"])["added"])
            try:
                more = ha_features.add_discovered(config, name)
                out["added"] += more["added"]
                out.update({k: v for k, v in more.items() if k != "added"})
            except CheckFailed:
                if not out["added"]:
                    raise
            if out["added"]:
                out["next"] = "ask the user what to call each new device and which room it is in (home_organize)"
            return out
        waiting = [w["name"] for w in ha_features.discovered(config)] + [b["name"] for b in missed]
        return {"discovered": waiting, "count": len(waiting)}
    except CheckFailed as exc:
        raise ToolFailed(str(exc)) from None


ASKED_TO_ORGANIZE = re.compile(r"\b(call|name|rename|label|room|is in|are in|put|move|belongs?|assign|office|"
                               r"bedroom|kitchen|living|bathroom|garage|hall|dining|basement|nursery)\b", re.I)


def _home_organize(args: dict, ctx: ToolContext) -> dict:
    if not ASKED_TO_ORGANIZE.search(ctx.question):
        raise ToolFailed("the user did not ask to rename or move anything; just answer them")
    try:
        return ha_features.organize(ctx.integration_config(HOME.id), args["device"], args.get("name", ""),
                                    args.get("room", ""))
    except CheckFailed as exc:
        raise ToolFailed(str(exc)) from None


# ---------------------------------------------------------------- the web
WEB = WebLookup()


def _web_search(args: dict, ctx: ToolContext) -> dict:
    return WEB.search(ctx.integration_config(WEB.id), args["query"], 5)


HERE = ("", "here", "home", "my location", "current location", "local", "my city", "my town", "my area")


def _weather(args: dict, ctx: ToolContext) -> dict:
    place = (args.get("place") or "").strip()
    if place.lower() in HERE:
        raise ToolFailed("weather needs a town. Work out where the user is (their memories, then computer_info) "
                         "and call weather again with it")
    return WEB.weather(place)


def _system_status(_args: dict, ctx: ToolContext) -> dict:
    info = dict(ctx.system_info()) if ctx.system_info else {}
    info["abilities"] = abilities(ctx)
    return info


def _computer_info(_args: dict, ctx: ToolContext) -> dict:
    from ..location import computer_place
    now = dt.datetime.now().astimezone()
    info = {"date": now.strftime("%A %d %B %Y"), "time": now.strftime("%H:%M"), "time_zone": now.tzname() or ""}
    place = computer_place()
    info["location"] = place or "unknown (this computer's location service is off)"
    if place and ctx.learn_place:
        ctx.learn_place(place)
    return info


def _web_read(args: dict, ctx: ToolContext) -> dict:
    return WEB.read(args["url"].strip())


# ---------------------------------------------------------------- the user's own API
API = CustomApi()


def _api_call(args: dict, ctx: ToolContext) -> dict:
    from ..integrations.base import CheckFailed
    try:
        return API.call(ctx.integration_config(API.id), args.get("method", "GET"), args.get("path", ""),
                        args.get("body", ""))
    except CheckFailed as exc:
        raise ToolFailed(str(exc)) from None


# ---------------------------------------------------------------- email
def _email_unread(args: dict, ctx: ToolContext) -> dict:
    return EMAIL.unread(ctx.integration_config(EMAIL.id), int(args.get("limit", 5)))


def _email_search(args: dict, ctx: ToolContext) -> dict:
    return EMAIL.search(ctx.integration_config(EMAIL.id), args["query"], int(args.get("limit", 5)))


TOOLS: tuple[Tool, ...] = (
    Tool("clock_now", "Date and time", "Get the current date and time.", _schema({}), _now),
    Tool("memory_remember", "Remember something",
         "Save a fact to memory when the user asks you to remember something.",
         _schema({"fact": {"type": "string", "description": (
             "the fact as one sentence in the third person, saying who it is about, e.g. "
             "'The user's sister is called Ana.' or 'The assistant's name is Somni.'")}}, ("fact",)), _remember,
         acts=True),
    Tool("memory_search", "Search memory",
         "Search your memory for something the user told you before, when it is not in the memories above. "
         "Try other words than the question (for 'what's my name?' search 'name call user').",
         _schema({"query": {"type": "string", "description": "words to look for"}}, ("query",)), _memory_search),
    Tool("sleep_start", "Go to sleep",
         "Go to sleep now to learn from today's conversations. Only when the user asks you to sleep.",
         _schema({}), _sleep, acts=True),
    Tool("home_states", "See devices", "List smart-home devices and their states (lights, switches, speakers, "
         "displays, TVs, sensors, climate). Use search for a room or device name.",
         _schema({"search": {"type": "string", "description": "words in the device name, e.g. 'kitchen'"},
                  "domain": {"type": "string", "description": "device type, e.g. light, switch, climate, sensor"}}),
         _home_states),
    Tool("home_control", "Control a device",
         "Control ONE device (for a whole room or several devices use home_assist, or call this once per "
         "device): turn it on or off, set brightness, color, scene, temperature or volume; play, pause or stop "
         "a speaker, display or TV; launch an app on a TV.",
         _schema({"entity_id": {"type": "string", "description": "the device's id from the list, or its name"},
                  "action": {"type": "string", "enum": ["turn_on", "turn_off", "toggle", "set", "play", "pause",
                                                        "stop", "volume_up", "volume_down", "launch"]},
                  "brightness_pct": {"type": "integer", "description": "0-100, lights only"},
                  "color": {"type": "string", "description": "lights only: a color name (red, teal, purple), "
                                                             "a white (warm white, cool white, daylight, "
                                                             "2700K) or a hex code"},
                  "scene": {"type": "string", "description": "lights only: one of the light's own scenes, "
                                                             "e.g. Cozy, Fireplace, Ocean, Relax"},
                  "temperature": {"type": "number", "description": "target temperature, climate only"},
                  "volume_pct": {"type": "integer", "description": "0-100, with action set, speakers only"},
                  "app": {"type": "string", "description": "app to open with action launch, e.g. Netflix"}},
                 ("entity_id", "action")), _home_control, acts=True),
    Tool("devices_find", "Find devices",
         "Look for smart devices on the home network (lights, plugs, speakers, displays, TVs) when the user "
         "wants to connect or add one, or asks what devices there are.", _schema({}), _devices_find),
    Tool("device_connect", "Connect a device",
         "Connect a device that devices_find found, by its name, or 'all' for everything found.",
         _schema({"name": {"type": "string", "description": "the device's name, or all"}}, ("name",)),
         _device_connect, acts=True),
    Tool("play_music", "Play music",
         "Play music on a speaker or display: internet radio for a genre, mood or station name (jazz, "
         "classical, 80s, BBC Radio 2). To pause or stop it afterwards use home_control.",
         _schema({"device": {"type": "string", "description": "the speaker's name (empty if there is only one)"},
                  "what": {"type": "string", "description": "a genre, mood or station; empty for popular hits"}}),
         _play_music, acts=True),
    Tool("speaker_say", "Say it on a speaker",
         "Say a message out loud on a speaker or display in the house, e.g. 'dinner is ready' on the kitchen "
         "display.",
         _schema({"device": {"type": "string", "description": "the speaker's name"},
                  "message": {"type": "string", "description": "what to say"}}, ("device", "message")),
         _speaker_say, acts=True),
    Tool("google_command", "Tell Google Assistant",
         "Send a command to Google Assistant, for devices set up in the Google Home app "
         "(use home_control first for devices listed above).",
         _schema({"command": {"type": "string", "description": "what to tell Google, e.g. 'turn on the bedroom fan'"}},
                 ("command",)), _google, needs=HOME.id, acts=True, capability=SDK_DOMAIN),
    Tool("home_assist", "Ask Home Assistant",
         "Give Home Assistant's own assistant a smart-home request in plain words. Best for several devices or a "
         "whole room at once ('turn off all the lights', 'turn on the kitchen lights'), timers, scenes, and "
         "anything home_control can't do. It answers what it did.",
         _schema({"request": {"type": "string", "description": "the request in plain English"}}, ("request",)),
         _home_assist, needs=HOME.id, acts=True),
    Tool("home_routine", "Routines",
         "Create, list or delete a routine (a Home Assistant automation) that runs by itself: at a time, at "
         "sunrise or sunset, or when a device changes. What it does is a request in plain words.",
         _schema({"action": {"type": "string", "enum": ["create", "list", "delete"]},
                  "name": {"type": "string", "description": "a short name; for delete, the routine's name"},
                  "do": {"type": "string", "description": "what to do, e.g. 'turn off all the lights'"},
                  "at": {"type": "string", "description": "a time like 22:00 or 7am, or sunrise / sunset"},
                  "days": {"type": "string", "description": "e.g. 'weekdays', 'sat sun'; empty for every day"},
                  "offset_minutes": {"type": "integer",
                                     "description": "minutes after sunrise/sunset (negative: before)"},
                  "device": {"type": "string", "description": "instead of a time: a device's entity_id"},
                  "becomes": {"type": "string", "description": "with device: the state, e.g. on, off, home"}},
                 ("action",)), _home_routine, needs=HOME.id, acts=True),
    Tool("home_discovered", "New devices",
         "Devices Home Assistant found on the network by itself and is waiting to add: list them, or add one "
         "by name (or 'all') when the user wants them added.",
         _schema({"action": {"type": "string", "enum": ["list", "add"]},
                  "name": {"type": "string", "description": "for add: the device's name, or all"}}),
         _home_discovered, needs=HOME.id, acts=True),
    Tool("home_organize", "Name and room",
         "Give a Home Assistant device a name and/or put it in a room (made if new), e.g. when the user says "
         "'the new bulb is the office light'. A new WiZ bulb not in Home Assistant yet is added first, "
         "automatically (device: 'the new bulb' or its name from devices_find). Then 'turn off the office' works.",
         _schema({"device": {"type": "string", "description": "the device's entity_id or current name"},
                  "name": {"type": "string", "description": "what to call it, e.g. 'Office light'"},
                  "room": {"type": "string", "description": "the room, e.g. 'Office'"}}, ("device",)),
         _home_organize, needs=HOME.id, acts=True),
    Tool("web_search", "Search the web",
         "Search the web for facts, news, prices, sports, opening hours or anything you are not sure about "
         "(not the weather: use weather). "
         "Returns results and the relevant text of the top pages.",
         _schema({"query": {"type": "string", "description": "what to search for"}}, ("query",)), _web_search,
         needs=WEB.id),
    Tool("weather", "Weather",
         "Current weather and the forecast for the next seven days in a town. Answer in one or two sentences "
         "with just what was asked.",
         _schema({"place": {"type": "string", "description": "a town or city, e.g. 'Chicago' or 'Paris, France'"}}),
         _weather,
         needs=WEB.id),
    Tool("system_status", "Check my systems",
         "Check Somni's own systems: which model it runs, how many memories and learned experts it has, what is "
         "connected, voice and listening, sleep, and the graphics card. Use it for questions about yourself.",
         _schema({}), _system_status),
    Tool("computer_info", "This computer",
         "What this computer knows: where it is (town, from its location service), its time zone, the date and "
         "time. Use it when you need something like where the user is and your memories do not say.",
         _schema({}), _computer_info),
    Tool("web_read", "Read a web page", "Read the text of a web page (for example a search result) to answer from it.",
         _schema({"url": {"type": "string", "description": "the page address, starting with https://"}}, ("url",)),
         _web_read, needs=WEB.id),
    Tool("api_call", "Use your API",
         "Call the API the user connected (see what it is for above). Somni adds the key; never ask for it.",
         _schema({"method": {"type": "string", "enum": ["GET", "POST"]},
                  "path": {"type": "string", "description": "the path after the API address, with any ?query"},
                  "body": {"type": "string", "description": "for POST: the JSON body as text"}}, ("method",)),
         _api_call, needs=API.id, acts=True),
    Tool("email_unread", "Unread email", "List the newest unread emails (sender, subject, date).",
         _schema({"limit": {"type": "integer", "description": "how many, default 5"}}), _email_unread,
         needs=EMAIL.id),
    Tool("email_search", "Search email", "Find emails by sender or subject.",
         _schema({"query": {"type": "string"}, "limit": {"type": "integer"}}, ("query",)), _email_search,
         needs=EMAIL.id),
)
BY_NAME = {t.name: t for t in TOOLS}


@dataclass
class CallRecord:
    name: str
    arguments: dict
    ok: bool
    result: dict = field(default_factory=dict)
    error: str = ""

    def for_model(self) -> str:
        import json
        body = {"result": self.result} if self.ok else {"error": self.error}
        text = json.dumps(body, ensure_ascii=False, default=str)
        return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "…(truncated)"

    def summary(self) -> str:
        if not self.ok:
            return self.error
        r = self.result
        if "done" in r:
            now = r.get("now") or {}
            return f"{now.get('name', '')} is {now.get('state', 'done')}".strip()
        if "devices" in r:
            return f"{r['count']} devices"
        if "playing" in r:
            return f"playing {r['playing']} on {r['on']}"
        if "said" in r:
            return f"said on {r['on']}"
        if "found" in r and "count" in r:
            return f"{r['count']} things on the network"
        if "connected" in r or "waiting" in r or "already" in r or "unsupported" in r:
            parts = [f"{k}: {', '.join(v)}" for k, v in r.items() if v]
            return "; ".join(parts)[:160]
        if "home_assistant_said" in r:
            return f"Home Assistant: {r['home_assistant_said'] or r['result']}"[:160]
        if "created" in r:
            return f"routine '{r['created']}' made"
        if "deleted" in r:
            return f"routine '{r['deleted']}' removed"
        if "routines" in r:
            return f"{r['count']} routines"
        if "organized" in r:
            return " in ".join(filter(None, [r.get("name") or r["organized"], r.get("room")]))
        if "discovered" in r:
            return f"{r['count']} new devices waiting"
        if "added" in r:
            return f"added {', '.join(r['added']) or 'nothing'}"
        if "sent" in r:
            return f"told Google: {r['sent']}"
        if "data" in r or ("text" in r and "url" not in r):
            return "the API answered"
        if "abilities" in r:
            return "checked my systems"
        if "time_zone" in r:
            return f"{r['location']}, {r['time']}"
        if "forecast" in r:
            return f"{r['place']}: {r['now']['temperature']}, {r['now']['sky']}"
        if "results" in r:
            return f"{len(r['results'])} results ({r.get('provider', 'web')})"
        if "text" in r and "url" in r:
            return f"read {r.get('title') or r['url']}"
        if "memories" in r:
            return f"{r['found']} memories found"
        if "messages" in r:
            n = r.get("unread", r.get("found", len(r["messages"])))
            return f"{n} emails"
        if "time" in r:
            return f"{r['date']}, {r['time']}"
        if r.get("saved"):
            return "saved to memory"
        if r.get("sleeping"):
            return "falling asleep after this answer"
        return "done"


def run_tool(name: str, arguments: Any, ctx: ToolContext) -> CallRecord:
    tool = BY_NAME.get(name)
    if tool is None or not tool.available(ctx):
        return CallRecord(name, arguments if isinstance(arguments, dict) else {}, False,
                          error=f"there is no tool called {name}")
    try:
        args = validate(tool, arguments)
        return CallRecord(name, args, True, tool.run(args, ctx))
    except ToolFailed as exc:
        return CallRecord(name, arguments if isinstance(arguments, dict) else {}, False, error=str(exc))
    except Exception as exc:                 # a network error must become an answer, not a crash
        log.exception("tool %s failed", name)
        return CallRecord(name, arguments if isinstance(arguments, dict) else {}, False,
                          error=f"{type(exc).__name__}: {exc}")


def available_tools(ctx: ToolContext) -> list[Tool]:
    return [t for t in TOOLS if t.available(ctx)]


# Which tools a request could need. Every description the model is given is read on every step,
# so offering all of them (2,700 tokens) to "name this bulb" made a 4B model slow, confused and,
# on a 16 GB card, out of memory. A small machine needs a short prompt; this is how it gets one.
TOOL_GROUPS: dict[str, tuple[tuple[str, ...], re.Pattern]] = {
    "home control": (("home_states", "home_control", "home_assist", "google_command"),
                     re.compile(r"\b(lights?|lamps?|bulbs?|plugs?|switch\w*|turn|dim\w*|bright\w*|colou?rs?|red|"
                                r"blue|green|purple|pink|orange|yellow|white|warm|thermostat|heat\w*|cool\w*|fan|"
                                r"locks?|doors?|camera|tv|roku|xbox|google|on|off|office|bedroom|kitchen|living|"
                                r"bathroom|garage|hall|downstairs|upstairs|everything|all)\b", re.I)),
    "home setup": (("devices_find", "device_connect", "home_discovered", "home_organize"),
                   re.compile(r"\b(new|add\w*|connect\w*|pair\w*|scan\w*|find|found|discover\w*|set ?up|install\w*|"
                              r"names?|named|rename|call (it|them)|rooms?|put|move|assign|network|devices?|wiz|hue|"
                              r"home assistant)\b", re.I)),
    "home media": (("play_music", "speaker_say", "home_control"),
                   re.compile(r"\b(music|play\w*|songs?|radio|jazz|rock|pop|station|speakers?|displays?|volume|"
                              r"louder|quieter|pause|stop|resume|say|announce|tell)\b", re.I)),
    "home routines": (("home_routine", "home_states"),
                      re.compile(r"\b(routines?|automations?|automate|every|each|daily|nightly|schedule\w*|"
                                 r"sunrise|sunset|whenever|at \d|mornings?|evenings?|nights?)\b", re.I)),
    "web": (("web_search", "web_read", "weather"),
            re.compile(r"\b(weather|rain\w*|snow\w*|forecast|temperature|hot|cold|search|look (it )?up|news|"
                       r"price|cost|how much|who|what|when|where|why|which|score|latest|today|tonight|tomorrow|"
                       r"open|hours|define|wiki\w*|internet|web|online)\b", re.I)),
    "email": (("email_unread", "email_search"), re.compile(r"\b(e-?mails?|inbox|mail|messages?)\b", re.I)),
    "self": (("system_status", "computer_info", "clock_now"),
             re.compile(r"\b(time|date|day|where am i|location|town|city|you|your|yourself|model|memor\w+|gpu|"
                        r"graphics|system|status|connected|abilities|can you)\b", re.I)),
}
ALWAYS = ("memory_remember", "memory_search", "sleep_start", "api_call")


def relevant_tools(tools: list[Tool], question: str, history: list[dict] = ()) -> list[Tool]:
    """The tools this request could need: by what was said now and just before (a follow-up like
    "yes, do it" keeps the last request's tools). Nothing recognised: every tool, as before."""
    last_user = next((m.get("content", "") for m in reversed(list(history)) if m.get("role") == "user"), "")
    text = f"{question} {last_user}"
    wanted = set(ALWAYS)
    for names, words in TOOL_GROUPS.values():
        if words.search(text):
            wanted.update(names)
    if wanted == set(ALWAYS):
        return tools
    return [t for t in tools if t.name in wanted]


def abilities(ctx: ToolContext) -> str:
    """What the assistant can actually do right now, in words (from the live tool and connection state),
    so "what are you connected to?" is answered from facts, not from what it could do in principle."""
    names = {t.name for t in available_tools(ctx)}
    can, missing = ["tell the date and time"], []
    if "computer_info" in names:
        can.append("find out where this computer is")
    if "weather" in names:
        can.append("check the weather anywhere")
    if "web_search" in names:
        provider = ctx.integration_config(WEB.id).get("provider") or "wikipedia"
        can.append("search Wikipedia" if provider == "wikipedia" else f"search the whole web (with {provider})")
        if provider == "wikipedia":
            missing.append("whole-web search (only Wikipedia is set up)")
    else:
        missing.append("web search")
    if "memory_remember" in names:
        can.append("remember what the user tells it")
    if "sleep_start" in names:
        can.append("go to sleep to learn when asked")
    links = ctx.devices.links if ctx.devices is not None else []
    if links:
        can.append("control these devices: " + ", ".join(link["name"] for link in links))
    if "devices_find" in names:
        can.append("look for smart devices on the network and connect them")
    if not links and not ctx.integration_ready(HOME.id):
        missing.append("smart devices (none connected yet)")
    if ctx.integration_ready(HOME.id):
        can.append("control devices through Home Assistant, by name or by room, and ask its own assistant")
        can.append("make routines that run by themselves (times, sunrise/sunset, a device changing)")
        can.append("add devices Home Assistant has discovered, and play music on its speakers")
    if "email_unread" in names:
        can.append("read the user's email")
    else:
        missing.append("email")
    if "api_call" in names:
        can.append(f"use the user's {ctx.integration_config(API.id).get('name') or 'custom'} API")
    missing.append("calendar")
    return f"Right now Somni can: {'; '.join(can)}. Not connected: {', '.join(missing)}."


def ready_checker(connections) -> Callable[[str], bool]:
    """integration id -> is it connected right now (from the last connection check)."""
    def ready(integration_id: str) -> bool:
        for integ in connections.integrations():
            if integ.id == integration_id:
                return connections.status_of(integ).state is State.CONNECTED
        return False
    return ready
