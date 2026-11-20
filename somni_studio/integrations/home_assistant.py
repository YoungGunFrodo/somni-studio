"""Home Assistant: the smart-home hub. One integration covers every device Home Assistant knows
(lights, switches, thermostats, sensors, media players...) instead of one integration per brand."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from .base import CheckFailed, Field, FieldKind, Integration, State, Status, ToolSpec

TIMEOUT_S = 6
CANDIDATES = ("http://homeassistant.local:8123", "http://homeassistant:8123", "http://localhost:8123")
MDNS_SERVICE = "_home-assistant._tcp.local"


def _announced(timeout: float = 2.0) -> list[str]:
    """Home Assistant instances announcing themselves on the LAN (mDNS), as http://<ip>:<port>."""
    from ..devices import net
    try:
        replies = net.mdns(timeout, (MDNS_SERVICE,))
    except OSError:
        return []
    return [f"http://{ip}:{info['ports'].get(MDNS_SERVICE, 8123)}" for ip, info in replies.items()
            if MDNS_SERVICE in info["services"] or MDNS_SERVICE in info["ports"]]


def _request(url: str, token: str, path: str, body: dict | None = None):
    if not url.startswith(("http://", "https://")):
        raise CheckFailed("the server address must start with http:// or https://")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(  # noqa: S310 - scheme checked above
        url.rstrip("/") + path, data=data, method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 - scheme checked above
        return json.loads(r.read() or b"null")


def _get(url: str, token: str, path: str):
    return _request(url, token, path)


_access: dict = {}                    # (url, refresh token) -> (access token, expires at)


def token_for(config: dict) -> str:
    """The token to call Home Assistant with: a pasted long-lived token, or a short-lived one from the
    "Sign in with Home Assistant" refresh token (renewed before it expires)."""
    if config.get("token"):
        return config["token"]
    refresh, url = config.get("refresh_token") or "", (config.get("url") or "").rstrip("/")
    if not refresh:
        raise CheckFailed("not signed in: press 'Sign in with Home Assistant'")
    cached = _access.get((url, refresh))
    if cached and cached[1] > time.time() + 60:
        return cached[0]
    reply = auth_token(url, {"grant_type": "refresh_token", "refresh_token": refresh,
                             "client_id": config.get("client_id") or ""})
    _access[(url, refresh)] = (reply["access_token"], time.time() + int(reply.get("expires_in", 1800)))
    return reply["access_token"]


def auth_token(url: str, form: dict) -> dict:
    """Home Assistant's /auth/token (sign-in codes and refresh tokens, as a form post)."""
    if not url.startswith(("http://", "https://")):
        raise CheckFailed("the server address must start with http:// or https://")
    req = urllib.request.Request(url.rstrip("/") + "/auth/token",  # noqa: S310 - scheme checked above
                                 data=urllib.parse.urlencode(form).encode(), method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 - scheme checked above
            return json.loads(r.read())
    except urllib.error.HTTPError:
        raise CheckFailed("Home Assistant refused the sign-in; sign in again") from None
    except (urllib.error.URLError, OSError) as exc:
        raise CheckFailed(f"cannot reach Home Assistant at {url}: {getattr(exc, 'reason', exc)}") from None


def _friendly(state: dict) -> dict:
    attrs = state.get("attributes") or {}
    out = {"entity_id": state.get("entity_id"), "name": attrs.get("friendly_name") or state.get("entity_id"),
           "state": state.get("state")}
    for key in ("brightness", "temperature", "current_temperature", "unit_of_measurement", "media_title"):
        if key in attrs and attrs[key] is not None:
            out[key] = attrs[key]
    if state.get("state") == "on" and str(state.get("entity_id", "")).startswith("light."):
        from ..devices import colors
        if attrs.get("effect") and str(attrs["effect"]).lower() not in ("none", "off"):
            out["scene"] = attrs["effect"]
        elif attrs.get("color_mode") == "color_temp" and attrs.get("color_temp_kelvin"):
            out["color"] = colors.describe_kelvin(int(attrs["color_temp_kelvin"]))
        elif attrs.get("rgb_color"):
            out["color"] = colors.describe_xy(*colors.rgb_to_xy(*attrs["rgb_color"][:3]))
        if attrs.get("effect_list"):
            out["scenes"] = len(attrs["effect_list"])
    return out


class HomeAssistant(Integration):
    id = "home_assistant"
    title = "Home Assistant"
    category = "Smart home"
    description = "Lights, switches, climate, sensors and scenes through your Home Assistant server."
    fields = (
        Field("url", "Server address", FieldKind.URL, "http://homeassistant.local:8123"),
        Field("token", "Long-lived access token", FieldKind.SECRET, "",
              "Not needed if you use 'Sign in with Home Assistant'. Otherwise: your profile → Security → "
              "Long-lived access tokens → Create token.", required=False),
        Field("vm", "VirtualBox VM", FieldKind.TEXT, "Home Assistant",
              "If Home Assistant runs in a VirtualBox VM on this PC, Somni starts it with itself. Empty: any VM "
              "whose name contains 'Home Assistant'.", required=False),
        Field("refresh_token", "Sign-in", FieldKind.SECRET, required=False, hidden=True),
        Field("client_id", "Sign-in client", FieldKind.TEXT, required=False, hidden=True),
    )
    tools = (
        ToolSpec("home.states", "See device states", "Read what every light, switch and sensor is doing."),
        ToolSpec("home.control", "Control devices", "Turn things on or off, set brightness, temperature or scenes."),
    )
    examples = ("Turn off the kitchen light.", "Is the garage light on?", "Dim the living room to 30 percent.",
                "Set the thermostat to 21.")
    setup = (
        "Optional. Most Wi-Fi lights, plugs, speakers and TVs connect directly (Devices, above). Home "
        "Assistant is for the rest: Zigbee and Z-Wave sensors, thermostats, locks, cloud-only brands.",
        "No Home Assistant yet? Install it on a spare PC, a Raspberry Pi or Home Assistant Green "
        "(home-assistant.io/installation), then add your devices there.",
        "Press 'Find on my network' (or type its address), then 'Sign in with Home Assistant' and log in. "
        "That's all: Somni keeps itself signed in.",
    )

    def check(self, config: dict) -> Status:
        url = (config.get("url") or "").strip()
        token = token_for(config)
        if not url.startswith(("http://", "https://")):
            raise CheckFailed("the server address must start with http:// or https://")
        try:
            info = _get(url, token, "/api/")
            states = _get(url, token, "/api/states")
        except urllib.error.HTTPError as exc:
            raise CheckFailed("the token was refused (401)" if exc.code == 401 else f"HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CheckFailed(f"cannot reach {url}: {getattr(exc, 'reason', exc)}") from None
        if not isinstance(states, list):
            raise CheckFailed("unexpected reply from /api/states")
        domains = Counter(str(s.get("entity_id", "")).split(".", 1)[0] for s in states)
        top = ", ".join(f"{n} {d}" for d, n in domains.most_common(4))
        try:
            services = sorted({s.get("domain", "") for s in _get(url, token, "/api/services") or []})
        except (urllib.error.URLError, OSError, ValueError):
            services = []
        return Status(State.CONNECTED, f"{len(states)} entities ({top})" if states else "connected, no entities",
                      {"entities": len(states), "domains": dict(domains.most_common(12)), "services": services,
                       "api": info.get("message", "") if isinstance(info, dict) else ""})

    def google_command(self, config: dict, command: str) -> dict:
        """Send a command to Google Assistant (Home Assistant's Google Assistant SDK integration), so
        devices only Google Home knows can be controlled too."""
        _request(config["url"], token_for(config), "/api/services/google_assistant_sdk/send_text_command",
                 {"command": command})
        return {"sent": command}

    @staticmethod
    def discover(hint: str = "") -> list[str]:
        """Home Assistant addresses that answer on this network (its API replies 401 without a token).

        Tries the address already typed in, Home Assistant's own mDNS announcement (by IP address, so
        it works where the PC can't resolve "homeassistant.local", as on many Windows PCs), then the
        usual names."""
        bases = [hint.rstrip("/")] if hint.startswith(("http://", "https://")) else []
        bases += _announced()
        bases += CANDIDATES
        found = []
        for base in dict.fromkeys(bases):
            try:
                urllib.request.urlopen(base + "/api/", timeout=2)  # noqa: S310 - typed, announced or fixed local
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    found.append(base)
            except (urllib.error.URLError, OSError, ValueError):
                continue
        return found

    # ------------------------------------------------------------ actions (used by tools)
    def states(self, config: dict, search: str = "", domain: str = "", limit: int = 40) -> list[dict]:
        """Devices and their states, filtered by a name/id search and/or a domain (light, switch...)."""
        rows = _request(config["url"], token_for(config), "/api/states") or []
        words = [w for w in search.lower().split() if w not in ("the", "a", "my", "all", "lights", "light")]
        in_room = self.rooms(config)
        out = []
        for s in rows:
            entity = str(s.get("entity_id", ""))
            if domain and not entity.startswith(domain.lower() + "."):
                continue
            room = in_room.get(entity, "")
            name = f"{entity} {(s.get('attributes') or {}).get('friendly_name', '')} {room}".lower()
            if words and not all(w in name for w in words):
                continue
            row = _friendly(s)
            if room:
                row["room"] = room
            out.append(row)
        return out[:limit]

    @staticmethod
    def rooms(config: dict) -> dict[str, str]:
        """entity_id -> room; empty if Home Assistant has no rooms set up (or can't say)."""
        from .ha_features import rooms
        try:
            return rooms(config)
        except (CheckFailed, OSError, ValueError):
            return {}

    def attributes(self, config: dict, entity_id: str) -> dict:
        state = _request(config["url"], token_for(config), f"/api/states/{entity_id}")
        return (state or {}).get("attributes") or {} if isinstance(state, dict) else {}

    def call(self, config: dict, entity_id: str, action: str, data: dict | None = None) -> dict:
        """Call a service on one entity (turn_on, turn_off, toggle...) and return its new state."""
        domain = entity_id.split(".", 1)[0]
        body = {"entity_id": entity_id, **(data or {})}
        token = token_for(config)
        _request(config["url"], token, f"/api/services/{domain}/{action}", body)
        state = _request(config["url"], token, f"/api/states/{entity_id}")
        return _friendly(state) if isinstance(state, dict) else {"entity_id": entity_id}
