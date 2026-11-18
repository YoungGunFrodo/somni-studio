"""Looking things up: web search and reading a page.

Search providers (only services that allow programs to search them):
  wikipedia  free, no key: encyclopedic facts (the default)
  tavily     Tavily: a search API made for AI assistants; returns the relevant page text (free plan with a key)
  brave      Brave Search API: the whole web; free plan with a key (api.search.brave.com)
  searxng    your own SearXNG instance: the whole web, nothing leaves your network but the search

Weather comes from Open-Meteo (open-meteo.com): free, no key, no account. Only the place name (or
its coordinates) is sent.

Reading a page fetches only public internet addresses: a page (or a redirect) pointing at your own
network (router, NAS, Home Assistant) is refused, so nothing the assistant reads can make it probe
your LAN.
"""
from __future__ import annotations

import datetime as dt
import html
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .base import CheckFailed, Field, FieldKind, Integration, State, Status, ToolSpec

TIMEOUT_S = 10
MAX_PAGE_BYTES = 1_500_000
MAX_PAGE_CHARS = 6000
USER_AGENT = "SomniStudio/0.3 (local personal assistant)"
PROVIDERS = ("wikipedia", "tavily", "brave", "searxng")
READ_TOP = 2                     # results whose pages are read along with the search
EXTRACT_CHARS = 1500             # of each page: the passage around the searched words


IMPERIAL_COUNTRIES = {"US", "LR", "MM"}
WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with hail"}


class Blocked(CheckFailed):
    """Not a public address."""


def _public(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise CheckFailed(f"cannot find {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise Blocked(f"{host} is not a public internet address; I only read public pages")


class _PublicOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlparse(newurl)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise Blocked("redirected somewhere that is not a web page")
        _public(parts.hostname)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, headers: dict | None = None, public_only: bool = True, method: str = "GET",
           body: bytes | None = None) -> tuple[bytes, str]:
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CheckFailed("only http and https addresses can be read")
    if public_only:
        _public(parts.hostname)
    opener = urllib.request.build_opener(_PublicOnlyRedirects if public_only else urllib.request.HTTPRedirectHandler)
    req = urllib.request.Request(url, data=body, method=method,  # noqa: S310 - http(s) only, checked above
                                 headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with opener.open(req, timeout=TIMEOUT_S) as r:
            body = r.read(MAX_PAGE_BYTES + 1)
            ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = {401: " (the key was refused)", 403: " (not allowed with this key)", 429: " (too many requests "
                  "for this plan; try later)"}.get(exc.code, "")
        raise CheckFailed(f"HTTP {exc.code} from {parts.hostname}{detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CheckFailed(f"cannot reach {parts.hostname}: {getattr(exc, 'reason', exc)}") from None
    return body[:MAX_PAGE_BYTES], ctype


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        self._in_title = tag == "title"

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in ("p", "div", "li", "h1", "h2", "h3", "br", "tr"):
            self.parts.append("\n")
        self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def around(text: str, query: str, size: int) -> str:
    """The `size` characters of `text` where the query's words are densest (the whole text if short)."""
    if len(text) <= size:
        return text
    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    low = text.lower()
    best, best_hits = 0, -1
    for start in range(0, len(text) - size + 1, max(1, size // 4)):
        window = low[start:start + size]
        hits = sum(window.count(t) for t in terms)
        if hits > best_hits:
            best, best_hits = start, hits
    return text[best:best + size]


def page_text(raw: bytes, ctype: str, limit: int = MAX_PAGE_CHARS) -> tuple[str, str]:
    charset = re.search(r"charset=([\w-]+)", ctype or "")
    text = raw.decode(charset.group(1) if charset else "utf-8", errors="replace")
    if "html" not in ctype and not text.lstrip().startswith("<"):
        return "", re.sub(r"\s+", " ", text)[:limit]
    parser = _Text()
    parser.feed(text)
    body = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    body = re.sub(r"\n\s*\n+", "\n", body).strip()
    return parser.title.strip(), body[:limit]


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


class WebLookup(Integration):
    id = "web"
    title = "Web lookup"
    category = "Internet"
    description = ("Look things up and read web pages when you ask. Searches go to the provider you choose; "
                   "nothing else leaves this computer.")
    fields = (
        Field("provider", "Search with", FieldKind.CHOICE, default="wikipedia", options=PROVIDERS,
              help="Wikipedia needs nothing but only knows encyclopedia topics. For the whole web (news, prices, "
                   "sports, opening hours) like a regular assistant: Tavily (app.tavily.com) or Brave "
                   "(api-dashboard.search.brave.com), both with a free plan and a key. SearXNG: one you host. "
                   "Any other API: the Custom API card."),
        Field("api_key", "API key", FieldKind.SECRET, required=False, help="For Tavily or Brave."),
        Field("url", "SearXNG address", FieldKind.URL, "http://localhost:8888", required=False,
              help="Only for SearXNG (JSON output must be enabled)."),
    )
    tools = (
        ToolSpec("web.search", "Search the web", "Find facts and current information."),
        ToolSpec("web.read", "Read a page", "Read a web page and use what it says."),
        ToolSpec("web.weather", "Weather", "Current weather and the next three days, anywhere."),
    )
    examples = ("What's the weather like?", "Will it rain tomorrow in Chicago?",
                "Look up how tall the Eiffel Tower is.", "Who won the last World Cup?")

    def check(self, config: dict) -> Status:
        provider = config.get("provider") or "wikipedia"
        found = self.search(config, "Eiffel Tower", 1, read=0)
        if not found["results"]:
            raise CheckFailed(f"{provider} answered, but found nothing for a test search")
        return Status(State.CONNECTED, f"search works ({provider})", {"provider": provider})

    # ------------------------------------------------------------ search
    def search(self, config: dict, query: str, limit: int = 5, read: int = READ_TOP) -> dict:
        """Results with snippets, plus the relevant passage of the top `read` pages."""
        query = query.strip()[:300]
        if not query:
            raise CheckFailed("nothing to search for")
        provider = config.get("provider") or "wikipedia"
        if provider == "tavily":
            return self._tavily(config, query, limit)        # its results already hold the page text
        if provider == "brave":
            found = self._brave(config, query, limit)
        elif provider == "searxng":
            found = self._searxng(config, query, limit)
        else:
            found = self._wikipedia(query, limit)
        if read and found["results"]:
            found["pages"] = self._read_top(found["results"][:read], query)
        return found

    def _read_top(self, results: list[dict], query: str) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        def one(result):
            try:
                raw, ctype = _fetch(result["url"])
            except CheckFailed:
                return None
            title, text = page_text(raw, ctype, limit=200_000)
            passage = around(text, query, EXTRACT_CHARS)
            return {"url": result["url"], "title": title or result["title"], "text": passage} if passage else None
        with ThreadPoolExecutor(max_workers=len(results), thread_name_prefix="web-read") as pool:
            return [p for p in pool.map(one, results) if p]

    def _wikipedia(self, query: str, limit: int) -> dict:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
                                    "format": "json", "utf8": 1})
        raw, _ = _fetch(f"https://en.wikipedia.org/w/api.php?{q}")
        hits = json.loads(raw).get("query", {}).get("search", [])
        results = [{"title": h["title"], "snippet": _strip_tags(h.get("snippet", "")),
                    "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(h["title"].replace(" ", "_"))}
                   for h in hits]
        summary = ""
        if results:
            title = urllib.parse.quote(results[0]["title"].replace(" ", "_"), safe="")
            try:
                raw, _ = _fetch(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
                summary = json.loads(raw).get("extract", "")
            except CheckFailed:
                summary = ""
        return {"provider": "wikipedia", "results": results, "summary": summary[:1500]}

    def _tavily(self, config: dict, query: str, limit: int) -> dict:
        key = config.get("api_key") or ""
        if not key:
            raise CheckFailed("Tavily needs an API key (app.tavily.com)")
        body = json.dumps({"query": query, "max_results": limit, "include_answer": True,
                           "search_depth": "basic"}).encode()
        raw, _ = _fetch("https://api.tavily.com/search", {"Authorization": f"Bearer {key}",
                                                          "Content-Type": "application/json"},
                        method="POST", body=body)
        data = json.loads(raw)
        results = [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": (h.get("content") or "")[:700]}
                   for h in data.get("results", [])]
        return {"provider": "tavily", "results": results, "summary": (data.get("answer") or "")[:1500]}

    def _brave(self, config: dict, query: str, limit: int) -> dict:
        key = config.get("api_key") or ""
        if not key:
            raise CheckFailed("Brave needs an API key (api.search.brave.com)")
        q = urllib.parse.urlencode({"q": query, "count": limit})
        raw, _ = _fetch(f"https://api.search.brave.com/res/v1/web/search?{q}",
                        {"X-Subscription-Token": key, "Accept": "application/json"})
        hits = json.loads(raw).get("web", {}).get("results", [])
        return {"provider": "brave", "results": [{"title": h.get("title", ""), "url": h.get("url", ""),
                                                  "snippet": _strip_tags(h.get("description", ""))} for h in hits]}

    def _searxng(self, config: dict, query: str, limit: int) -> dict:
        base = (config.get("url") or "").rstrip("/")
        if not base:
            raise CheckFailed("SearXNG needs its address")
        q = urllib.parse.urlencode({"q": query, "format": "json"})
        raw, _ = _fetch(f"{base}/search?{q}", public_only=False)     # your own server, on purpose
        hits = json.loads(raw).get("results", [])[:limit]
        return {"provider": "searxng", "results": [{"title": h.get("title", ""), "url": h.get("url", ""),
                                                    "snippet": h.get("content", "")} for h in hits]}

    # ------------------------------------------------------------ weather
    def weather(self, place: str, latitude: float | None = None, longitude: float | None = None) -> dict:
        """Now and the next seven days at `place` (a town name, "Denver" or "Paris, France"), or at the
        given coordinates (then `place` is only its name)."""
        if latitude is not None and longitude is not None:
            spot = {"name": place, "latitude": latitude, "longitude": longitude,
                    "country_code": "US" if -170 < longitude < -50 and latitude > 15 else ""}
        else:
            spot = self._geocode(place)
        imperial = spot.get("country_code") in IMPERIAL_COUNTRIES
        q = urllib.parse.urlencode({
            "latitude": spot["latitude"], "longitude": spot["longitude"], "timezone": "auto", "forecast_days": 7,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,"
                       "precipitation",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": "fahrenheit" if imperial else "celsius",
            "wind_speed_unit": "mph" if imperial else "kmh"})
        raw, _ = _fetch(f"https://api.open-meteo.com/v1/forecast?{q}")
        data = json.loads(raw)
        deg, wind = ("°F", "mph") if imperial else ("°C", "km/h")
        now = data.get("current", {})
        daily = data.get("daily", {})
        days = []
        for i, day in enumerate(daily.get("time", [])[:7]):
            weekday = dt.date.fromisoformat(day).strftime("%A")
            days.append({"day": ("today", "tomorrow")[i] + f" ({weekday})" if i < 2 else weekday,
                         "sky": WEATHER_CODES.get(daily["weather_code"][i], "unknown"),
                         "high": f"{round(daily['temperature_2m_max'][i])}{deg}",
                         "low": f"{round(daily['temperature_2m_min'][i])}{deg}",
                         "chance_of_rain": f"{daily['precipitation_probability_max'][i] or 0}%"})
        where = ", ".join(x for x in (spot.get("name"), spot.get("admin1"), spot.get("country")) if x)
        return {"place": where, "now": {
            "sky": WEATHER_CODES.get(now.get("weather_code"), "unknown"),
            "temperature": f"{round(now.get('temperature_2m', 0))}{deg}",
            "feels_like": f"{round(now.get('apparent_temperature', 0))}{deg}",
            "humidity": f"{now.get('relative_humidity_2m', 0)}%",
            "wind": f"{round(now.get('wind_speed_10m', 0))} {wind}"}, "forecast": days,
                "how_to_answer": "in one or two short spoken sentences, only what the user asked about"}

    def _geocode(self, place: str) -> dict:
        name, _, region = place.strip().partition(",")
        q = urllib.parse.urlencode({"name": name.strip()[:80], "count": 10, "language": "en", "format": "json"})
        raw, _ = _fetch(f"https://geocoding-api.open-meteo.com/v1/search?{q}")
        results = json.loads(raw).get("results") or []
        if not results:
            raise CheckFailed(f"I could not find a place called {place!r}")
        region = region.strip().lower()
        if region:                          # "Paris, Texas": prefer the one in that state or country
            for r in results:
                if region in " ".join(str(r.get(k, "")) for k in ("admin1", "country", "country_code")).lower():
                    return r
        return results[0]

    # ------------------------------------------------------------ reading
    def read(self, url: str) -> dict:
        raw, ctype = _fetch(url)
        title, text = page_text(raw, ctype)
        if not text:
            raise CheckFailed("that page has no readable text")
        return {"url": url, "title": title, "text": text}
