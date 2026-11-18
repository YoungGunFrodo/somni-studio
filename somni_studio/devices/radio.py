"""Internet radio: something to play when you ask for music, with no account.

Stations come from Radio Browser (radio-browser.info), a free, open directory of tens of thousands
of stations. Only the words you asked for are sent ("jazz", "BBC Radio 2"). The speaker then
streams the station directly.
"""
from __future__ import annotations

import json
import urllib.parse

from ..integrations.base import CheckFailed

API_HOSTS = ("https://de1.api.radio-browser.info", "https://nl1.api.radio-browser.info",
             "https://all.api.radio-browser.info")
PLAYABLE = {"MP3": "audio/mpeg", "AAC": "audio/aac", "AAC+": "audio/aac"}   # what cast speakers play
GENERIC = {"", "music", "some music", "anything", "something", "songs", "a song", "the radio", "radio"}
DEFAULT_TAG = "pop music"            # measured: the most-played English hits station (Capital)


def _get(path: str, params: dict) -> list:
    from ..integrations.web import _fetch
    query = urllib.parse.urlencode(params)
    for host in API_HOSTS:
        try:
            raw, _ = _fetch(f"{host}/json/{path}?{query}")
            return json.loads(raw)
        except (CheckFailed, ValueError):
            continue
    raise CheckFailed("the radio directory (radio-browser.info) is not answering")


def find_station(what: str) -> dict:
    """The most listened-to working station for a genre or a station name."""
    wanted = what.strip().lower()
    wanted = "" if wanted in GENERIC else wanted.removesuffix(" music").removesuffix(" radio").strip()
    common = {"hidebroken": "true", "order": "clickcount", "reverse": "true", "limit": 25}
    by_tag = {"tag": wanted or DEFAULT_TAG, "tagExact": "true", **common}
    candidates = _get("stations/search", {**by_tag, "language": "english"})      # the user speaks English
    if not any(s.get("codec") in PLAYABLE for s in candidates):
        candidates = _get("stations/search", by_tag)
    if wanted and not any(s.get("codec") in PLAYABLE for s in candidates):
        candidates = _get("stations/search", {"name": wanted, **common})
    playable = [s for s in candidates if s.get("codec") in PLAYABLE and (s.get("url_resolved") or s.get("url"))]
    playable.sort(key=lambda s: not (s.get("url_resolved") or s.get("url")).startswith("https://"))  # https first
    for station in playable:
        url = station.get("url_resolved") or station.get("url")
        return {"name": station.get("name", "").strip() or "a radio station", "url": url,
                "content_type": PLAYABLE[station["codec"]], "country": station.get("country", "")}
    raise CheckFailed(f"I could not find a radio station for {what!r}")
