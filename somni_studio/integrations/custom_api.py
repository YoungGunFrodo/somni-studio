"""Any API you have a key for: give its address, the key, and a sentence on what it is for.

The model never sees the key. It gets a tool (api_call) that calls this API, and Somni adds the
key to each call on the way out, so the key cannot end up in chat, in memories, or in what sleep
learns. Calls can only go to this API's own address.
"""
from __future__ import annotations

import json
import urllib.parse

from .base import CheckFailed, Field, FieldKind, Integration, State, Status, ToolSpec

AUTH_STYLES = ("bearer", "header", "query", "none")


class CustomApi(Integration):
    id = "custom_api"
    title = "Custom API"
    category = "Internet"
    description = ("Any web API you have a key for (search, news, sports, a calendar service...). Say what it is "
                   "for, and Somni uses it when it fits. The key stays here: Somni adds it to each call itself.")
    fields = (
        Field("name", "Name", FieldKind.TEXT, "Tavily", help="What you call it."),
        Field("url", "API address", FieldKind.URL, "https://api.example.com/v1",
              help="The base address from the API's documentation."),
        Field("auth", "Key goes in", FieldKind.CHOICE, default="bearer", options=AUTH_STYLES,
              help="bearer: 'Authorization: Bearer KEY' (most APIs). header: a header named below. query: a URL "
                   "parameter named below. none: no key."),
        Field("auth_name", "Header or parameter name", FieldKind.TEXT, "X-API-Key", required=False,
              help="Only for header or query, e.g. X-API-Key or api_key."),
        Field("api_key", "API key", FieldKind.SECRET, required=False),
        Field("about", "What it is for", FieldKind.TEXT,
              "Search the web: POST /search with {\"query\": \"...\"}",
              help="One or two sentences for the assistant: what to use it for, and the main call if you know it."),
    )
    tools = (ToolSpec("api.call", "Use your API", "Call the API you added, for what you said it is for."),)
    examples = ("Use my API to look up today's news.",)
    setup = ("Copy the API's address and your key from its website, say what it is for, and press Connect.",)

    def check(self, config: dict) -> Status:
        name = config.get("name") or "the API"
        try:
            self.call(config, "GET", "")
        except CheckFailed as exc:
            text = str(exc)
            if "HTTP 401" in text or "HTTP 403" in text:
                raise CheckFailed(f"{name} refused the key") from None
            if "HTTP " not in text:                       # not even an HTTP answer: wrong address
                raise
        return Status(State.CONNECTED, f"{name} answers")

    @staticmethod
    def request(config: dict, method: str, path: str) -> tuple[str, dict]:
        """The full address and headers for one call, the key added; only under the configured address."""
        base = (config.get("url") or "").rstrip("/")
        parts = urllib.parse.urlparse(base)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise CheckFailed("the API address must start with https://")
        path = path.strip()
        if "://" in path or path.startswith("//"):
            raise CheckFailed("give a path on the API (like /search), not a whole address")
        url = base + ("/" + path.lstrip("/") if path else "")
        if urllib.parse.urlparse(url).hostname != parts.hostname:
            raise CheckFailed("calls can only go to the API's own address")
        key, style, name = config.get("api_key") or "", config.get("auth") or "bearer", config.get("auth_name") or ""
        headers = {"Accept": "application/json"}
        if key and style == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif key and style == "header":
            headers[name or "X-API-Key"] = key
        elif key and style == "query":
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode({name or "api_key": key})
        if method.upper() not in ("GET", "POST"):
            raise CheckFailed("only GET and POST calls are allowed")
        return url, headers

    def call(self, config: dict, method: str, path: str, body: str = "") -> dict:
        from .web import _fetch, _public
        url, headers = self.request(config, method, path)
        host = urllib.parse.urlparse(url).hostname
        try:
            _public(host)
            public = True
        except CheckFailed:
            public = False                               # the user's own server on their network: allowed
        data = None
        if method.upper() == "POST":
            try:
                data = json.dumps(json.loads(body or "{}")).encode()
            except ValueError:
                raise CheckFailed("body must be JSON, like {\"query\": \"...\"}") from None
            headers["Content-Type"] = "application/json"
        try:
            raw, ctype = _fetch(url, headers, public_only=public, method=method.upper(), body=data)
        except CheckFailed as exc:
            key = config.get("api_key") or ""
            raise CheckFailed(str(exc).replace(key, "***") if key else str(exc)) from None
        text = raw.decode("utf-8", "replace")
        try:
            return {"data": json.loads(text)}
        except ValueError:
            return {"text": text[:4000]}

    @staticmethod
    def note(config: dict) -> str:
        """What the model is told about it (never the key)."""
        about = (config.get("about") or "").strip()
        return (f"The user connected an API called {config.get('name') or 'Custom API'} at {config.get('url')}. "
                f"What it is for: {about or 'not described'}. Use api_call to call it (Somni adds the key).")
