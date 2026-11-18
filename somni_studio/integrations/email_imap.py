"""Email over IMAP (read-only). Works with Gmail, Outlook.com, iCloud and most providers; those
with two-factor sign-in need an app password. Nothing is ever sent or deleted."""
from __future__ import annotations

import contextlib
import email
import email.header
import imaplib
import re
import ssl

from .base import CheckFailed, Field, FieldKind, Integration, State, Status, ToolSpec

TIMEOUT_S = 10


class EmailImap(Integration):
    id = "email"
    title = "Email"
    category = "Email"
    description = "Read-only access to your inbox over IMAP: unread mail, search, summaries."
    fields = (
        Field("host", "IMAP server", FieldKind.TEXT, "imap.gmail.com"),
        Field("port", "Port", FieldKind.INT, "993", default=993),
        Field("username", "Email address", FieldKind.TEXT, "you@example.com"),
        Field("password", "App password", FieldKind.SECRET, "",
              "Gmail: Google Account → Security → 2-Step Verification → App passwords. "
              "Your normal password usually does not work over IMAP."),
    )
    tools = (
        ToolSpec("email.unread", "Unread mail", "List and summarize unread messages."),
        ToolSpec("email.search", "Search mail", "Find messages by sender, subject or text."),
    )

    def check(self, config: dict) -> Status:
        host, user, password = (config.get("host") or "").strip(), (config.get("username") or "").strip(), \
            config.get("password") or ""
        try:
            port = int(config.get("port") or 993)
        except (TypeError, ValueError):
            raise CheckFailed("the port must be a number") from None
        if not host or not user or not password:
            raise CheckFailed("server, email address and app password are all needed")
        try:
            imap = imaplib.IMAP4_SSL(host, port, ssl_context=ssl.create_default_context(), timeout=TIMEOUT_S)
        except (TimeoutError, OSError, ssl.SSLError) as exc:
            raise CheckFailed(f"cannot reach {host}:{port}: {exc}") from None
        try:
            imap.login(user, password)
            imap.select("INBOX", readonly=True)
            _, data = imap.search(None, "UNSEEN")
            unread = len(data[0].split()) if data and data[0] else 0
        except imaplib.IMAP4.error as exc:
            message = str(exc)
            raise CheckFailed("sign-in refused: check the address and app password"
                              if "auth" in message.lower() or "login" in message.lower() else message) from None
        finally:
            with contextlib.suppress(imaplib.IMAP4.error, OSError):
                imap.logout()
        return Status(State.CONNECTED, f"{unread} unread in the inbox", {"unread": unread, "account": user})

    # ------------------------------------------------------------ actions (used by tools)
    def _open(self, config: dict):
        imap = imaplib.IMAP4_SSL(config["host"], int(config.get("port") or 993),
                                 ssl_context=ssl.create_default_context(), timeout=TIMEOUT_S)
        imap.login(config["username"], config["password"])
        imap.select("INBOX", readonly=True)            # read-only: nothing is marked as read
        return imap

    def _headers(self, imap, ids: list[bytes], limit: int) -> list[dict]:
        out = []
        for msg_id in reversed(ids[-limit:]):          # newest first
            _, data = imap.fetch(msg_id, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            raw = data[0][1] if data and isinstance(data[0], tuple) else b""
            msg = email.message_from_bytes(raw)
            out.append({"from": _decode(msg.get("From", "")), "subject": _decode(msg.get("Subject", "")),
                        "date": msg.get("Date", "")})
        return out

    def unread(self, config: dict, limit: int = 5) -> dict:
        imap = self._open(config)
        try:
            _, data = imap.search(None, "UNSEEN")
            ids = data[0].split() if data and data[0] else []
            return {"unread": len(ids), "messages": self._headers(imap, ids, limit)}
        finally:
            with contextlib.suppress(imaplib.IMAP4.error, OSError):
                imap.logout()

    def search(self, config: dict, query: str, limit: int = 5) -> dict:
        query = re.sub(r'["\\\r\n]', " ", query).strip()[:100]
        if not query:
            return {"found": 0, "messages": []}
        imap = self._open(config)
        try:
            _, data = imap.search(None, "OR", "FROM", f'"{query}"', "SUBJECT", f'"{query}"')
            ids = data[0].split() if data and data[0] else []
            return {"found": len(ids), "messages": self._headers(imap, ids, limit)}
        finally:
            with contextlib.suppress(imaplib.IMAP4.error, OSError):
                imap.logout()


def _decode(value: str) -> str:
    parts = []
    for text, charset in email.header.decode_header(value):
        parts.append(text.decode(charset or "utf-8", "replace") if isinstance(text, bytes) else text)
    return " ".join(" ".join(parts).split())[:200]
