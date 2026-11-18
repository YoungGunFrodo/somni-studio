"""Turning a streamed answer into speakable sentences."""
from __future__ import annotations

import re

CITATION = re.compile(r"\[(?:memory:\s*)?[^\]\s]*[/:][^\]]*\]", re.I)    # [memory: key] or a bare [subject/attr]
MARKDOWN = re.compile(r"[*_`#>|]+")
SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n{2,}")
MAX_CHUNK = 260          # a clause this long is spoken without waiting for its full stop
MIN_CHUNK = 12           # very short fragments wait to be joined with the next sentence


def speakable(text: str) -> str:
    """What should be read aloud: no citations, no markdown symbols, tidy whitespace."""
    text = CITATION.sub("", text)
    text = MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Feed it the answer as it streams; it returns sentences as soon as they are complete.

    Tokens normally extend the text. A provider may also replace the whole text (a server that is
    not streamed, or a tokenizer merge); then only what was not spoken yet is used.
    """

    def __init__(self) -> None:
        self._text = ""
        self._spoken = 0           # characters of self._text already turned into sentences

    def feed(self, piece: str, replace: bool = False) -> list[str]:
        if replace:
            if not piece.startswith(self._text[: self._spoken]):
                self._spoken = 0 if not self._spoken else len(piece)    # cannot re-align: skip what was said
            self._text = piece
        else:
            self._text += piece
        return self._take(final=False)

    def finish(self, final_text: str | None = None) -> list[str]:
        out = []
        if final_text is not None and final_text != self._text:
            out += self.feed(final_text, replace=True)
        return out + self._take(final=True)

    def _take(self, final: bool) -> list[str]:
        out = []
        while True:
            rest = self._text[self._spoken:]
            match = SENTENCE_END.search(rest)
            if match:
                end = match.end()
            elif len(rest) > MAX_CHUNK:
                cut = max(rest.rfind(", ", 0, MAX_CHUNK), rest.rfind("; ", 0, MAX_CHUNK))
                end = cut + 2 if cut > MIN_CHUNK else MAX_CHUNK
            elif final and rest.strip():
                end = len(rest)
            else:
                break
            chunk = speakable(rest[:end])
            if len(chunk) < MIN_CHUNK and not final and match and end < len(rest):
                # too short on its own ("Yes."): join it with the next sentence
                nxt = SENTENCE_END.search(rest, end)
                if nxt is None:
                    break
                end = nxt.end()
                chunk = speakable(rest[:end])
            self._spoken += end
            if chunk:
                out.append(chunk)
        return out
