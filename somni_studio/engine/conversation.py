"""The running conversation: what was just said, and the episode it is logged in.

One episode is one conversation, not one question: "you can call me sir" only makes sense next to
the "what's my name?" before it, and sleep's memory formation reads an episode as a whole. A
conversation ends after a quiet gap, when sleep has already formed its episode, or when it grows
long; the next turn then starts a new episode.

The last few turns also go back into the prompt, so the model knows what "it" and "that" mean.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

GAP_S = 30 * 60                  # this long without a word: a new conversation
MAX_TURNS = 40                   # user + assistant messages in one episode
HISTORY_EXCHANGES = 3            # question/answer pairs given back to the model
HISTORY_ANSWER_CHARS = 400       # a long answer is cut in the prompt (the episode keeps all of it)


@dataclass
class Conversation:
    episode: int | None = None
    turns: list[dict] = field(default_factory=list)      # what the episode holds
    last_at: float = 0.0

    def history(self, now: float | None = None) -> list[dict]:
        """The last exchanges as chat messages (user / assistant), for the prompt."""
        if self._stale(now):
            return []
        said = [t for t in self.turns if t.get("role") in ("user", "assistant")]
        out = []
        for t in said[-2 * HISTORY_EXCHANGES:]:
            text = str(t.get("content", ""))
            if t["role"] == "assistant" and len(text) > HISTORY_ANSWER_CHARS:
                text = text[:HISTORY_ANSWER_CHARS] + "…"
            out.append({"role": t["role"], "content": text})
        return out

    def _stale(self, now: float | None) -> bool:
        return not self.turns or (now or time.time()) - self.last_at > GAP_S

    def log(self, store, turns: list[dict], day: int, outcome: str = "unknown") -> int:
        """Add one exchange: appended to this conversation's episode, or the start of a new one."""
        now = time.time()
        if self.episode is not None and not self._stale(now) and len(self.turns) + len(turns) <= MAX_TURNS:
            merged = self.turns + turns
            cur = store.db.execute("UPDATE episodes SET transcript = ? WHERE id = ? AND formed = 0",
                                   (json.dumps(merged), self.episode))
            store.db.commit()
            if cur.rowcount == 1:                            # not formed by a sleep in the meantime
                self.turns, self.last_at = merged, now
                return self.episode
        self.episode = store.log_episode("conversation", turns, outcome=outcome, day=day)
        self.turns, self.last_at = list(turns), now
        return self.episode

    def reset(self) -> None:
        self.episode, self.turns, self.last_at = None, [], 0.0


def with_history(messages: list[dict], history: list[dict]) -> list[dict]:
    """Insert earlier turns after the system message (if any) and before the question."""
    if not history:
        return messages
    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    return head + history + messages[len(head):]
