"""The engine worker: one thread that owns the provider and runs every model call in order."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..events import EventBus, EventType
from ..settings import StudioSettings
from .types import Emitter, Provider

if TYPE_CHECKING:
    from ..speech.service import SpeechService

log = logging.getLogger(__name__)

MAX_PENDING = 4                 # questions waiting at once; more is refused (HTTP 429)
MAX_QUESTION_CHARS = 4000


class EngineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


class EngineBusy(RuntimeError):
    """Too many questions are already waiting."""


ProviderFactory = Callable[[StudioSettings], Provider]


class SpeakingEmitter(Emitter):
    """Also feeds generated text to text-to-speech, so sentences are spoken while the model writes."""

    def __init__(self, bus: EventBus, request_id: str, session) -> None:
        super().__init__(bus, request_id)
        self._session = session

    def __call__(self, kind: EventType, **payload: Any) -> None:
        super().__call__(kind, **payload)
        if kind is EventType.TOKEN:
            try:
                self._session.feed(payload.get("text", ""), bool(payload.get("replace")))
            except Exception:
                log.exception("speech feed failed")


def _provider_key(s: StudioSettings) -> tuple:
    """Settings that require a different provider (everything else applies without a reload)."""
    return (s.provider, s.somni)


class EngineWorker:
    def __init__(self, bus: EventBus, settings: StudioSettings, factory: ProviderFactory,
                 speech: SpeechService | None = None, tool_env=None) -> None:
        self._bus = bus
        self._speech = speech
        self._tool_env = tool_env
        self._settings = settings
        self._factory = factory
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="somni-engine")
        self._lock = threading.Lock()
        self._provider: Provider | None = None
        self._state = EngineState.UNLOADED
        self._detail = ""
        self._pending = 0

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        with self._lock:
            info = self._provider.describe() if self._provider is not None else {}
            return {"state": self._state.value, "detail": self._detail, "pending": self._pending,
                    "provider": self._settings.provider.kind, "read_only": self._settings.somni.read_only,
                    **{k: v for k, v in info.items() if k not in ("provider", "read_only")}}

    def _set(self, state: EngineState, detail: str = "") -> None:
        with self._lock:
            self._state, self._detail = state, detail
        self._bus.publish(EventType.ENGINE_STATUS, **self.status())

    # ------------------------------------------------------------ provider lifecycle (worker thread)
    def _ensure_loaded(self) -> Provider:
        if self._provider is not None:
            return self._provider
        self._set(EngineState.LOADING, f"loading {self._settings.provider.kind} provider")
        started = time.monotonic()
        provider = self._factory(self._settings)
        if hasattr(provider, "tool_env"):
            provider.tool_env = self._tool_env
        try:
            provider.load()
        except Exception as exc:
            try:
                provider.close()
            finally:
                self._set(EngineState.ERROR, f"{type(exc).__name__}: {exc}")
            raise
        with self._lock:
            self._provider = provider
        self._set(EngineState.READY, f"loaded in {time.monotonic() - started:.1f}s")
        return provider

    def _close(self) -> None:
        with self._lock:
            provider, self._provider = self._provider, None
        if provider is not None:
            try:
                provider.close()
            except Exception:
                log.exception("closing the provider failed")
        self._set(EngineState.UNLOADED)

    def _run_load(self) -> None:
        try:
            self._ensure_loaded()
        except Exception:
            log.exception("loading the provider failed")

    def _run_ask(self, request_id: str, question: str) -> None:
        session = self._speech.session(request_id) if self._speech is not None else None
        emit = SpeakingEmitter(self._bus, request_id, session) if session else Emitter(self._bus, request_id)
        try:
            provider = self._ensure_loaded()
            self._set(EngineState.BUSY, "answering")
            emit(EventType.CHAT_START, question=question)
            started = time.monotonic()
            outcome = provider.ask(question, emit)
            emit(EventType.CHAT_DONE, text=outcome.text, used_node=outcome.used_node, caveat=outcome.caveat,
                 episode_id=outcome.episode_id, recall=[r.to_wire() for r in outcome.recall], tools=outcome.tools,
                 elapsed_ms=round((time.monotonic() - started) * 1000), spoken=session is not None)
            if session is not None:
                session.finish(outcome.text)
            self._set(EngineState.READY)
        except Exception as exc:
            log.exception("answering failed")
            emit(EventType.CHAT_ERROR, message=f"{type(exc).__name__}: {exc}")
            with self._lock:
                failed_to_load = self._provider is None
            if not failed_to_load:
                self._set(EngineState.READY, "last question failed")
        finally:
            with self._lock:
                self._pending -= 1

    # ------------------------------------------------------------ public API (any thread)
    def load(self) -> Future:
        return self._pool.submit(self._run_load)

    def unload(self) -> Future:
        return self._pool.submit(self._close)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending > 0

    def ask(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("the question is empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"the question is longer than {MAX_QUESTION_CHARS} characters")
        with self._lock:
            if self._pending >= MAX_PENDING:
                raise EngineBusy(f"{self._pending} questions are already waiting")
            self._pending += 1
        request_id = uuid.uuid4().hex[:12]
        self._pool.submit(self._run_ask, request_id, question)
        return request_id

    def reconfigure(self, settings: StudioSettings) -> Future | None:
        """Apply new settings. The loaded provider is closed only if it depends on what changed;
        the next question loads the new one."""
        old = self._settings
        self._settings = settings
        if _provider_key(old) != _provider_key(settings):
            return self._pool.submit(self._close)
        return None

    def shutdown(self) -> None:
        self._pool.submit(self._close)
        self._pool.shutdown(wait=True, cancel_futures=True)
