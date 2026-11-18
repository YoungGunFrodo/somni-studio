"""On-device text-to-speech with Kokoro (82M parameters, Apache-2.0) through ONNX Runtime on the CPU.

Why this design keeps the system fast:
  * the CPU does speech while the GPU does the language model, so they do not compete;
  * a fixed, small thread budget (voice.threads, default 4) leaves the rest of the CPU free;
  * sentences are synthesized as soon as the model finishes writing them, one worker in order,
    so the first words play about half a second after they are written (measured 4.2x faster
    than real time on 4 threads of a desktop CPU with the full-precision model; the int8 build
    was 10x slower there, so it is not used).

Audio is kept in a small in-memory cache and fetched by the browser as WAV.
"""
from __future__ import annotations

import collections
import io
import logging
import threading
import uuid
import wave
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..settings import VoiceSettings
from .text import SentenceSplitter, speakable

log = logging.getLogger(__name__)

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
CACHE_CLIPS = 64
PREVIEW_MAX_CHARS = 400

# Friendly names for the English voices (Kokoro v1.0). Others are listed by id.
VOICE_LABELS = {
    "af_heart": "Heart — US, female", "af_bella": "Bella — US, female", "af_nicole": "Nicole — US, female (soft)",
    "af_sarah": "Sarah — US, female", "af_sky": "Sky — US, female", "af_nova": "Nova — US, female",
    "af_aoede": "Aoede — US, female", "af_kore": "Kore — US, female", "af_alloy": "Alloy — US, female",
    "af_jessica": "Jessica — US, female", "af_river": "River — US, female",
    "am_michael": "Michael — US, male", "am_fenrir": "Fenrir — US, male", "am_puck": "Puck — US, male",
    "am_adam": "Adam — US, male", "am_echo": "Echo — US, male", "am_eric": "Eric — US, male",
    "am_liam": "Liam — US, male", "am_onyx": "Onyx — US, male",
    "bf_emma": "Emma — UK, female", "bf_isabella": "Isabella — UK, female", "bf_alice": "Alice — UK, female",
    "bf_lily": "Lily — UK, female", "bm_george": "George — UK, male", "bm_fable": "Fable — UK, male",
    "bm_daniel": "Daniel — UK, male", "bm_lewis": "Lewis — UK, male",
}

Publish = Callable[[dict], None]       # speech event payload -> the event bus


def _lang(voice: str) -> str:
    return "en-gb" if voice.startswith("b") else "en-us"


def _wav(samples, rate: int) -> bytes:
    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class SpeechUnavailable(RuntimeError):
    pass


class KokoroVoice:
    """The model, loaded on first use."""

    def __init__(self, model_dir: Path, threads: int) -> None:
        self.model_dir, self.threads = model_dir, threads
        self._kokoro = None
        self._lock = threading.Lock()

    def missing(self) -> str | None:
        try:
            import kokoro_onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return "the kokoro-onnx package is not installed (pip install -e .[voice])"
        for name in (MODEL_FILE, VOICES_FILE):
            if not (self.model_dir / name).is_file():
                return f"voice model file {name} is missing (python scripts/fetch_voice.py)"
        return None

    def _load(self):
        with self._lock:
            if self._kokoro is None:
                problem = self.missing()
                if problem:
                    raise SpeechUnavailable(problem)
                import onnxruntime as ort
                from kokoro_onnx import Kokoro
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.threads
                opts.inter_op_num_threads = 1
                session = ort.InferenceSession(str(self.model_dir / MODEL_FILE), opts,
                                               providers=["CPUExecutionProvider"])
                self._kokoro = Kokoro.from_session(session, str(self.model_dir / VOICES_FILE))
            return self._kokoro

    def voices(self) -> list[str]:
        return sorted(self._load().get_voices())

    def synth(self, text: str, style: VoiceStyle) -> bytes:
        import numpy as np

        from .effects import apply
        kokoro = self._load()
        voice = style.voice
        if style.blend and style.blend_amount > 0:      # a mix of two voices' style vectors
            a = style.blend_amount
            voice = (1 - a) * kokoro.get_voice_style(style.voice) + a * kokoro.get_voice_style(style.blend)
        samples, rate = kokoro.create(text, voice=voice, speed=style.speed, lang=_lang(style.voice))
        samples = apply(style.effect, samples, rate)
        if style.pause > 0:
            samples = np.concatenate([samples, np.zeros(int(rate * style.pause), dtype=np.float32)])
        return _wav(samples, rate)


@dataclass(frozen=True)
class VoiceStyle:
    """How Somni sounds: a voice (optionally blended with another), speed, effect, pacing."""
    voice: str
    speed: float = 1.0
    blend: str = ""
    blend_amount: float = 0.0
    effect: str = "none"
    pause: float = 0.0

    @classmethod
    def of(cls, settings: VoiceSettings, **overrides) -> VoiceStyle:
        values = {k: getattr(settings, k) for k in ("voice", "speed", "blend", "blend_amount", "effect", "pause")}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


class SpeechSession:
    """One answer: sentences go to the synthesis worker in order as they complete."""

    def __init__(self, service: SpeechService, request_id: str) -> None:
        self._service, self.request_id = service, request_id
        self._splitter = SentenceSplitter()
        self._seq = 0

    def feed(self, piece: str, replace: bool = False) -> None:
        for sentence in self._splitter.feed(piece, replace):
            self._say(sentence)

    def finish(self, final_text: str | None) -> None:
        for sentence in self._splitter.finish(final_text):
            self._say(sentence)

    def _say(self, sentence: str) -> None:
        self._service.enqueue(self.request_id, self._seq, sentence)
        self._seq += 1


class SpeechService:
    def __init__(self, settings: VoiceSettings, publish: Publish) -> None:
        self._publish = publish
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech")
        self._clips: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._settings = settings
        self._voice = KokoroVoice(settings.model_path, settings.threads)
        self._warned = False

    # ------------------------------------------------------------ configuration
    def configure(self, settings: VoiceSettings) -> None:
        if (settings.model_path, settings.threads) != (self._settings.model_path, self._settings.threads):
            self._voice = KokoroVoice(settings.model_path, settings.threads)
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def status(self) -> dict:
        problem = self._voice.missing()
        s = self._settings
        return {"enabled": s.enabled, "ready": problem is None, "problem": problem, "voice": s.voice,
                "speed": s.speed, "threads": s.threads, "blend": s.blend, "blend_amount": s.blend_amount,
                "effect": s.effect, "pause": s.pause}

    def voices(self) -> list[dict]:
        ids = self._voice.voices() if self._voice.missing() is None else list(VOICE_LABELS)
        english = [v for v in ids if v[:2] in ("af", "am", "bf", "bm")]
        return [{"id": v, "label": VOICE_LABELS.get(v, v)} for v in english]

    def warm_up(self) -> Future:
        """Load the model in the background so the first answer speaks without delay."""
        return self._pool.submit(self._warm)

    def _warm(self) -> None:
        if self._settings.enabled and self._voice.missing() is None:
            try:
                self._voice.synth("Ready.", VoiceStyle.of(self._settings))
            except Exception:
                log.exception("warming up the voice failed")

    # ------------------------------------------------------------ speaking
    def session(self, request_id: str) -> SpeechSession | None:
        if not self._settings.enabled:
            return None
        problem = self._voice.missing()
        if problem:
            if not self._warned:
                log.warning("speech is on but unavailable: %s", problem)
                self._warned = True
            return None
        return SpeechSession(self, request_id)

    def enqueue(self, request_id: str, seq: int, sentence: str) -> None:
        self._pool.submit(self._speak, request_id, seq, sentence, VoiceStyle.of(self._settings))

    def _store(self, audio: bytes) -> str:
        clip = uuid.uuid4().hex
        with self._lock:
            self._clips[clip] = audio
            while len(self._clips) > CACHE_CLIPS:
                self._clips.popitem(last=False)
        return clip

    def _speak(self, request_id: str, seq: int, sentence: str, style: VoiceStyle) -> None:
        if not self._settings.enabled:            # muted while this was waiting
            return
        try:
            clip = self._store(self._voice.synth(sentence, style))
        except Exception as exc:
            log.exception("speech synthesis failed")
            self._publish({"request_id": request_id, "seq": seq, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._publish({"request_id": request_id, "seq": seq, "clip": clip, "text": sentence})

    def preview(self, text: str, voice: str | None = None, speed: float | None = None, **style) -> str:
        """Blocking: synthesize `text` now (call from a thread). Returns the clip id. Unset style
        values (blend, blend_amount, effect, pause) come from the settings."""
        text = speakable(text)[:PREVIEW_MAX_CHARS]
        if not text:
            raise ValueError("nothing to say")
        chosen = VoiceStyle.of(self._settings, voice=voice or None, speed=speed or None, **style)
        future = self._pool.submit(self._voice.synth, text, chosen)
        return self._store(future.result(timeout=60))

    def clip(self, clip_id: str) -> bytes | None:
        with self._lock:
            return self._clips.get(clip_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
