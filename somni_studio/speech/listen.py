"""Speech-to-text with Whisper (faster-whisper, int8 on the CPU), and the wake word.

The browser records 16 kHz mono audio and posts one utterance at a time as a WAV file. The CPU
transcribes it in a worker process (see Transcriber), so the GPU stays with the language model;
base.en transcribes about 8x faster than real time on 4 threads of a desktop CPU.

Hands-free mode sends every utterance the browser's voice detector hears, and the server only
passes on what follows the wake word ("Somni, turn on the kitchen lights"). Whisper spells an
unusual name several ways, so Whisper is primed with the wake word (hotwords) and the match is by
a rough sound key ("Somni", "Somny", "Somnee" all match; "Sammy" and "Sunny" do not).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ..settings import STUDIO_ROOT, ListenSettings

SAMPLE_RATE = 16_000
MAX_SECONDS = 60
MAX_WAV_BYTES = 44 + SAMPLE_RATE * 2 * MAX_SECONDS
WAKE_SIMILARITY = 0.8           # how close a word must sound to the wake word (Whisper is also
                                # told to expect it, so it is usually spelled right)
WAKE_WINDOW = 3                 # the wake word must be among the first few words
FILLERS = {"hey", "hi", "ok", "okay", "yo", "hello", "so", "um", "uh", "oh"}


class ListenUnavailable(RuntimeError):
    pass


class BadAudio(ValueError):
    pass


def _sound(word: str) -> str:
    """A crude phonetic key: lower case, letters only, doubled letters and vowels folded."""
    w = re.sub(r"[^a-z]", "", word.lower())
    w = re.sub(r"(.)\1+", r"\1", w)
    w = w.replace("y", "i").replace("ee", "i").replace("ie", "i")
    return re.sub(r"[aeiou]+$", "i", w) if w else w


@dataclass(frozen=True)
class WakeResult:
    heard: bool
    command: str


def split_wake_word(text: str, wake_word: str) -> WakeResult:
    """Did the utterance start with the wake word? Returns what follows it."""
    words = text.split()
    target = _sound(wake_word)
    if not target:
        return WakeResult(False, "")
    for i, raw in enumerate(words[: WAKE_WINDOW + 1]):
        key = _sound(raw)
        if not key:
            continue
        if SequenceMatcher(None, key, target).ratio() >= WAKE_SIMILARITY:
            rest = " ".join(words[i + 1:]).lstrip(" ,.!?;:-")
            return WakeResult(True, rest[:1].upper() + rest[1:] if rest else "")
        if raw.lower().strip(",.!?") not in FILLERS:
            break                       # something else came first: not addressed to us
    return WakeResult(False, "")


def read_wav(data: bytes):
    """16-bit mono WAV -> float32 samples at 16 kHz (the format the browser sends)."""
    import numpy as np
    if len(data) > MAX_WAV_BYTES:
        raise BadAudio(f"audio longer than {MAX_SECONDS} s")
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise BadAudio("expected 16-bit mono audio")
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise BadAudio(f"not a WAV file: {exc}") from None
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if rate != SAMPLE_RATE and len(samples):
        positions = np.arange(0, len(samples), rate / SAMPLE_RATE)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


class Transcriber:
    """Whisper in a worker process (speech/stt_worker.py), started on first use.

    Never import faster-whisper in this process: its OpenMP runtime and PyTorch's cannot share a
    process (it aborts with "OMP: Error #15"). The worker never imports torch.
    """

    def __init__(self, settings: ListenSettings) -> None:
        self._settings = settings
        self._proc: subprocess.Popen | None = None
        self._key = None
        self._lock = threading.Lock()

    def configure(self, settings: ListenSettings) -> None:
        with self._lock:
            self._settings = settings

    def _path(self, settings: ListenSettings) -> Path:
        return settings.model_path / settings.model

    def missing(self) -> str | None:
        if importlib.util.find_spec("faster_whisper") is None:      # find, do not import (see above)
            return "the faster-whisper package is not installed (pip install -e .[listen])"
        if not (self._path(self._settings) / "model.bin").is_file():
            return f"Whisper {self._settings.model} is not downloaded (python scripts/fetch_listen.py)"
        return None

    def status(self) -> dict:
        s = self._settings
        problem = self.missing()
        return {"enabled": s.enabled, "ready": problem is None, "problem": problem, "model": s.model,
                "wake_word": s.wake_word, "threads": s.threads}

    # ------------------------------------------------------------ the worker process
    def _worker(self) -> subprocess.Popen:
        settings = self._settings
        key = (settings.model, settings.threads, str(settings.model_path))
        if self._proc is not None and self._proc.poll() is None and self._key == key:
            return self._proc
        self._stop_worker()
        problem = self.missing()
        if problem:
            raise ListenUnavailable(problem)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(STUDIO_ROOT), os.environ.get("PYTHONPATH", "")])}
        proc = subprocess.Popen(
            [sys.executable, "-m", "somni_studio.speech.stt_worker", "--model", str(self._path(settings)),
             "--threads", str(settings.threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
            creationflags=flags)
        ready = self._read(proc)
        if not ready.get("ready"):
            proc.kill()
            raise ListenUnavailable(ready.get("error") or "the speech recognizer did not start")
        self._proc, self._key = proc, key
        return proc

    @staticmethod
    def _read(proc: subprocess.Popen) -> dict:
        line = proc.stdout.readline() if proc.stdout else b""
        if not line:
            return {"error": "the speech recognizer stopped"}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"error": "the speech recognizer sent something unreadable"}

    def _stop_worker(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
            try:
                proc.wait(3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def warm_up(self) -> None:
        with self._lock:
            if self._settings.enabled and self.missing() is None:
                self._worker()

    def transcribe(self, wav: bytes) -> str:
        """Blocking (call from a thread)."""
        read_wav(wav)                                   # validate here, with a clear error
        with self._lock:
            reply = self._ask(wav)
            if reply.get("error") and not reply.get("bad_audio"):
                self._stop_worker()                     # one restart, then give up
                reply = self._ask(wav)
        if "error" in reply:
            raise (BadAudio if reply.get("bad_audio") else ListenUnavailable)(reply["error"])
        return reply.get("text", "")

    def _ask(self, wav: bytes) -> dict:
        try:
            proc = self._worker()
            header = {"bytes": len(wav), "hotwords": self._settings.wake_word.strip().capitalize()}
            proc.stdin.write((json.dumps(header) + "\n").encode("utf-8") + wav)
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            return {"error": f"the speech recognizer is not running: {exc}"}
        return self._read(proc)

    def shutdown(self) -> None:
        with self._lock:
            self._stop_worker()
