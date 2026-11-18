"""Whisper in its own process: python -m somni_studio.speech.stt_worker --model DIR --threads N

Why a separate process: faster-whisper's CTranslate2 and PyTorch each ship an OpenMP runtime, and
loading both into one process aborts it ("OMP: Error #15"). The worker never imports torch.

Protocol (binary-safe, over stdin/stdout):
    request:  one JSON line {"bytes": n, "hotwords": "Somni"}, then n bytes of 16-bit mono WAV
    response: one JSON line {"text": "..."} or {"error": "..."}
The worker prints {"ready": true} once the model is loaded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


def _reply(out, payload: dict) -> None:
    out.write((json.dumps(payload) + "\n").encode("utf-8"))
    out.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    from faster_whisper import WhisperModel

    from .listen import SAMPLE_RATE, BadAudio, read_wav
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    _reply(stdout, {"ready": True})
    while True:
        header = stdin.readline()
        if not header:
            return 0                                 # the studio closed the pipe
        try:
            request = json.loads(header)
            audio = stdin.read(int(request["bytes"]))
            samples = read_wav(audio)
            if len(samples) < SAMPLE_RATE // 5:
                _reply(stdout, {"text": ""})
                continue
            segments, _info = model.transcribe(samples, language="en", beam_size=1, vad_filter=True,
                                               condition_on_previous_text=False,
                                               hotwords=request.get("hotwords") or None)
            text = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()
            _reply(stdout, {"text": text})
        except BadAudio as exc:
            _reply(stdout, {"error": str(exc), "bad_audio": True})
        except Exception as exc:                     # report, keep serving
            _reply(stdout, {"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    sys.exit(main())
