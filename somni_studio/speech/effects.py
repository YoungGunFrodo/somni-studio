"""Voice effects, applied to each spoken sentence on the CPU (numpy only, a few milliseconds).

assistant  the sound of a synthetic house AI: a faint metallic sheen (two short comb taps), a small
           hard room (early reflections that die in 80 ms), low rumble trimmed, presence lifted and
           harsh highs softened. Subtle on purpose: it should sound produced, not robotic.
"""
from __future__ import annotations

import numpy as np

EFFECTS = ("none", "assistant")
PEAK = 0.89                       # -1 dBFS

# (milliseconds, gain): the comb taps make the sheen, the rest is the room
SHEEN = ((4.1, 0.22), (9.7, -0.12))
ROOM = ((17, 0.11), (23, -0.08), (31, 0.07), (43, 0.05), (59, 0.035), (79, 0.02))
TAIL_MS = 90


def assistant(samples: np.ndarray, rate: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x

    def at(ms: float) -> int:
        return int(rate * ms / 1000)

    ir = np.zeros(at(TAIL_MS) + 1, dtype=np.float32)
    ir[0] = 1.0
    for ms, gain in SHEEN + ROOM:
        ir[at(ms)] += gain
    length = x.size + ir.size - 1
    size = 1 << (length - 1).bit_length()
    spectrum = np.fft.rfft(x, size) * np.fft.rfft(ir, size)
    f = np.fft.rfftfreq(size, 1 / rate)
    eq = np.clip(f / 120.0, 0.3, 1.0)                                  # trim rumble below 120 Hz
    eq = eq * (1 + 0.41 * np.exp(-(((f - 4000) / 1800) ** 2)))         # +3 dB presence around 4 kHz
    eq = eq * np.where(f > 9000, np.exp(-(f - 9000) / 6000), 1.0)      # soften the very top
    y = np.fft.irfft(spectrum * eq, size)[: x.size + at(TAIL_MS)]
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * PEAK).astype(np.float32)


def apply(effect: str, samples: np.ndarray, rate: int) -> np.ndarray:
    if effect == "assistant":
        return assistant(samples, rate)
    return np.asarray(samples, dtype=np.float32)
