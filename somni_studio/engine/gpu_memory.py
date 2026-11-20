"""Keeping the model inside the graphics card's own memory.

When a program asks for more graphics memory than the card has, Windows does not refuse: it quietly
moves the rest into ordinary RAM ("shared GPU memory"). The model then runs at a fraction of its
speed and the whole PC stalls (measured: a 4B model holding 14 GB of a 16 GB card plus 14.7 GB of
shared memory; one answer took 407 s and the desktop froze).

Two rules prevent that:
- after each answer, the memory PyTorch kept for reuse is handed back, so other programs (the
  desktop, the browser) have room and the next answer starts from the model's own size;
- Somni may use at most CAP_FRACTION of the card. Past that an answer fails with a clear message
  instead of spilling into RAM.
"""
from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)

CAP_FRACTION = 0.8


class OutOfGraphicsMemory(RuntimeError):
    """An answer needed more graphics memory than Somni may use."""


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def cap(fraction: float = CAP_FRACTION) -> None:
    torch = _torch()
    if torch is None:
        return
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        total = torch.cuda.get_device_properties(0).total_memory
        log.info("graphics memory: Somni may use %.1f of %.1f GB", fraction * total / 2**30, total / 2**30)
    except (RuntimeError, AttributeError) as exc:      # not every backend supports a cap
        log.warning("could not cap graphics memory: %s", exc)


def release() -> None:
    """Give back what PyTorch holds for reuse (the model itself stays loaded)."""
    torch = _torch()
    if torch is None:
        return
    gc.collect()
    torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    torch = _torch()
    if torch is not None and isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


def usage() -> dict:
    torch = _torch()
    if torch is None:
        return {}
    return {"allocated_gb": round(torch.cuda.memory_allocated(0) / 2**30, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 2**30, 2)}
