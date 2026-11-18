"""Inference precision for the in-process model.

Somni loads its student in bf16, the right choice for training. For inference, GPUs without
native bf16 math run it through slow fallback kernels: measured on a Radeon RX 6800 (RDNA2,
gfx1030) Qwen3-1.7B generates 5.5 tok/s in bf16 and 26.1 tok/s in fp16, with identical text.
`auto` therefore picks fp16 on such GPUs and bf16 where the hardware has it.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

# ROCm architectures with native bf16 matrix math: CDNA2/3 (MI200/MI300) and RDNA3/4.
_ROCM_FAST_BF16 = ("gfx90a", "gfx94", "gfx95", "gfx11", "gfx12")


def resolve_dtype(choice: str):
    """torch dtype for `choice` on this machine's device."""
    import torch
    if choice != "auto":
        return getattr(torch, choice)
    if not torch.cuda.is_available():
        return torch.float32
    if torch.version.hip:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
        return torch.bfloat16 if arch.startswith(_ROCM_FAST_BF16) else torch.float16
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16       # NVIDIA: bf16 from Ampere on


@contextlib.contextmanager
def somni_load_dtype(dtype) -> Iterator[None]:
    """Make Somni's model loader (sleepai.models.hf._load, which asks peft_trainer.pick_device for
    the device and dtype at call time) load in `dtype`. Everything else about loading (expert
    wrapping, the model cache, GPU hand-over) stays Somni's own. The studio never trains, so
    nothing in this process depends on the training dtype."""
    from sleepai.models import peft_trainer

    original = peft_trainer.pick_device

    def pick_device():
        device, _ = original()
        return device, dtype

    peft_trainer.pick_device = pick_device
    try:
        yield
    finally:
        peft_trainer.pick_device = original
