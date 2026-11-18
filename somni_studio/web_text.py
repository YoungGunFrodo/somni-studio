"""Small text helpers shared by the server."""
from __future__ import annotations

import re


def model_name(model: str) -> str:
    """'Qwen3-4B-Instruct-2507' from 'Qwen/Qwen3-4B-Instruct-2507' or a Hugging Face snapshot path."""
    path = str(model or "").replace("\\", "/").rstrip("/")
    hub = re.search(r"models--([^/]+)", path)
    return hub.group(1).split("--")[-1] if hub else path.split("/")[-1]
