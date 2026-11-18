"""Choosing the model: what is installed on this computer, and what the home runs."""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...somni.bootstrap import (
    ModelLocked,
    local_models,
    model_locked_by,
    parse_model_dirs,
    pending_transfer,
    set_model,
)
from ...somni.home import base_model
from ..state import Studio, get_studio

router = APIRouter(prefix="/api", tags=["model"])
SERVER_TIMEOUT_S = 3


class ModelChoice(BaseModel):
    model: str = Field(min_length=1, max_length=500)
    relearn: bool = False            # experts exist: archive them and relearn every memory


def _server_models(url: str) -> dict:
    """Models an OpenAI-compatible server (LM Studio, llama-server) offers."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=SERVER_TIMEOUT_S) as r:  # noqa: S310
            data = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"reachable": False, "error": str(getattr(exc, "reason", exc)), "models": []}
    return {"reachable": True, "error": "", "models": sorted(m.get("id", "") for m in data.get("data", []))}


@router.get("/model")
async def get_model(studio: Studio = Depends(get_studio)):
    paths = studio.paths
    found = await asyncio.to_thread(local_models, parse_model_dirs(studio.settings.provider.model_dirs))
    current = base_model(paths)
    return {
        "current": current,
        "installed": any(m.load == current for m in found) or Path(current).is_dir(),
        "locked": model_locked_by(paths),
        "relearn_pending": pending_transfer(paths),
        "read_only": studio.settings.somni.read_only,
        "local": [m.to_wire() for m in found],
        "provider": asdict(studio.settings.provider),
        "engine": studio.engine.status(),
    }


@router.get("/model/server")
async def get_server_models(studio: Studio = Depends(get_studio)):
    return await asyncio.to_thread(_server_models, studio.settings.provider.url)


@router.put("/model")
async def put_model(choice: ModelChoice, studio: Studio = Depends(get_studio)):
    if studio.settings.somni.read_only:
        raise HTTPException(409, "the Somni home is read-only; its model cannot be changed from here")
    if studio.sleep.running:
        raise HTTPException(409, "the assistant is asleep; wait for the sleep to finish")
    try:
        result = await asyncio.to_thread(set_model, studio.paths, choice.model, choice.relearn)
    except ModelLocked as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    await asyncio.wrap_future(studio.engine.unload())       # the next question loads the new model
    await studio.graph.refresh(force=True)                   # the brain is drawn for the new model
    return {"current": base_model(studio.paths), **result}
