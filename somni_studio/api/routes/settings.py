"""Studio settings."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ...settings import DTYPE_CHOICES, LISTEN_MODELS, PROVIDER_KINDS, field_types, save_settings
from ..state import Studio, get_studio

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/settings")
async def get_settings(studio: Studio = Depends(get_studio)):
    return {"settings": studio.settings.to_dict(), "path": str(studio.settings_path), "types": field_types(),
            "choices": {"provider.kind": list(PROVIDER_KINDS), "provider.dtype": list(DTYPE_CHOICES),
                        "listen.model": list(LISTEN_MODELS)}}


async def apply_settings(studio: Studio, patch: dict) -> dict:
    new = studio.settings.merged(patch)
    note = "server settings take effect after a restart" if new.server != studio.settings.server else ""
    await asyncio.to_thread(save_settings, new, studio.settings_path)
    studio.settings = new
    studio.speech.configure(new.voice)
    studio.listen.configure(new.listen)
    if studio.gpu is not None:
        studio.gpu.configure(new.gpu)
    studio.engine.reconfigure(new)
    await studio.graph.reconfigure(new)
    return {"settings": new.to_dict(), "note": note}


@router.put("/settings")
async def put_settings(patch: dict, studio: Studio = Depends(get_studio)):
    return await apply_settings(studio, patch)
