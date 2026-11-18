"""Auto-pause for games: status, and the two buttons (resume now, always ignore this program)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..state import Studio, get_studio
from .settings import apply_settings

router = APIRouter(prefix="/api", tags=["gpu"])


class ResumeRequest(BaseModel):
    ignore_program: bool = False      # also never pause for this program again


@router.get("/gpu")
async def gpu_status(studio: Studio = Depends(get_studio)):
    if studio.gpu is None:
        raise HTTPException(404, "no GPU guard")
    users = await asyncio.to_thread(studio.gpu._monitor.sample) if studio.gpu._monitor.available else []
    return {**studio.gpu.status(), "using_now": [{"name": u.name, "percent": u.percent} for u in users[:6]]}


@router.post("/gpu/resume")
async def gpu_resume(body: ResumeRequest, studio: Studio = Depends(get_studio)):
    if studio.gpu is None or not studio.gpu.paused:
        raise HTTPException(409, "not paused")
    program = studio.gpu.status()["by"]["name"]
    if body.ignore_program:
        ignored = studio.settings.gpu.ignored | {program.lower()}
        await apply_settings(studio, {"gpu": {"ignore": ", ".join(sorted(ignored))}})
    await studio.gpu.resume(f"resumed by the user ({program})")
    return studio.gpu.status()
