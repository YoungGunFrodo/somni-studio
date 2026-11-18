"""Text-to-speech: status, voices, previews and the audio clips the browser plays."""
from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from ...speech.service import SpeechUnavailable
from ..state import Studio, get_studio

router = APIRouter(prefix="/api", tags=["speech"])
CLIP_ID = re.compile(r"^[0-9a-f]{32}$")


class Preview(BaseModel):
    text: str = Field(min_length=1, max_length=400)
    voice: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    blend: str | None = None
    blend_amount: float | None = Field(default=None, ge=0.0, le=0.8)
    effect: str | None = Field(default=None, pattern="^(none|assistant)$")
    pause: float | None = Field(default=None, ge=0.0, le=1.5)


@router.get("/speech")
async def speech_status(studio: Studio = Depends(get_studio)):
    try:
        voices = await asyncio.to_thread(studio.speech.voices)
    except SpeechUnavailable:
        voices = []
    return {**studio.speech.status(), "voices": voices}


@router.post("/speech/preview")
async def preview(body: Preview, studio: Studio = Depends(get_studio)):
    try:
        clip = await asyncio.to_thread(studio.speech.preview, body.text, body.voice, body.speed,
                                       blend=body.blend, blend_amount=body.blend_amount, effect=body.effect,
                                       pause=body.pause)
    except SpeechUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"clip": clip}


@router.get("/speech/clips/{clip}.wav")
async def clip(clip: str, studio: Studio = Depends(get_studio)):
    if not CLIP_ID.match(clip):
        raise HTTPException(404, "no such clip")
    audio = studio.speech.clip(clip)
    if audio is None:
        raise HTTPException(404, "no such clip (clips are kept briefly)")
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "private, max-age=600"})
