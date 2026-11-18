"""Listening: the browser posts one utterance (16 kHz mono WAV) and gets the transcript back.

    POST /api/listen?mode=talk   push to talk: everything said is the message
    POST /api/listen?mode=wake   hands-free: only what follows the wake word ("Somni, ...")
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from ...speech.listen import MAX_WAV_BYTES, BadAudio, ListenUnavailable, split_wake_word
from ..state import Studio, get_studio

router = APIRouter(prefix="/api", tags=["listen"])


@router.get("/listen")
async def listen_status(studio: Studio = Depends(get_studio)):
    return studio.listen.status()


@router.post("/listen")
async def listen(request: Request, mode: str = "talk", studio: Studio = Depends(get_studio)):
    if mode not in ("talk", "wake"):
        raise HTTPException(400, "mode must be talk or wake")
    if not studio.settings.listen.enabled:
        raise HTTPException(409, "listening is switched off (Voice tab)")
    if int(request.headers.get("content-length") or 0) > MAX_WAV_BYTES:
        raise HTTPException(413, "that recording is too long")
    audio = await request.body()
    if len(audio) > MAX_WAV_BYTES:
        raise HTTPException(413, "that recording is too long")
    try:
        text = await asyncio.to_thread(studio.listen.transcribe, audio)
    except BadAudio as exc:
        raise HTTPException(400, str(exc)) from None
    except ListenUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    if mode == "talk":
        return {"text": text, "heard_wake": False, "command": text}
    wake = split_wake_word(text, studio.settings.listen.wake_word)
    return {"text": text, "heard_wake": wake.heard, "command": wake.command if wake.heard else ""}
