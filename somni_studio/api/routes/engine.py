"""Loading the model and asking questions."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...engine.worker import MAX_QUESTION_CHARS, EngineBusy
from ..state import Studio, get_studio, require_gpu

router = APIRouter(prefix="/api", tags=["engine"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def _not_while_sleeping(studio: Studio) -> None:
    if studio.sleep.running:
        raise HTTPException(409, "the assistant is asleep; wake it first")


@router.get("/engine")
async def engine_status(studio: Studio = Depends(get_studio)):
    return studio.engine.status()


@router.post("/engine/load", status_code=202)
async def engine_load(studio: Studio = Depends(get_studio)):
    _not_while_sleeping(studio)
    require_gpu(studio)
    studio.engine.load()
    return studio.engine.status()


@router.post("/engine/unload", status_code=202)
async def engine_unload(studio: Studio = Depends(get_studio)):
    studio.engine.unload()
    return studio.engine.status()


@router.get("/conversation")
async def conversation(limit: int = 40, studio: Studio = Depends(get_studio)):
    """Recent exchanges, so the chat survives a page reload (they live in the episode log)."""
    return {"messages": await asyncio.to_thread(studio.reader.conversations, max(1, min(200, limit)))}


@router.post("/chat", status_code=202)
async def chat(body: ChatRequest, studio: Studio = Depends(get_studio)):
    _not_while_sleeping(studio)
    if studio.settings.provider.kind == "somni":        # a server provider does not need this GPU
        require_gpu(studio)
    try:
        return {"request_id": studio.engine.ask(body.question)}
    except EngineBusy as exc:
        raise HTTPException(429, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
