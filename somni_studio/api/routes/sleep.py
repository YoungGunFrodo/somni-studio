"""Sleep on demand and morning review."""
from __future__ import annotations

import asyncio
import logging
import sys

from fastapi import APIRouter, Depends, HTTPException

from ...somni import sleep_runner
from ...somni.sleep_runner import SleepError
from ..state import Studio, get_studio, require_gpu, require_writable

router = APIRouter(prefix="/api", tags=["sleep"])
log = logging.getLogger(__name__)


@router.get("/sleep")
async def get_sleep(studio: Studio = Depends(get_studio)):
    return await asyncio.to_thread(studio.sleep.status, studio.paths)


async def begin_sleep(studio: Studio) -> None:
    """Unload the chat model (training needs the whole GPU) and fall asleep."""
    if studio.sleep.running:
        raise SleepError("already asleep")
    await asyncio.wrap_future(studio.engine.unload())
    studio.sleep.start(studio.settings.somni.repo_path, studio.paths, python=sys.executable)


async def sleep_when_idle(studio: Studio, wait_s: float = 120.0) -> None:
    """For the "go to sleep" tool: it is called while the assistant is still answering, so wait
    for the answer to finish first."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    while studio.engine.busy and loop.time() < deadline:
        await asyncio.sleep(0.25)
    try:
        await begin_sleep(studio)
    except SleepError as exc:
        log.warning("could not fall asleep: %s", exc)


@router.post("/sleep", status_code=202)
async def start_sleep(studio: Studio = Depends(get_studio)):
    require_writable(studio, "sleeping")
    require_gpu(studio)
    if studio.engine.busy:
        raise HTTPException(409, "the assistant is answering; try again when it is done")
    try:
        await begin_sleep(studio)
    except SleepError as exc:
        raise HTTPException(409, str(exc)) from None
    return studio.sleep.status(studio.paths)


@router.post("/sleep/wake", status_code=202)
async def wake(studio: Studio = Depends(get_studio)):
    try:
        studio.sleep.wake()
    except SleepError as exc:
        raise HTTPException(409, str(exc)) from None
    return studio.sleep.status(studio.paths)


@router.post("/sleep/stop", status_code=202)
async def stop(studio: Studio = Depends(get_studio)):
    try:
        await asyncio.to_thread(studio.sleep.stop)
    except SleepError as exc:
        raise HTTPException(409, str(exc)) from None
    return studio.sleep.status(studio.paths)


@router.get("/sleep/report")
async def report(studio: Studio = Depends(get_studio)):
    return {"text": await asyncio.to_thread(sleep_runner.report_text, studio.paths)}


@router.post("/review/{checkpoint}/{decision}")
async def review(checkpoint: str, decision: str, studio: Studio = Depends(get_studio)):
    require_writable(studio, "morning review")
    if decision not in ("approve", "reject"):
        raise HTTPException(404, "decision must be approve or reject")
    if checkpoint not in {c["checkpoint"] for c in sleep_runner.pending_review(studio.paths)}:
        raise HTTPException(404, f"{checkpoint} is not awaiting review")
    action = sleep_runner.approve if decision == "approve" else sleep_runner.reject
    try:
        await asyncio.to_thread(action, studio.paths, checkpoint)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    if decision == "approve":
        await asyncio.wrap_future(studio.engine.unload())    # the next question runs the new expert
    await studio.graph.refresh(force=True)
    return await asyncio.to_thread(studio.sleep.status, studio.paths)
