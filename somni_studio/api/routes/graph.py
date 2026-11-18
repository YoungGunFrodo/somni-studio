"""The graph, node details and memory files."""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from ...settings import DATA_DIR
from ..details import NodeNotFound, node_detail
from ..state import Studio, get_studio, require_local

router = APIRouter(prefix="/api", tags=["graph"])
OPEN_CACHE = DATA_DIR / "cache" / "files"


def _open_with_system(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - a file we just wrote, opened with its default app
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


@router.get("/graph")
async def get_graph(studio: Studio = Depends(get_studio)):
    return studio.graph.snapshot()


@router.get("/nodes/{node_id:path}")
async def get_node(node_id: str, studio: Studio = Depends(get_studio)):
    try:
        return await asyncio.to_thread(node_detail, studio.graph.graph, studio.reader, node_id)
    except NodeNotFound:
        raise HTTPException(404, f"no node {node_id}") from None


@router.get("/files/{episode_id}")
async def get_file(episode_id: int, studio: Studio = Depends(get_studio)):
    record = await asyncio.to_thread(studio.reader.file, episode_id)
    if record is None:
        raise HTTPException(404, f"no memory file for episode {episode_id}")
    return record


@router.post("/files/{episode_id}/open")
async def open_file(episode_id: int, request: Request, studio: Studio = Depends(get_studio)):
    require_local(request, "opening files in a desktop app")
    record = await asyncio.to_thread(studio.reader.file, episode_id)
    if record is None:
        raise HTTPException(404, f"no memory file for episode {episode_id}")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", record["title"]).strip("-")[:60] or "memory-file"
    path = OPEN_CACHE / f"{episode_id}-{slug}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{record['title']}\n\n{record['text']}\n", encoding="utf-8")
    await asyncio.to_thread(_open_with_system, path)
    return {"opened": str(path)}
