"""Connections (smart home, email) and the tool catalogue."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...integrations.service import UnknownIntegration
from ..state import Studio, get_studio, require_local

router = APIRouter(prefix="/api", tags=["connections"])


class ConnectionValues(BaseModel):
    values: dict


@router.get("/connections")
async def list_connections(studio: Studio = Depends(get_studio)):
    return {"connections": studio.connections.list()}


@router.put("/connections/{integration_id}")
async def save_connection(integration_id: str, body: ConnectionValues, request: Request,
                          studio: Studio = Depends(get_studio)):
    require_local(request, "changing connections")
    try:
        status = await asyncio.to_thread(studio.connections.save, integration_id, body.values)
    except UnknownIntegration:
        raise HTTPException(404, f"no integration {integration_id}") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"status": status.to_wire()}


@router.post("/connections/{integration_id}/test")
async def test_connection(integration_id: str, studio: Studio = Depends(get_studio)):
    try:
        status = await asyncio.to_thread(studio.connections.check, integration_id)
    except UnknownIntegration:
        raise HTTPException(404, f"no integration {integration_id}") from None
    return {"status": status.to_wire()}


@router.delete("/connections/{integration_id}")
async def remove_connection(integration_id: str, request: Request, studio: Studio = Depends(get_studio)):
    require_local(request, "changing connections")
    try:
        await asyncio.to_thread(studio.connections.remove, integration_id)
    except UnknownIntegration:
        raise HTTPException(404, f"no integration {integration_id}") from None
    return {"removed": integration_id}


@router.post("/connections/home_assistant/discover")
async def discover_home_assistant(request: Request, studio: Studio = Depends(get_studio)):
    """Look for Home Assistant at its usual addresses on this network."""
    require_local(request, "looking for Home Assistant")
    from ...integrations.home_assistant import HomeAssistant
    return {"found": await asyncio.to_thread(HomeAssistant.discover)}


@router.get("/tools")
async def list_tools(studio: Studio = Depends(get_studio)):
    return {"tools": studio.connections.tools(
        speech_ready=studio.speech.status()["ready"], listen_ready=studio.listen.status()["ready"],
        tools_enabled=studio.settings.tools.enabled, writable=not studio.settings.somni.read_only,
        device_links=studio.devices.links if studio.devices is not None else [])}
