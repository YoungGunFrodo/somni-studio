"""Devices Somni controls directly on the network: scan, connect, switch, forget."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...devices.base import Command, DeviceError
from ..state import Studio, get_studio, require_local

router = APIRouter(prefix="/api/devices", tags=["devices"])


class ConnectBody(BaseModel):
    target: str                       # a found device's key, name, or "all"


class ControlBody(BaseModel):
    action: str
    brightness: int | None = None
    volume: int | None = None
    app: str = ""


def _state(studio: Studio, devices: list | None = None) -> dict:
    hub = studio.devices
    return {"links": [{k: v for k, v in link.items() if k != "key"} for link in hub.links],   # never the Hue key
            "devices": [d.to_wire() for d in (devices if devices is not None else [])],
            "found": [f.to_wire() for f in hub.found()]}


@router.get("")
async def list_devices(studio: Studio = Depends(get_studio)):
    return _state(studio, await asyncio.to_thread(studio.devices.devices, 5.0))


@router.post("/scan")
async def scan(request: Request, studio: Studio = Depends(get_studio)):
    require_local(request, "scanning the network")
    await asyncio.to_thread(studio.devices.scan)
    return _state(studio, await asyncio.to_thread(studio.devices.devices, 5.0))


@router.post("/connect")
async def connect(body: ConnectBody, request: Request, studio: Studio = Depends(get_studio)):
    require_local(request, "connecting devices")
    try:
        report = await asyncio.to_thread(studio.devices.connect, body.target)
    except DeviceError as exc:
        raise HTTPException(404, str(exc)) from None
    return {"report": report, **_state(studio, await asyncio.to_thread(studio.devices.devices))}


@router.post("/{device_id:path}/control")
async def control(device_id: str, body: ControlBody, studio: Studio = Depends(get_studio)):
    found = [d for d in await asyncio.to_thread(studio.devices.devices, 5.0) if d.id == device_id]
    if not found:
        raise HTTPException(404, f"no device {device_id}")
    try:
        now = await asyncio.to_thread(studio.devices.control, found[0],
                                      Command(body.action, body.brightness, body.volume, body.app))
    except DeviceError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"device": now.to_wire()}


@router.delete("/links/{link_id:path}")
async def forget(link_id: str, request: Request, studio: Studio = Depends(get_studio)):
    require_local(request, "removing devices")
    if not await asyncio.to_thread(studio.devices.forget, link_id):
        raise HTTPException(404, f"no connection {link_id}")
    return _state(studio, await asyncio.to_thread(studio.devices.devices))
