"""Connections (smart home, email) and the tool catalogue."""
from __future__ import annotations

import asyncio
import secrets
import time
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
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


# ---- "Sign in with Home Assistant": its own login page, no token to copy (OAuth2 as Home Assistant does it)
SIGN_IN_TTL_S = 600


def _origin(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.get("/connections/home_assistant/signin")
async def ha_sign_in(request: Request, url: str = "", studio: Studio = Depends(get_studio)):
    require_local(request, "signing in to Home Assistant")
    ha = (url or studio.connections.config("home_assistant").get("url") or "").strip().rstrip("/")
    if not ha.startswith(("http://", "https://")):
        raise HTTPException(400, "enter or find the Home Assistant address first")
    state = secrets.token_urlsafe(24)
    pending = request.app.state.__dict__.setdefault("ha_sign_ins", {})
    for key in [k for k, (_, at) in pending.items() if time.time() - at > SIGN_IN_TTL_S]:
        pending.pop(key)
    pending[state] = (ha, time.time())
    origin = _origin(request)
    query = urllib.parse.urlencode({"client_id": origin + "/", "state": state,
                                    "redirect_uri": origin + "/api/connections/home_assistant/callback"})
    return RedirectResponse(f"{ha}/auth/authorize?{query}", status_code=303)


@router.get("/connections/home_assistant/callback")
async def ha_callback(request: Request, code: str = "", state: str = "", studio: Studio = Depends(get_studio)):
    require_local(request, "signing in to Home Assistant")
    pending = request.app.state.__dict__.get("ha_sign_ins", {})
    entry = pending.pop(state, None)
    if entry is None or not code:
        return RedirectResponse("/?signin=expired", status_code=303)
    ha, client_id = entry[0], _origin(request) + "/"
    from ...integrations.base import CheckFailed
    from ...integrations.home_assistant import auth_token
    try:
        reply = await asyncio.to_thread(auth_token, ha, {"grant_type": "authorization_code", "code": code,
                                                         "client_id": client_id})
        status = await asyncio.to_thread(studio.connections.sign_in, "home_assistant", {
            "url": ha, "token": "", "refresh_token": reply["refresh_token"], "client_id": client_id})
    except (CheckFailed, KeyError):
        return RedirectResponse("/?signin=failed", status_code=303)
    return RedirectResponse(f"/?signin={'connected' if status.state.value == 'connected' else 'failed'}",
                            status_code=303)


# ---- the Home Assistant VM, run by Somni
@router.get("/connections/home_assistant/vm")
async def ha_vm_status(studio: Studio = Depends(get_studio)):
    from ...integrations import ha_vm
    return await asyncio.to_thread(ha_vm.status, studio.connections.config("home_assistant").get("vm", ""))


@router.post("/connections/home_assistant/vm/{action}")
async def ha_vm_control(action: str, request: Request, studio: Studio = Depends(get_studio)):
    require_local(request, "starting or stopping Home Assistant")
    from ...integrations import ha_vm
    vm = await asyncio.to_thread(ha_vm.find_vm, studio.connections.config("home_assistant").get("vm", ""))
    if vm is None:
        raise HTTPException(404, "no Home Assistant virtual machine found")
    if action not in ("start", "stop"):
        raise HTTPException(404, f"unknown action {action}")
    try:
        await asyncio.to_thread(ha_vm.start if action == "start" else ha_vm.stop, vm)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return await asyncio.to_thread(ha_vm.status, vm)


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
