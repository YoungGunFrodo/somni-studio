"""The FastAPI application: REST for state and commands (api/routes/), one WebSocket for live
events, and the static UI. Commands are REST because they have answers and errors; everything that
streams is an event.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..devices.hub import DeviceHub
from ..engine.providers import ToolEnvironment, make_provider
from ..engine.worker import EngineWorker
from ..events import EventBus, EventType
from ..gpu_guard import GpuGuard, GuardHooks
from ..graph.service import GraphService
from ..integrations.service import ConnectionsService, ConnectionStore
from ..settings import DATA_DIR, STUDIO_ROOT, SettingsError, StudioSettings
from ..somni.bootstrap import init_home
from ..somni.home import HomePaths
from ..somni.identity import ensure_identity, note_connection
from ..somni.runtime import SomniNotFound, ensure_sleepai
from ..somni.sleep_runner import SleepRunner
from ..speech.listen import Transcriber
from ..speech.service import SpeechService
from .routes import ROUTERS
from .state import Studio

log = logging.getLogger(__name__)

WEB_ROOT = STUDIO_ROOT / "web"


# ---------------------------------------------------------------- security helpers
def _import_map_hash() -> str:
    """CSP hash of the inline import map in index.html (the only inline script)."""
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    if not m:
        return ""
    return "'sha256-" + base64.b64encode(hashlib.sha256(m.group(1).encode("utf-8")).digest()).decode() + "'"


def _csp() -> str:
    return "; ".join([
        "default-src 'self'",
        f"script-src 'self' {_import_map_hash()}".strip(),
        "style-src 'self'",
        "img-src 'self' data: blob:",
        "media-src 'self' blob:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
    ])


def _same_origin(origin: str | None, host_header: str | None) -> bool:
    """Browsers always send Origin on WebSocket upgrades; a page on another site must not be able
    to listen in on (or drive) the studio."""
    if not origin:
        return True
    return urlparse(origin).netloc == (host_header or "")


def prepare_home(settings: StudioSettings) -> str | None:
    """Create the assistant's blank home on first run. Returns a problem to report, or None."""
    try:
        ensure_sleepai(settings.somni.repo_path)
    except SomniNotFound as exc:
        return str(exc)
    paths = HomePaths(settings.somni.home_path)
    if settings.somni.read_only:
        return None
    if init_home(paths):
        log.info("created a new, empty Somni home at %s", paths.home)
    if paths.config.exists() and ensure_identity(paths):
        log.info("added the assistant's core identity memories")
    return None


def _gpu_hooks(studio: Studio) -> GuardHooks:
    from .routes.sleep import begin_sleep

    async def unload() -> None:
        await asyncio.wrap_future(studio.engine.unload())

    async def stop_sleep(reason: str) -> None:
        await asyncio.to_thread(studio.sleep.stop, reason)

    return GuardHooks(
        model_loaded=lambda: studio.engine.status()["state"] in ("ready", "busy", "loading"),
        unload_model=unload, load_model=studio.engine.load,
        sleeping=lambda: studio.sleep.running, stop_sleep=stop_sleep,
        start_sleep=lambda: begin_sleep(studio))


def say_on_speaker(speech: SpeechService, devices: DeviceHub, device, message: str) -> None:
    """Somni's own voice, played on a speaker or display in the house."""
    from ..devices.base import DeviceError
    from ..devices.share import share_once
    voice = speech.status()
    if not voice["ready"]:
        raise DeviceError(f"my voice is not set up ({voice['problem']}), so I cannot speak on a speaker")
    try:
        audio = speech.clip(speech.preview(message))
    except Exception as exc:                       # missing voice model, nothing speakable
        raise DeviceError(f"I could not make the voice for that: {exc}") from None
    link = next((x for x in devices.links if x["id"] == device.link), None)
    if link is None or audio is None:
        raise DeviceError(f"{device.name} is not connected")
    devices.play(device, share_once(audio, link["address"]), "Somni")


def system_info(studio) -> dict:
    """Somni's own systems, for its system_status tool."""
    import sqlite3
    engine = studio.engine.status()
    info = {"model": engine.get("model", ""), "precision": engine.get("dtype"),
            "learned_experts": engine.get("experts", 0)}
    try:
        with sqlite3.connect(f"file:{studio.paths.memory_db}?mode=ro", uri=True) as db:
            info["memories"] = db.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]
            info["conversations_logged"] = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            info["not_yet_learned_in_sleep"] = db.execute(
                "SELECT COUNT(*) FROM episodes WHERE formed = 0").fetchone()[0]
    except sqlite3.Error:
        pass
    sleep = studio.sleep.status()
    info["sleep"] = sleep["state"] + (f", last woke {_ago(sleep['ended_at'])}" if sleep.get("ended_at") else "")
    info["voice"] = "ready" if studio.speech.status()["ready"] else "not set up"
    info["listening"] = "ready" if studio.listen.status()["ready"] else "not set up"
    if studio.gpu is not None:
        gpu = studio.gpu.status()
        info["graphics_card"] = (f"paused for {gpu['by']['name']}" if gpu.get("paused") and gpu.get("by")
                                 else "free" if gpu.get("available") else "not watched")
    info["connections"] = {c["title"]: c["status"]["state"] for c in studio.connections.list()}
    info["devices"] = [link["name"] for link in studio.devices.links] if studio.devices else []
    return info


def _ago(when: float) -> str:
    import time
    minutes = int((time.time() - when) / 60)
    return f"{minutes} minutes ago" if minutes < 120 else f"{minutes // 60} hours ago"


# ---------------------------------------------------------------- application
def create_app(settings: StudioSettings, settings_path: Path, provider_factory=make_provider,
               graph_builder=None, data_dir: Path = DATA_DIR, gpu_monitor=None) -> FastAPI:
    bus = EventBus()
    speech = SpeechService(settings.voice, lambda payload: bus.publish(EventType.SPEECH, **payload))
    sleep_events = {"status": EventType.SLEEP_STATUS, "log": EventType.SLEEP_LOG, "phase": EventType.SLEEP_PHASE}
    sleep = SleepRunner(lambda kind, payload: bus.publish(sleep_events[kind], **payload))
    sleep.watch_home(settings.somni.home_path)       # a sleep left running by an earlier studio is ours
    connections = ConnectionsService(ConnectionStore(data_dir / "connections.json"),
                                     lambda payload: bus.publish(EventType.CONNECTION, **payload))
    devices = DeviceHub(data_dir / "devices.json")
    tool_env = ToolEnvironment(connections, devices=devices,
                               say_on=lambda device, message: say_on_speaker(speech, devices, device, message))
    note = lambda thing, sentence: note_connection(HomePaths(settings.somni.home_path), thing, sentence)  # noqa: E731
    connections.on_change = devices.on_change = note
    studio = Studio(
        settings=settings, settings_path=settings_path, bus=bus,
        graph=GraphService(bus, settings, **({"builder": graph_builder} if graph_builder else {})),
        engine=EngineWorker(bus, settings, provider_factory, speech, tool_env), speech=speech, sleep=sleep,
        connections=connections, listen=Transcriber(settings.listen), devices=devices)
    studio.gpu = GpuGuard(bus, settings.gpu, _gpu_hooks(studio), monitor=gpu_monitor)
    tool_env.system_info = lambda: system_info(studio)

    background: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()
        bus.attach(loop)
        from .routes.sleep import sleep_when_idle
        # The "go to sleep" tool runs on the engine thread, mid-answer: hand it to the event loop.
        tool_env.request_sleep = lambda: loop.call_soon_threadsafe(
            lambda: background.add(asyncio.create_task(sleep_when_idle(studio))))
        problem = await asyncio.to_thread(prepare_home, studio.settings)
        if problem:
            log.warning("%s", problem)
        await studio.graph.refresh(force=True)
        studio.graph.start()
        studio.gpu.start()
        speech.warm_up()
        background.add(asyncio.create_task(asyncio.to_thread(connections.check_all)))
        background.add(asyncio.create_task(asyncio.to_thread(studio.listen.warm_up)))
        try:
            yield
        finally:
            for task in background:
                task.cancel()
            await studio.graph.stop()
            await studio.gpu.stop()
            sleep.shutdown()
            speech.shutdown()
            await asyncio.to_thread(studio.listen.shutdown)
            await asyncio.to_thread(studio.engine.shutdown)

    app = FastAPI(title="Somni Studio", version=__version__, lifespan=lifespan, docs_url="/api/docs",
                  redoc_url=None, openapi_url="/api/openapi.json")
    app.state.studio = studio
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    csp = _csp()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", csp)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith("/vendor/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.exception_handler(SettingsError)
    async def settings_error(_request: Request, exc: SettingsError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/health")
    async def health():
        return {"ok": True, "version": __version__, "revision": studio.graph.revision,
                "graph_error": studio.graph.last_error, "clients": bus.client_count}

    for router in ROUTERS:
        app.include_router(router)

    @app.websocket("/ws")
    async def events(ws: WebSocket):
        if not _same_origin(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(code=1008)
            return
        await ws.accept()
        queue = bus.subscribe()
        try:
            await ws.send_json({"type": EventType.HELLO.value, "version": __version__,
                                "revision": studio.graph.revision, "engine": studio.engine.status(),
                                "sleep": {k: v for k, v in studio.sleep.status().items() if k != "log"},
                                "speech": studio.speech.status(), "listen": studio.listen.status(),
                                "gpu": studio.gpu.status()})

            async def pump() -> None:
                while True:
                    await ws.send_json(await queue.get())

            async def drain() -> None:            # we never expect messages; this sees the close
                while True:
                    await ws.receive_text()

            tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                    log.debug("websocket closed: %r", t.exception())
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(queue)

    app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
    return app
