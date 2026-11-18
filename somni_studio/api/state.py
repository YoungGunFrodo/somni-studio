"""The objects every route shares, and small request helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from ..engine.worker import EngineWorker
from ..events import EventBus
from ..gpu_guard import GpuGuard
from ..graph.service import GraphService
from ..integrations.service import ConnectionsService
from ..settings import LOOPBACK_HOSTS, StudioSettings
from ..somni.home import HomePaths
from ..somni.sleep_runner import SleepRunner
from ..somni.store_reader import StoreReader
from ..speech.listen import Transcriber
from ..speech.service import SpeechService


@dataclass
class Studio:
    settings: StudioSettings
    settings_path: Path
    bus: EventBus
    graph: GraphService
    engine: EngineWorker
    speech: SpeechService
    sleep: SleepRunner
    connections: ConnectionsService
    listen: Transcriber
    gpu: GpuGuard | None = None
    devices: object = None                 # devices.hub.DeviceHub

    @property
    def paths(self) -> HomePaths:
        return HomePaths(self.settings.somni.home_path)

    @property
    def reader(self) -> StoreReader:
        return StoreReader(self.paths.memory_db)


def get_studio(request: Request) -> Studio:
    return request.app.state.studio


def is_loopback(host: str | None) -> bool:
    return (host or "") in LOOPBACK_HOSTS


def require_local(request: Request, what: str) -> None:
    """Some actions (credentials, opening desktop apps) are only allowed from this computer, even
    if the server is ever exposed on a network."""
    if not is_loopback(request.client.host if request.client else None):
        raise HTTPException(403, f"{what} is only allowed from this computer")


def require_gpu(studio: Studio) -> None:
    """While a game (or anything else) has the GPU, the assistant does not take it back."""
    if studio.gpu is not None and studio.gpu.paused:
        raise HTTPException(409, studio.gpu.pause_message())


def require_writable(studio: Studio, what: str) -> None:
    if studio.settings.somni.read_only:
        raise HTTPException(409, f"the Somni home is read-only, so {what} is off (Settings → somni.read_only)")
