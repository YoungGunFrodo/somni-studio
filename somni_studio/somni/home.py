"""The files of a Somni home, and what the brain graph needs from them.

A home holds memory.db (memory store), checkpoints/registry.json (adopted checkpoints) and
adapters/ (expert folders written by the stacked-experts trainer). The experts the model runs are
the active checkpoint's; a benchmark home that never adopted one falls back to the newest trained
expert folder, and the UI says so.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FALLBACK_LAYERS = 28


@dataclass(frozen=True)
class HomePaths:
    home: Path

    @property
    def memory_db(self) -> Path:
        return self.home / "memory.db"

    @property
    def registry(self) -> Path:
        return self.home / "checkpoints" / "registry.json"

    @property
    def adapters(self) -> Path:
        return self.home / "adapters"

    @property
    def config(self) -> Path:
        return self.home / "config.json"

    def watched(self) -> list[Path]:
        """Files whose changes mean the graph may have changed."""
        db = self.memory_db
        return [db, db.with_name(db.name + "-journal"), db.with_name(db.name + "-wal"),
                self.registry, self.config, self.adapters]


@dataclass(frozen=True)
class Expert:
    index: int
    threshold: float | None            # routing threshold; None = always on
    calibration: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertSet:
    directory: Path | None
    source: str                         # active | latest | override | none
    checkpoint: str | None
    targets: tuple[str, ...]
    rank: int | None
    experts: tuple[Expert, ...]


@dataclass(frozen=True)
class ModelShape:
    name: str
    layers: int
    hidden_size: int | None
    exact: bool                         # False when config.json was not found (layer count guessed)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def registry(paths: HomePaths) -> dict:
    return read_json(paths.registry)


def _expert_set(directory: Path, source: str, checkpoint: str | None) -> ExpertSet:
    cfg = read_json(directory / "experts.json")
    count = int(cfg.get("experts", 0))
    thresholds = list(cfg.get("route") or [None] * count)
    thresholds += [None] * (count - len(thresholds))
    calibration = cfg.get("route_calibration") or {}
    experts = tuple(Expert(k, None if t is None else float(t), calibration if k == count - 1 else {})
                    for k, t in enumerate(thresholds[:count]))
    return ExpertSet(directory, source, checkpoint, tuple(cfg.get("targets") or DEFAULT_TARGETS),
                     cfg.get("rank"), experts)


def find_experts(paths: HomePaths, override: str = "") -> ExpertSet:
    none = ExpertSet(None, "none", None, DEFAULT_TARGETS, None, ())
    if override:
        directory = Path(override).expanduser()
        return _expert_set(directory, "override", None) if (directory / "experts.json").exists() else none
    active = registry(paths).get("active")
    if active:
        directory = paths.home / "checkpoints" / active / "adapter"
        if (directory / "experts.json").exists():
            return _expert_set(directory, "active", active)
    if paths.adapters.is_dir():
        trained = [d for d in paths.adapters.iterdir() if (d / "experts.json").exists()]
        if trained:
            newest = max(trained, key=lambda d: (d.stat().st_mtime, d.name))
            return _expert_set(newest, "latest", None)
    return none


def base_model(paths: HomePaths) -> str:
    """The model this home runs and trains: `base_model` in the home's config.json (Somni's own)."""
    return read_json(paths.config).get("base_model") or DEFAULT_BASE_MODEL


def _hf_hub_dirs() -> list[Path]:
    out = []
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]) / "hub")
    out.append(Path.home() / ".cache" / "huggingface" / "hub")
    return out


def model_config(name: str) -> dict:
    """config.json of a local model folder or a cached Hugging Face model, without importing torch."""
    local = Path(name).expanduser()
    if (local / "config.json").exists():
        return read_json(local / "config.json")
    folder = "models--" + name.replace("/", "--")
    for hub in _hf_hub_dirs():
        snapshots = hub / folder / "snapshots"
        if snapshots.is_dir():
            found = sorted(snapshots.glob("*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                return read_json(found[0])
    return {}


def model_shape(name: str) -> ModelShape:
    cfg = model_config(name)
    layers = cfg.get("num_hidden_layers")
    return ModelShape(name, int(layers or FALLBACK_LAYERS), cfg.get("hidden_size"), exact=bool(layers))
