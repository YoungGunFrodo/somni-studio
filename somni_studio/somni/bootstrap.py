"""Creating the assistant's own Somni home, and choosing the model it runs.

A fresh install starts with an empty home (data/home by default): a Somni config.json set up for
the real system with stacked experts, and an empty memory store. Nothing is copied from anywhere.

The model is `base_model` in that config.json, because Somni's sleep trains experts for exactly
that model. Experts only fit the model they were trained on, so once a home has experts its model
is locked; moving learned knowledge to a new model is Somni's brain transfer.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .home import DEFAULT_BASE_MODEL, HomePaths, base_model, find_experts, read_json

# Somni settings for a home the studio creates: the real system with stacked experts, routed per
# question, subject tags after memory formation (they shape the memory graph), and wake in-process.
NEW_HOME_CONFIG = {
    "backend": "llamacpp",          # Somni's switch for "the real system" (not the toy brain)
    "trainer": "experts",
    "expert_route": True,
    "wake": "hf",
    "sleep": {"tags": True},
}


class ModelLocked(RuntimeError):
    """The home has experts trained for its current model."""


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_blank(home: Path) -> bool:
    return not home.exists() or not any(home.iterdir())


def init_home(paths: HomePaths, model: str = DEFAULT_BASE_MODEL) -> bool:
    """Create a blank home if there is nothing at `paths.home`. Returns True if it created one.
    An existing folder is never touched (it may be someone's Somni run)."""
    if not is_blank(paths.home):
        return False
    from sleepai.config import Config
    from sleepai.memory.store import MemoryStore

    paths.home.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(None, home=str(paths.home), base_model=model,
                      **{k: v for k, v in NEW_HOME_CONFIG.items() if k != "sleep"},
                      sleep=NEW_HOME_CONFIG["sleep"])
    cfg.save(paths.config)
    MemoryStore(paths.memory_db).db.close()         # an empty store with Somni's current schema
    return True


# Files tied to one model: its experts and checkpoints, and caches of that model's own answers
# (rehearsal = how the base model answers practice questions; reusing them for another model would
# teach it to imitate the old one). On a model switch they move to archive/, never deleted.
MODEL_SPECIFIC = ("checkpoints", "adapters", "probes.json", "retention_reference.json", "lookup_answers.json",
                  "training_pairs.json", "implications.json", "rehearsal.json")
TRANSFER_MARKER = "TRANSFER"            # the next sleep relearns every memory (sleep_entry.py)


def model_locked_by(paths: HomePaths) -> str | None:
    """Why switching needs a relearn, or None when the model can change freely."""
    experts = find_experts(paths)
    if experts.directory is not None:
        return (f"Somni has learned experts for {base_model(paths)}; they only fit that model. Switching keeps "
                "every memory, archives the old experts, and the next sleep teaches the new model everything again.")
    return None


def pending_transfer(paths: HomePaths) -> dict | None:
    """The relearn the next sleep will do after a model switch, or None."""
    marker = paths.home / TRANSFER_MARKER
    if not marker.exists():
        return None
    return read_json(marker) or {"to": base_model(paths)}


def _archive_model_files(paths: HomePaths, old_model: str) -> Path | None:
    import re
    import shutil
    import time
    items = [paths.home / name for name in MODEL_SPECIFIC if (paths.home / name).exists()]
    items += sorted(paths.home.glob("rehearsal-*.json"))
    if not items:
        return None
    slug = re.sub(r"[^A-Za-z0-9]+", "-", old_model.replace("\\", "/").split("models--")[-1]).strip("-")[:60]
    dest = paths.home / "archive" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug or 'model'}"
    dest.mkdir(parents=True)
    for item in items:
        shutil.move(str(item), str(dest / item.name))
    return dest


def set_model(paths: HomePaths, model: str, relearn: bool = False) -> dict:
    """Make `model` the one this home runs. If Somni has already trained experts for the current
    model, that needs `relearn`: the old model's files are archived, every memory is marked as not
    yet learned, and the next sleep relearns the whole memory store (Somni's brain transfer)."""
    model = model.strip()
    if not model:
        raise ValueError("choose a model")
    old = base_model(paths)
    if model == old:
        return {"changed": False}
    reason = model_locked_by(paths)
    if reason and not relearn:
        raise ModelLocked(reason)
    if not paths.config.exists():
        raise ValueError(f"{paths.home} has no Somni config.json")
    archived = _archive_model_files(paths, old)
    relearning = False
    if archived is not None:
        relearning = _forget_training(paths)
        if relearning:
            _write_json(paths.home / TRANSFER_MARKER, {"from": old, "to": model, "archive": str(archived)})
    cfg = read_json(paths.config)
    cfg["base_model"] = model
    _write_json(paths.config, cfg)
    from .identity import ensure_identity
    ensure_identity(paths)                      # "Somni currently thinks with <model>" stays true
    return {"changed": True, "archived": str(archived) if archived else None, "relearn": relearning}


def _forget_training(paths: HomePaths) -> bool:
    """Every memory goes back to "not in the weights yet". True if there is anything to relearn."""
    import sqlite3
    if not paths.memory_db.exists():
        return False
    db = sqlite3.connect(paths.memory_db)
    try:
        db.execute("UPDATE memories SET consolidated_in = '[]', last_probe_result = ''")
        db.commit()
        return db.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0] > 0
    finally:
        db.close()


# ---------------------------------------------------------------- models on this computer
@dataclass(frozen=True)
class LocalModel:
    id: str                          # display name (Hugging Face id, or folder name)
    load: str                        # what to put in base_model: the id (default cache) or a local folder
    source: str                      # huggingface-cache | folder
    size_gb: float
    layers: int | None
    architecture: str

    def to_wire(self) -> dict:
        return {"id": self.id, "load": self.load, "source": self.source, "size_gb": self.size_gb,
                "layers": self.layers, "architecture": self.architecture}


def _hub_roots() -> list[Path]:
    """Hugging Face caches that `from_pretrained(<id>)` reads without extra settings."""
    roots = []
    for env, sub in (("HF_HUB_CACHE", ""), ("HF_HOME", "hub")):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]) / sub)
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(r.resolve() for r in roots))


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt", ".gguf"):
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _causal_lm(cfg: dict) -> str | None:
    archs = cfg.get("architectures") or []
    return next((a for a in archs if a.endswith("ForCausalLM")), None)


MIN_WEIGHTS_BYTES = 50_000_000          # smaller downloads are tokenizers or configs, not models


def _scan_hub(root: Path, by_id: bool, min_bytes: int) -> list[LocalModel]:
    """Models in one Hugging Face cache. `by_id`: the cache transformers uses by default, so the
    model loads by its id; otherwise it loads from its snapshot folder (no re-download)."""
    out = []
    for folder in root.glob("models--*"):
        snaps = sorted(folder.glob("snapshots/*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snaps:
            continue
        cfg = read_json(snaps[0])
        arch = _causal_lm(cfg)
        size = _dir_size(folder / "blobs") or _dir_size(snaps[0].parent)
        if not arch or size < min_bytes:
            continue
        model_id = folder.name[len("models--"):].replace("--", "/")
        out.append(LocalModel(model_id, model_id if by_id else str(snaps[0].parent), "huggingface-cache",
                              round(size / 1e9, 2), cfg.get("num_hidden_layers"), arch))
    return out


def _scan_folder(root: Path, min_bytes: int) -> list[LocalModel]:
    """Plain model folders (each child holding config.json and weights)."""
    out = []
    for cfg_path in root.glob("*/config.json"):
        cfg = read_json(cfg_path)
        arch = _causal_lm(cfg)
        size = _dir_size(cfg_path.parent)
        if arch and size >= min_bytes:
            out.append(LocalModel(cfg_path.parent.name, str(cfg_path.parent), "folder", round(size / 1e9, 2),
                                  cfg.get("num_hidden_layers"), arch))
    return out


def parse_model_dirs(text: str) -> list[Path]:
    """`model_dirs` setting: folders separated by ';' (or new lines)."""
    from ..settings import resolve_path
    return [resolve_path(part.strip()) for part in text.replace("\n", ";").split(";") if part.strip()]


def local_models(extra_dirs: list[Path] = (), min_bytes: int = MIN_WEIGHTS_BYTES) -> list[LocalModel]:
    """Causal language models with downloaded weights: the default Hugging Face cache, plus any
    extra folders (another Hugging Face cache, or a folder of model folders). Embedding models,
    tokenizer-only downloads and models whose weights were never fetched are skipped."""
    found: dict[str, LocalModel] = {}
    for root in _hub_roots():
        if root.is_dir():
            for m in _scan_hub(root, by_id=True, min_bytes=min_bytes):
                found[m.load] = m
    for root in extra_dirs:
        root = Path(root)
        hub = root / "hub" if (root / "hub").is_dir() else root
        if not hub.is_dir():
            continue
        models = (_scan_hub(hub, by_id=False, min_bytes=min_bytes) if any(hub.glob("models--*"))
                  else _scan_folder(hub, min_bytes))
        for m in models:
            if not any(f.id == m.id for f in found.values()):      # already loadable by id
                found[m.load] = m
    return sorted(found.values(), key=lambda m: (m.size_gb, m.id))
