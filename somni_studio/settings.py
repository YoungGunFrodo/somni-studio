"""Studio settings: one JSON file with typed sections, validated on load and on every update.

    {
      "server":   {"host": "127.0.0.1", "port": 8765},
      "somni":    {"repo": "../somni", "home": "data/home", "adapter": "", "read_only": false, "use_tags": true},
      "provider": {"kind": "somni", "url": "http://127.0.0.1:1234", "model": "", "max_new_tokens": 256,
                   "dtype": "auto"},
      "voice":    {"enabled": true, "voice": "af_heart", "speed": 1.0, "threads": 4, "model_dir": "data/models/tts"},
      "listen":   {"enabled": true, "model": "base.en", "threads": 4, "wake_word": "somni",
                   "model_dir": "data/models/stt"},
      "tools":    {"enabled": true, "allowed_domains": "light,switch,fan,media_player,scene,climate,input_boolean"},
      "graph":    {"poll_seconds": 2.0, "max_memories": 20000}
    }

Unknown keys and wrong types are errors, so a typo never silently falls back to a default.
Which model the in-process assistant runs is not a studio setting: it is `base_model` in the Somni
home's own config.json, because Somni's sleep trains for that model (see somni/bootstrap.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

STUDIO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = STUDIO_ROOT / "data"                 # everything local to this install (gitignored)
DEFAULT_SETTINGS_PATH = STUDIO_ROOT / "studio.json"
SETTINGS_ENV = "SOMNI_STUDIO_CONFIG"

PROVIDER_KINDS = ("somni", "openai")
DTYPE_CHOICES = ("auto", "float16", "bfloat16", "float32")   # mirrored in engine/precision.py
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class SettingsError(ValueError):
    """Settings that cannot be used. The message names the offending key."""


def resolve_path(text: str) -> Path:
    """`~` expands to the home folder; relative paths are relative to the studio folder (not to
    wherever the server happened to be started)."""
    path = Path(text).expanduser()
    return path if path.is_absolute() else (STUDIO_ROOT / path).resolve()


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8765

    def validate(self) -> None:
        if not self.host.strip():
            raise SettingsError("server.host must not be empty")
        if not 1 <= self.port <= 65535:
            raise SettingsError("server.port must be between 1 and 65535")

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


@dataclass(frozen=True)
class SomniSettings:
    repo: str = "../somni"              # the Somni framework checkout (contains sleepai/)
    home: str = "data/home"             # the assistant's own Somni home, created blank on first run
    adapter: str = ""                   # optional expert folder overriding the active checkpoint
    read_only: bool = False             # True: chat never writes episodes (for browsing other homes)
    use_tags: bool = True               # recall also scans subject tags (Somni Orchestrator.use_tags)

    def validate(self) -> None:
        if not self.repo.strip():
            raise SettingsError("somni.repo must not be empty")
        if not self.home.strip():
            raise SettingsError("somni.home must not be empty")

    @property
    def repo_path(self) -> Path:
        return resolve_path(self.repo)

    @property
    def home_path(self) -> Path:
        return resolve_path(self.home)

    @property
    def adapter_path(self) -> str:
        return str(resolve_path(self.adapter)) if self.adapter.strip() else ""


@dataclass(frozen=True)
class ProviderSettings:
    kind: str = "somni"                 # somni: in-process model with live telemetry; openai: a server
    url: str = "http://127.0.0.1:1234"  # OpenAI-compatible server (LM Studio, llama-server)
    model: str = ""                     # model id sent to that server
    max_new_tokens: int = 256
    dtype: str = "auto"                 # in-process model precision (engine/precision.py)
    model_dirs: str = ""                # extra folders to look for models in, separated by ';'

    def validate(self) -> None:
        if self.dtype not in DTYPE_CHOICES:
            raise SettingsError(f"provider.dtype must be one of {', '.join(DTYPE_CHOICES)}")
        if self.kind not in PROVIDER_KINDS:
            raise SettingsError(f"provider.kind must be one of {', '.join(PROVIDER_KINDS)}")
        if self.kind == "openai" and not self.url.startswith(("http://", "https://")):
            raise SettingsError("provider.url must start with http:// or https://")
        if not 1 <= self.max_new_tokens <= 4096:
            raise SettingsError("provider.max_new_tokens must be between 1 and 4096")


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = True                # speak answers aloud (text-to-speech on the CPU)
    voice: str = "af_heart"             # a Kokoro voice id (see the Voice tab)
    speed: float = 1.0
    blend: str = ""                     # a second voice mixed into the first ("" = none)
    blend_amount: float = 0.0           # how much of the second voice, 0-0.8
    effect: str = "none"                # none | assistant (a produced, synthetic-AI sound)
    pause: float = 0.0                  # extra silence after each sentence, seconds
    threads: int = 4                    # CPU threads for speech; the GPU stays with the model
    model_dir: str = "data/models/tts"  # kokoro-v1.0.onnx + voices-v1.0.bin (scripts/fetch_voice.py)

    def validate(self) -> None:
        if not self.voice.strip():
            raise SettingsError("voice.voice must not be empty")
        if not 0.5 <= self.speed <= 2.0:
            raise SettingsError("voice.speed must be between 0.5 and 2.0")
        if not 0.0 <= self.blend_amount <= 0.8:
            raise SettingsError("voice.blend_amount must be between 0 and 0.8")
        if self.effect not in ("none", "assistant"):
            raise SettingsError("voice.effect must be none or assistant")
        if not 0.0 <= self.pause <= 1.5:
            raise SettingsError("voice.pause must be between 0 and 1.5 seconds")
        if not 1 <= self.threads <= 32:
            raise SettingsError("voice.threads must be between 1 and 32")

    @property
    def model_path(self) -> Path:
        return resolve_path(self.model_dir)


LISTEN_MODELS = ("tiny.en", "base.en", "small.en")


@dataclass(frozen=True)
class ListenSettings:
    enabled: bool = True                # the microphone button and hands-free listening
    model: str = "base.en"              # Whisper size: tiny.en | base.en | small.en (CPU, int8)
    threads: int = 4                    # CPU threads for speech recognition
    wake_word: str = "somni"            # hands-free mode only acts on what follows this word
    model_dir: str = "data/models/stt"  # downloaded Whisper models (scripts/fetch_listen.py)

    def validate(self) -> None:
        if self.model not in LISTEN_MODELS:
            raise SettingsError(f"listen.model must be one of {', '.join(LISTEN_MODELS)}")
        if not 1 <= self.threads <= 32:
            raise SettingsError("listen.threads must be between 1 and 32")
        if not self.wake_word.strip() or len(self.wake_word) > 40:
            raise SettingsError("listen.wake_word must be 1-40 characters")

    @property
    def model_path(self) -> Path:
        return resolve_path(self.model_dir)


@dataclass(frozen=True)
class ToolSettings:
    enabled: bool = True                # the assistant may use tools (clock, memory, home, email)
    # Device types the assistant may switch by itself. Locks, alarms, covers (garage doors) and
    # scripts are left out on purpose: add them here only if you are sure.
    allowed_domains: str = "light,switch,fan,media_player,scene,climate,input_boolean"
    max_rounds: int = 3                 # tool calls per question at most

    def validate(self) -> None:
        if not 1 <= self.max_rounds <= 6:
            raise SettingsError("tools.max_rounds must be between 1 and 6")

    @property
    def domains(self) -> frozenset[str]:
        return frozenset(d.strip().lower() for d in self.allowed_domains.split(",") if d.strip())


@dataclass(frozen=True)
class GpuSettings:
    auto_pause: bool = True             # step aside when a game or another program needs the GPU
    busy_percent: float = 40.0          # another program above this (Task Manager's GPU %) counts as busy
    resume_after_s: float = 60.0        # free for this long before the assistant comes back
    ignore: str = ""                    # programs that never count, e.g. "obs64.exe, chrome.exe"

    def validate(self) -> None:
        if not 5 <= self.busy_percent <= 100:
            raise SettingsError("gpu.busy_percent must be between 5 and 100")
        if not 5 <= self.resume_after_s <= 3600:
            raise SettingsError("gpu.resume_after_s must be between 5 and 3600")

    @property
    def ignored(self) -> set[str]:
        return {n.strip().lower() for n in self.ignore.split(",") if n.strip()}


@dataclass(frozen=True)
class GraphSettings:
    poll_seconds: float = 2.0           # how often the memory store and checkpoints are checked
    max_memories: int = 20000           # most-used memories drawn when the store is larger

    def validate(self) -> None:
        if not 0.25 <= self.poll_seconds <= 60:
            raise SettingsError("graph.poll_seconds must be between 0.25 and 60")
        if not 1 <= self.max_memories <= 200_000:
            raise SettingsError("graph.max_memories must be between 1 and 200000")


SECTIONS: dict[str, type] = {
    "server": ServerSettings,
    "somni": SomniSettings,
    "provider": ProviderSettings,
    "voice": VoiceSettings,
    "listen": ListenSettings,
    "tools": ToolSettings,
    "gpu": GpuSettings,
    "graph": GraphSettings,
}


def _build_section(name: str, cls: type, data: Any):
    if not isinstance(data, dict):
        raise SettingsError(f"{name} must be an object")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise SettingsError(f"unknown setting {name}.{unknown[0]}")
    defaults = cls()
    values = {}
    for key, value in data.items():
        expected = type(getattr(defaults, key))
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if type(value) is not expected:
            raise SettingsError(f"{name}.{key} must be {expected.__name__}")
        values[key] = value
    section = cls(**values)
    section.validate()
    return section


@dataclass(frozen=True)
class StudioSettings:
    server: ServerSettings = field(default_factory=ServerSettings)
    somni: SomniSettings = field(default_factory=SomniSettings)
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    listen: ListenSettings = field(default_factory=ListenSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    gpu: GpuSettings = field(default_factory=GpuSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)

    @classmethod
    def from_dict(cls, data: dict) -> StudioSettings:
        if not isinstance(data, dict):
            raise SettingsError("settings must be a JSON object")
        unknown = sorted(set(data) - set(SECTIONS))
        if unknown:
            raise SettingsError(f"unknown settings section {unknown[0]}")
        return cls(**{name: _build_section(name, sec, data.get(name, {})) for name, sec in SECTIONS.items()})

    def to_dict(self) -> dict:
        return asdict(self)

    def merged(self, patch: dict) -> StudioSettings:
        """A copy with `patch` ({section: {key: value}}) applied and validated."""
        if not isinstance(patch, dict):
            raise SettingsError("a settings update must be a JSON object")
        data = self.to_dict()
        for name, values in patch.items():
            if name not in data:
                raise SettingsError(f"unknown settings section {name}")
            if not isinstance(values, dict):
                raise SettingsError(f"{name} must be an object")
            data[name].update(values)
        return StudioSettings.from_dict(data)


def field_types() -> dict[str, str]:
    """{"section.key": "str" | "int" | "float" | "bool"} for clients that edit settings (JSON
    cannot tell 2.0 from 2)."""
    return {f"{name}.{f.name}": type(getattr(cls(), f.name)).__name__
            for name, cls in SECTIONS.items() for f in fields(cls)}


def settings_path(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(SETTINGS_ENV)
    return Path(env).expanduser() if env else DEFAULT_SETTINGS_PATH


def load_settings(path: Path) -> StudioSettings:
    if not path.exists():
        return StudioSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"{path} is not valid JSON: {exc}") from exc
    return StudioSettings.from_dict(data)


def save_settings(settings: StudioSettings, path: Path) -> None:
    """Atomic write: a crash mid-save never leaves a half-written settings file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
