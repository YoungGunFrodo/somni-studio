"""Download a Whisper speech-recognition model (faster-whisper, CTranslate2 int8) into data/models/stt/.

    python scripts/fetch_listen.py              # base.en, about 145 MB (recommended)
    python scripts/fetch_listen.py small.en     # about 465 MB, more accurate, 3x slower
    python scripts/fetch_listen.py tiny.en      # about 75 MB, fastest

Models come from the Systran faster-whisper repositories on Hugging Face at pinned revisions, so
the same files are installed every time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "models" / "stt"
REVISIONS = {
    "tiny.en": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
    "base.en": "3d3d5dee26484f91867d81cb899cfcf72b96be6c",
    "small.en": "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
}


def fetch(name: str) -> Path:
    from huggingface_hub import snapshot_download
    target = DEST / name
    if (target / "model.bin").is_file():
        print(f"{name}: present")
        return target
    path = snapshot_download(f"Systran/faster-whisper-{name}", revision=REVISIONS[name], local_dir=str(target),
                             allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*",
                                             "preprocessor_config.json"])
    print(f"{name}: installed in {path}")
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=["base.en"], choices=sorted(REVISIONS))
    for name in parser.parse_args(argv).models:
        fetch(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
