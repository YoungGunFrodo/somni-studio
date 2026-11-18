"""Download the text-to-speech model (Kokoro v1.0, Apache-2.0) into data/models/tts/.

    python scripts/fetch_voice.py            # about 354 MB, once
    python scripts/fetch_voice.py --force    # download again

Files come from the kokoro-onnx project's GitHub release and are checked against pinned SHA-256
hashes before they are kept. The full-precision model is used on purpose: on desktop CPUs the
int8 build measured about 10x slower.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "models" / "tts"
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
FILES = {
    "kokoro-v1.0.onnx": "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, expected: str, force: bool) -> None:
    target = DEST / name
    if target.exists() and not force and _sha256(target) == expected:
        print(f"{name}: present")
        return
    url = RELEASE + name
    if not url.startswith("https://github.com/"):
        raise SystemExit(f"refusing to download from {url}")
    part = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "somni-studio"})  # noqa: S310 - https checked
    with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as out:  # noqa: S310 - https checked
        total, done = int(r.headers.get("Content-Length") or 0), 0
        for block in iter(lambda: r.read(CHUNK), b""):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r{name}: {done * 100 // total:3d}% of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    actual = _sha256(part)
    if actual != expected:
        part.unlink(missing_ok=True)
        raise SystemExit(f"{name} failed its checksum ({actual}); nothing was installed")
    part.replace(target)
    print(f"{name}: installed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    DEST.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        fetch(name, digest, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
