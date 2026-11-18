"""Fetch the browser libraries the studio serves locally (no CDN at runtime, works offline).

Downloads the pinned three.js release from the npm registry, verifies the tarball against the
registry's sha512 integrity value, and extracts only the files the UI imports into
web/vendor/three/. Safe to re-run: it does nothing when the pinned version is already present.

    python scripts/vendor.py            # install if missing
    python scripts/vendor.py --force    # re-download
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

THREE_VERSION = "0.169.0"
REGISTRY = "https://registry.npmjs.org/three/" + THREE_VERSION

# Every file the UI imports, plus everything those files import (checked after extraction).
THREE_FILES = (
    "LICENSE",
    "build/three.module.js",
    "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/renderers/CSS2DRenderer.js",
    "examples/jsm/postprocessing/EffectComposer.js",
    "examples/jsm/postprocessing/MaskPass.js",
    "examples/jsm/postprocessing/OutputPass.js",
    "examples/jsm/postprocessing/Pass.js",
    "examples/jsm/postprocessing/RenderPass.js",
    "examples/jsm/postprocessing/ShaderPass.js",
    "examples/jsm/postprocessing/UnrealBloomPass.js",
    "examples/jsm/shaders/CopyShader.js",
    "examples/jsm/shaders/LuminosityHighPassShader.js",
    "examples/jsm/shaders/OutputShader.js",
)

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "web" / "vendor" / "three"
STAMP = DEST / "VERSION"


def _fetch(url: str) -> bytes:
    if not url.startswith("https://registry.npmjs.org/"):
        raise SystemExit(f"refusing to download from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "somni-studio-vendor"})  # noqa: S310 - checked above
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return r.read()


def _verify(data: bytes, integrity: str) -> None:
    algo, _, expected = integrity.partition("-")
    if algo != "sha512":
        raise SystemExit(f"unexpected integrity algorithm: {algo}")
    actual = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if actual != expected:
        raise SystemExit("three.js tarball failed its integrity check; nothing was installed")


def _check_imports() -> None:
    """Every relative import inside the extracted files must resolve to an extracted file."""
    import re
    missing = []
    for path in DEST.rglob("*.js"):
        for spec in re.findall(r"""from\s+['"](\.{1,2}/[^'"]+)['"]""", path.read_text(encoding="utf-8")):
            if not (path.parent / spec).resolve().exists():
                missing.append(f"{path.relative_to(DEST)} -> {spec}")
    if missing:
        raise SystemExit("vendored files import files that were not extracted:\n  " + "\n  ".join(missing))


def install(force: bool = False) -> None:
    if not force and STAMP.exists() and STAMP.read_text(encoding="utf-8").strip() == THREE_VERSION:
        print(f"three.js {THREE_VERSION} already vendored")
        return
    meta = json.loads(_fetch(REGISTRY))["dist"]
    data = _fetch(meta["tarball"])
    _verify(data, meta["integrity"])
    if DEST.exists():
        shutil.rmtree(DEST)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for rel in THREE_FILES:
            member = tar.getmember("package/" + rel)
            src = tar.extractfile(member)
            if src is None:
                raise SystemExit(f"not a file in the tarball: {rel}")
            out = DEST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(src.read())
    _check_imports()
    STAMP.write_text(THREE_VERSION + "\n", encoding="utf-8")
    print(f"vendored three.js {THREE_VERSION} into {DEST}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    install(parser.parse_args(argv).force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
