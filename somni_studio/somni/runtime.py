"""Importing the Somni framework (`sleepai`) from its checkout.

The studio does not vendor or fork Somni: it imports the user's checkout, so the UI always runs
the same recall, routing and salience code as wake and sleep do.
"""
from __future__ import annotations

import sys
from pathlib import Path


class SomniNotFound(RuntimeError):
    """The configured repo is not a Somni checkout."""


def ensure_sleepai(repo: Path) -> None:
    """Make `import sleepai` resolve to `repo/sleepai`."""
    repo = repo.resolve()
    if not (repo / "sleepai" / "__init__.py").exists():
        raise SomniNotFound(f"no Somni checkout at {repo} (expected {repo / 'sleepai'})")
    loaded = sys.modules.get("sleepai")
    if loaded is not None:
        origin = Path(getattr(loaded, "__file__", "") or "").resolve().parent.parent
        if origin != repo:
            raise SomniNotFound(f"sleepai is already imported from {origin}; restart the studio to switch repos")
        return
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
