"""Running Home Assistant in a VirtualBox VM as part of Somni Studio.

Home Assistant needs Linux, so on a Windows (or Mac) PC it runs as Home Assistant OS in a virtual
machine. Somni Studio starts that VM in the background when it starts, so it feels built in: no
VirtualBox window, no extra step. The VM is found by name (the Home Assistant card's "VirtualBox
VM" setting, or any VM whose name contains "Home Assistant").
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

TIMEOUT_S = 30
NAME_HINT = "home assistant"


def vboxmanage() -> str | None:
    found = shutil.which("VBoxManage")
    if found:
        return found
    for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), "/Applications"):
        if not root:
            continue
        for candidate in (Path(root) / "Oracle" / "VirtualBox" / "VBoxManage.exe",
                          Path(root) / "VirtualBox.app" / "Contents" / "MacOS" / "VBoxManage"):
            if candidate.is_file():
                return str(candidate)
    return None


def _run(*args: str) -> str:
    tool = vboxmanage()
    if tool is None:
        raise RuntimeError("VirtualBox is not installed")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    out = subprocess.run([tool, *args], capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
                         creationflags=flags)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout).strip().splitlines()[-1] if (out.stderr or out.stdout)
                           else f"VBoxManage {args[0]} failed")
    return out.stdout


def find_vm(name: str = "") -> str | None:
    """The VM's name: the one asked for, or one that looks like Home Assistant."""
    if vboxmanage() is None:
        return None
    names = [line.split('"')[1] for line in _run("list", "vms").splitlines() if line.count('"') >= 2]
    if name:
        return name if name in names else None
    return next((n for n in names if NAME_HINT in n.lower()), None)


def state(name: str) -> str:
    """running | poweroff | saved | aborted | starting | stopping | ... (VirtualBox's own words)."""
    for line in _run("showvminfo", name, "--machinereadable").splitlines():
        if line.startswith("VMState="):
            return line.split("=", 1)[1].strip('"')
    return "unknown"


def start(name: str) -> None:
    if state(name) != "running":
        _run("startvm", name, "--type", "headless")


def stop(name: str) -> None:
    """A clean shutdown (like pressing the power button): Home Assistant closes properly."""
    if state(name) == "running":
        _run("controlvm", name, "acpipowerbutton")


def status(name: str = "") -> dict:
    try:
        vm = find_vm(name)
        if vm is None:
            return {"vm": None, "state": "none", "virtualbox": vboxmanage() is not None}
        return {"vm": vm, "state": state(vm), "virtualbox": True}
    except Exception as exc:                       # VirtualBox busy or broken: say so, never crash
        return {"vm": name or None, "state": "unknown", "virtualbox": vboxmanage() is not None, "error": str(exc)}
