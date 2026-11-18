"""Where this computer is, from its operating system.

Windows: the location service (Wi-Fi positioning, about 100 m), when Settings → Privacy → Location
lets apps use it. The town is named with OpenStreetMap's Nominatim; only coordinates rounded to
about 1 km are sent. Other systems: unknown (the assistant then asks).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.parse

log = logging.getLogger(__name__)

CACHE_S = 30 * 60
ROUND = 2                        # decimals of latitude/longitude sent anywhere: about 1 km

# WinRT's Geolocator from Windows PowerShell 5.1 (ships with Windows; no extra install).
WINDOWS_SCRIPT = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Devices.Geolocation.Geolocator, Windows.Devices.Geolocation, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and
  $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$access = $asTask.MakeGenericMethod([Windows.Devices.Geolocation.GeolocationAccessStatus]).Invoke($null,
  @([Windows.Devices.Geolocation.Geolocator]::RequestAccessAsync()))
if (-not $access.Wait(10000) -or $access.Result -ne 'Allowed') { '{"error": "location is off for apps"}'; exit }
$pos = $asTask.MakeGenericMethod([Windows.Devices.Geolocation.Geoposition]).Invoke($null,
  @((New-Object Windows.Devices.Geolocation.Geolocator).GetGeopositionAsync()))
if (-not $pos.Wait(15000)) { '{"error": "no position"}'; exit }
$c = $pos.Result.Coordinate
@{latitude = $c.Point.Position.Latitude; longitude = $c.Point.Position.Longitude; accuracy = $c.Accuracy;
  source = "$($c.PositionSource)"} | ConvertTo-Json -Compress
"""

_cache: dict = {"at": 0.0, "place": ""}


def computer_location() -> tuple[float, float] | None:
    """This PC's position from the operating system, or None (other systems, location switched off)."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_SCRIPT],
                             capture_output=True, text=True, timeout=40, check=False).stdout.strip()
        data = json.loads(out.splitlines()[-1]) if out else {}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        log.info("no location from Windows: %s", exc)
        return None
    if "latitude" not in data:
        log.info("no location from Windows: %s", data.get("error"))
        return None
    return round(float(data["latitude"]), ROUND), round(float(data["longitude"]), ROUND)


def town_name(latitude: float, longitude: float) -> str:
    """'Portland, Oregon' for a position (OpenStreetMap Nominatim; rounded coordinates only)."""
    from .integrations.web import _fetch
    q = urllib.parse.urlencode({"format": "jsonv2", "lat": round(latitude, ROUND), "lon": round(longitude, ROUND),
                                "zoom": 10, "addressdetails": 1})
    raw, _ = _fetch(f"https://nominatim.openstreetmap.org/reverse?{q}")
    address = json.loads(raw).get("address", {})
    town = address.get("city") or address.get("town") or address.get("village") or address.get("county") or ""
    region = address.get("state") or address.get("country") or ""
    return ", ".join(x for x in (town, region) if x)


def computer_place() -> str:
    """The town this computer is in ("Portland, Oregon"), or "" when the system will not say."""
    if _cache["place"] and time.monotonic() - _cache["at"] < CACHE_S:
        return _cache["place"]
    spot = computer_location()
    if spot is None:
        return ""
    try:
        name = town_name(*spot)
    except Exception as exc:                  # coordinates still work for the weather
        log.info("could not name %s: %s", spot, exc)
        name = ""
    place = name or f"{spot[0]}, {spot[1]}"
    _cache.update(at=time.monotonic(), place=place)
    return place
