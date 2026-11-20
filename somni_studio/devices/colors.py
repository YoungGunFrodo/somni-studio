"""Colors in words: "red", "teal", "warm white", "#ff8800", "2700K".

A light either shows a color (a point on the CIE 1931 xy chart, which is what Hue and most bulbs
take) or a shade of white (a color temperature in kelvin). parse() turns what the user said into
one of those; describe() turns what a bulb reports back into a word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NAMED_RGB = {
    "red": (255, 0, 0), "orange": (255, 120, 0), "amber": (255, 170, 0), "yellow": (255, 230, 0),
    "lime": (160, 255, 0), "green": (0, 255, 0), "mint": (90, 255, 170), "teal": (0, 200, 170),
    "cyan": (0, 255, 255), "turquoise": (60, 225, 210), "sky blue": (100, 190, 255), "blue": (0, 0, 255),
    "indigo": (75, 0, 200), "purple": (140, 0, 255), "violet": (170, 80, 255), "lavender": (190, 150, 255),
    "magenta": (255, 0, 255), "pink": (255, 100, 180), "hot pink": (255, 30, 140), "rose": (255, 60, 110),
    "coral": (255, 110, 80), "gold": (255, 200, 40), "crimson": (220, 20, 60),
}
NAMED_KELVIN = {
    "candlelight": 2000, "candle": 2000, "warm white": 2700, "warm": 2700, "soft white": 2700,
    "white": 4000, "neutral white": 4000, "neutral": 4000, "cool white": 5000, "cool": 5000,
    "daylight": 6500, "bright white": 6500, "cold white": 6500,
}
HEX = re.compile(r"^#?([0-9a-f]{6})$", re.I)
KELVIN = re.compile(r"^(\d{4,5})\s*k(elvin)?$", re.I)


@dataclass(frozen=True)
class Color:
    """Either xy (a color) or kelvin (a white); name is what to call it."""
    name: str
    xy: tuple[float, float] | None = None
    kelvin: int | None = None
    rgb: tuple[int, int, int] | None = None

    @property
    def is_white(self) -> bool:
        return self.kelvin is not None


def known_names() -> list[str]:
    return sorted(NAMED_RGB) + sorted(NAMED_KELVIN)


def parse(text: str) -> Color:
    """'red' / 'warm white' / '#ff8800' / '2700K' -> Color. Raises ValueError for anything else."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower()).removesuffix(" color").removesuffix(" colour")
    if t in NAMED_KELVIN:
        return Color(t, kelvin=NAMED_KELVIN[t])
    if t in NAMED_RGB:
        return Color(t, xy=rgb_to_xy(*NAMED_RGB[t]), rgb=NAMED_RGB[t])
    m = KELVIN.match(t)
    if m:
        kelvin = max(1500, min(9000, int(m.group(1))))
        return Color(f"{kelvin}K white", kelvin=kelvin)
    m = HEX.match(t)
    if m:
        rgb = tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        return Color("#" + m.group(1).lower(), xy=rgb_to_xy(*rgb), rgb=rgb)
    raise ValueError(f"'{text}' is not a color I know; try a name like red, teal or warm white, "
                     "a hex code like #ff8800, or a white like 2700K")


def rgb_to_xy(r: int, g: int, b: int) -> tuple[float, float]:
    """sRGB (0-255) -> CIE xy, the wide-gamut conversion Philips documents for Hue."""
    def linear(c: float) -> float:
        c /= 255
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    rl, gl, bl = linear(r), linear(g), linear(b)
    x = rl * 0.664511 + gl * 0.154324 + bl * 0.162028
    y = rl * 0.283881 + gl * 0.668433 + bl * 0.047685
    z = rl * 0.000088 + gl * 0.072310 + bl * 0.986039
    total = x + y + z
    if total == 0:
        return (0.3127, 0.3290)                      # black: call it white
    return (round(x / total, 4), round(y / total, 4))


def describe_xy(x: float, y: float) -> str:
    """The nearest named color to a point a bulb reports."""
    return min(NAMED_RGB, key=lambda n: (rgb_to_xy(*NAMED_RGB[n])[0] - x) ** 2
               + (rgb_to_xy(*NAMED_RGB[n])[1] - y) ** 2)


def describe_kelvin(kelvin: int) -> str:
    if kelvin < 2400:
        return f"candlelight ({kelvin}K)"
    if kelvin < 3300:
        return f"warm white ({kelvin}K)"
    if kelvin < 4600:
        return f"neutral white ({kelvin}K)"
    if kelvin < 5800:
        return f"cool white ({kelvin}K)"
    return f"daylight ({kelvin}K)"


def mireds(kelvin: int) -> int:
    return round(1_000_000 / max(1000, kelvin))
