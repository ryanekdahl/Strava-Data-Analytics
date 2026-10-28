"""Shareable stat stickers: a transparent PNG for an Instagram story.

Strava lets you drop a run onto a story as a sticker -- a few stats and the
route outline in one colour on a transparent background, so the photo shows
through.  :func:`render_sticker` draws the same thing with Pillow: an optional
route trace on top, then each ``(label, value)`` pair stacked underneath, then
a short caption.  Pasting the PNG into an Instagram story makes it a sticker.

The image is drawn at twice its final size and scaled down, which smooths the
edges of the route line and the text (Pillow draws neither anti-aliased).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

WIDTH_PX = 1080          # an Instagram story is 1080 px wide
SUPERSAMPLE = 2
ROUTE_BOX_PX = (720, 520)
ROUTE_POINTS = 1500      # longer tracks are thinned to this many points
PADDING_PX = 48

# Ink colours, as RGBA.  White suits most photos; black for bright ones.
INKS: dict[str, tuple[int, int, int, int]] = {
    "White": (255, 255, 255, 255),
    "Black": (11, 11, 11, 255),
    "Orange": (252, 82, 0, 255),
}

# Bold sans-serif faces tried in order; the first that exists is used, and
# Pillow's built-in font (FreeType Aileron) is the fallback everywhere else.
_BOLD_FONTS = [
    ("/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("C:/Windows/Fonts/arialbd.ttf", 0),
]
_REGULAR_FONTS = [
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("C:/Windows/Fonts/arial.ttf", 0),
]


def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    for path, index in _BOLD_FONTS if bold else _REGULAR_FONTS:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    return ImageFont.load_default(size)


def png_size(png: bytes) -> tuple[int, int]:
    """``(width, height)`` of a PNG, read from its header."""
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def route_points(route: pd.DataFrame, box: tuple[int, int]) -> list[tuple[float, float]]:
    """The route's lat/lon as pixel points fitted, centred, inside a ``box`` of (w, h).

    Uses an equirectangular projection (longitude scaled by the cosine of the
    mean latitude), which is exact enough over the few kilometres of one run.
    Returns ``[]`` when there are fewer than two positions.
    """
    track = route[["lat", "lon"]].dropna()
    if len(track) < 2:
        return []
    if len(track) > ROUTE_POINTS:
        track = track.iloc[np.linspace(0, len(track) - 1, ROUTE_POINTS).astype(int)]
    lat, lon = track["lat"].to_numpy(), track["lon"].to_numpy()
    x = (lon - lon.min()) * np.cos(np.radians(lat.mean()))
    y = lat.max() - lat  # north up
    span = max(x.max(), y.max())
    if span == 0:
        return []
    width, height = box
    scale = min(width / x.max() if x.max() else np.inf, height / y.max() if y.max() else np.inf)
    x, y = x * scale, y * scale
    x += (width - x.max()) / 2
    y += (height - y.max()) / 2
    return list(zip(x.tolist(), y.tolist()))


def render_sticker(stats: list[tuple[str, str]], caption: str = "",
                   route: pd.DataFrame | None = None, ink: str = "White") -> bytes:
    """A transparent PNG with the route (optional), the stats and a caption, as bytes.

    ``stats`` are ``(label, value)`` pairs, drawn top to bottom as a small
    label over a large value.  The image is cropped to its content plus a margin.
    """
    s = SUPERSAMPLE
    colour = INKS[ink]
    label_font, value_font = _font(34 * s, bold=False), _font(92 * s, bold=True)
    caption_font = _font(32 * s, bold=False)
    width = WIDTH_PX * s
    height = (ROUTE_BOX_PX[1] + 220 * max(len(stats), 1) + 200) * s
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    y = PADDING_PX * s
    points = route_points(route, (ROUTE_BOX_PX[0] * s, ROUTE_BOX_PX[1] * s)) \
        if route is not None else []
    if points:
        left = (width - ROUTE_BOX_PX[0] * s) / 2
        draw.line([(px + left, py + y) for px, py in points], fill=colour, width=11 * s,
                  joint="curve")
        y += ROUTE_BOX_PX[1] * s + 64 * s

    def centred(text: str, font, top: float) -> float:
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (box[2] - box[0])) / 2 - box[0], top - box[1]), text,
                  font=font, fill=colour)
        return top + (box[3] - box[1])

    for label, value in stats:
        y = centred(label, label_font, y) + 18 * s
        y = centred(value, value_font, y) + 56 * s
    if caption:
        y = centred(caption, caption_font, y + 8 * s)

    bbox = image.getbbox()
    if bbox is None:  # nothing drawn
        bbox = (0, 0, 1, 1)
    pad = PADDING_PX * s
    image = image.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                        min(bbox[2] + pad, width), min(bbox[3] + pad, height)))
    image = image.resize((image.width // s, image.height // s), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
