"""Tests for the shareable stat sticker."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

from src.sticker import ROUTE_BOX_PX, png_size, render_sticker, route_points

STATS = [("Distance", "10.02 km"), ("Pace", "5:12 /km"), ("Time", "52:08")]


def _route(n: int = 200) -> pd.DataFrame:
    angle = np.linspace(0, 2 * np.pi, n)
    return pd.DataFrame({"lat": 51.5 + 0.01 * np.sin(angle), "lon": -0.1 + 0.02 * np.cos(angle)})


def test_sticker_is_a_transparent_png():
    png = render_sticker(STATS, "Morning Run")
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG" and image.mode == "RGBA"
    alpha = np.asarray(image)[..., 3]
    assert alpha[0, 0] == 0          # the corner is see-through
    assert alpha.max() == 255        # and something opaque is drawn
    assert png_size(png) == image.size


def test_ink_colour_is_used():
    pixels = np.asarray(Image.open(io.BytesIO(render_sticker(STATS, ink="Black"))))
    opaque = pixels[pixels[..., 3] == 255]
    assert opaque[..., :3].max() < 40


def test_route_makes_the_sticker_taller():
    plain = png_size(render_sticker(STATS))
    with_route = png_size(render_sticker(STATS, route=_route()))
    assert with_route[1] > plain[1] + ROUTE_BOX_PX[1] / 2


def test_route_points_fit_the_box_and_keep_north_up():
    points = route_points(_route(), (400, 300))
    xs, ys = zip(*points)
    assert min(xs) >= -1e-9 and max(xs) <= 400 + 1e-9
    assert min(ys) >= -1e-9 and max(ys) <= 300 + 1e-9
    # The first point is the eastmost and mid-latitude; the quarter point is northmost (top).
    assert ys[50] == min(ys)


def test_route_points_handle_missing_or_degenerate_tracks():
    assert route_points(pd.DataFrame({"lat": [], "lon": []}), (100, 100)) == []
    assert route_points(pd.DataFrame({"lat": [1.0, 1.0], "lon": [2.0, 2.0]}), (100, 100)) == []
    long = _route(10_000)
    assert len(route_points(long, (100, 100))) <= 1500
