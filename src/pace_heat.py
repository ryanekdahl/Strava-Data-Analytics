"""Pace heat map: where on the route you ran faster or slower than usual.

Strava's answer to "how was my pace spread over this run?" is a line chart you
scrub with a finger.  This module turns the run into something you read at a
glance instead: every stretch of the route goes into one of a few **pace
bands** centred on the run's own median pace, so the map can be painted hot
where you pushed and cool where you eased off.

Bands
-----
The middle band is centred on the distance-weighted median pace of the run,
so "neutral" means *your typical pace on this run*, not some absolute number.
Bands are equal-width steps of a whole number of 5-second increments, wide
enough that the 5th-95th percentile of the run spans all of them; the two
outer bands are open-ended so a GPS spike or a traffic light lands in "or
slower" (labelled "over") rather than stretching the scale.  Everything is in the display unit
(min/km or min/mi), so band edges read as round numbers in that unit.

Band 0 is the fastest, band ``n - 1`` the slowest.  Segments without a usable
pace (a stop, a gap in the recording) get band ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.performance import format_pace
from src.units import METRIC, Units

N_BANDS = 7
STEP_MIN = 5.0 / 60.0  # band widths are whole multiples of 5 seconds of pace
NO_PACE = -1

SEGMENT_COLUMNS = ["lat0", "lon0", "lat1", "lon1", "distance_m", "length_m", "pace", "band"]
BAND_COLUMNS = ["band", "low", "high", "label", "share"]
EXTREME_COLUMNS = ["kind", "split", "pace", "lat", "lon"]


@dataclass(frozen=True)
class PaceHeat:
    """A run cut into pace-banded segments.

    ``segments``: one row per pair of consecutive route points, with its
    midpoint ``distance_m``, its ``length_m``, its ``pace`` in display units
    and its ``band``.  ``bands``: each band's ``low``/``high`` edges (display
    pace; the outer ones open-ended as -inf/inf), a ``label`` and the
    ``share`` of the run's distance spent in it.  ``median`` is the run's
    distance-weighted median pace and ``step`` the band width, both in display
    minutes.
    """

    segments: pd.DataFrame
    bands: pd.DataFrame
    median: float
    step: float

    @property
    def empty(self) -> bool:
        return self.segments.empty


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The ``q`` quantile of ``values`` when each counts in proportion to its weight."""
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return float("nan")
    order = np.argsort(values[keep])
    v, w = values[keep][order], weights[keep][order]
    cumulative = np.cumsum(w) - 0.5 * w  # the midpoint of each value's share
    return float(np.interp(q * w.sum(), cumulative, v))


def band_edges(median: float, spread: float, n_bands: int = N_BANDS) -> tuple[np.ndarray, float]:
    """Inner band edges around ``median`` and the band width.

    ``spread`` is how far from the median the run reaches (the larger of the
    5th and 95th percentile gaps).  The width is that range split into
    ``n_bands``, rounded to a whole number of 5-second steps and never under
    one.  Returns the ``n_bands - 1`` inner edges, fastest first, and the width.
    """
    raw = 2.0 * max(spread, 0.0) / n_bands
    step = max(round(raw / STEP_MIN), 1) * STEP_MIN
    offsets = (np.arange(1, n_bands) - n_bands / 2.0) * step
    return median + offsets, step


def _band_label(low: float, high: float) -> str:
    if not np.isfinite(low):
        return f"under {format_pace(high, suffix='')}"
    if not np.isfinite(high):
        return f"over {format_pace(low, suffix='')}"
    return f"{format_pace(low, suffix='')}–{format_pace(high, suffix='')}"


def pace_heat(route: pd.DataFrame, units: Units = METRIC, n_bands: int = N_BANDS) -> PaceHeat:
    """Cut a route from :func:`src.streams.route_for` into pace-banded segments."""
    empty = PaceHeat(pd.DataFrame(columns=SEGMENT_COLUMNS),
                     pd.DataFrame(columns=BAND_COLUMNS), float("nan"), STEP_MIN)
    if len(route) < 2:
        return empty
    lat = route["lat"].to_numpy(dtype="float64")
    lon = route["lon"].to_numpy(dtype="float64")
    distance = route["distance_m"].to_numpy(dtype="float64")
    point_pace = units.pace(route["pace_min_per_km"].to_numpy(dtype="float64"))
    point_pace = np.where(np.isfinite(point_pace) & (point_pace > 0), point_pace, np.nan)

    length = np.diff(distance)
    ends = np.vstack([point_pace[:-1], point_pace[1:]])
    known = np.isfinite(ends).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = np.where(known > 0, np.nansum(ends, axis=0) / known, np.nan)
    usable = np.isfinite(pace) & (length > 0)
    if not usable.any():
        return empty

    median = weighted_quantile(pace, length, 0.5)
    spread = max(median - weighted_quantile(pace, length, 0.05),
                 weighted_quantile(pace, length, 0.95) - median)
    edges, step = band_edges(median, spread, n_bands)
    band = np.where(usable, np.searchsorted(edges, pace, side="right"), NO_PACE)

    segments = pd.DataFrame({
        "lat0": lat[:-1], "lon0": lon[:-1], "lat1": lat[1:], "lon1": lon[1:],
        "distance_m": (distance[:-1] + distance[1:]) / 2.0 - distance[0],
        "length_m": np.maximum(length, 0.0), "pace": pace, "band": band.astype(int),
    }, columns=SEGMENT_COLUMNS)

    lows = np.concatenate([[-np.inf], edges])
    highs = np.concatenate([edges, [np.inf]])
    counted = segments[segments["band"] != NO_PACE]
    by_band = counted.groupby("band")["length_m"].sum()
    total = float(counted["length_m"].sum())
    bands = pd.DataFrame({
        "band": np.arange(n_bands), "low": lows, "high": highs,
        "label": [_band_label(lo, hi) for lo, hi in zip(lows, highs)],
        "share": [float(by_band.get(b, 0.0)) / total if total > 0 else 0.0
                  for b in range(n_bands)],
    }, columns=BAND_COLUMNS)
    return PaceHeat(segments, bands, median, step)


def split_extremes(route: pd.DataFrame, units: Units = METRIC, min_splits: int = 3) -> pd.DataFrame:
    """The fastest and slowest whole split, with the point halfway through each.

    Returns ``DataFrame[kind, split, pace, lat, lon]`` with a ``"fastest"`` and a
    ``"slowest"`` row (``pace`` in display units), or no rows when the run has
    fewer than ``min_splits`` whole kilometres (or miles): picking the "fastest
    of two" says little.
    """
    if len(route) < 2:
        return pd.DataFrame(columns=EXTREME_COLUMNS)
    distance = np.maximum.accumulate(route["distance_m"].to_numpy(dtype="float64"))
    moving = np.maximum.accumulate(route["moving_s"].to_numpy(dtype="float64"))
    count = int((distance[-1] - distance[0]) // units.split_m)
    if count < max(min_splits, 2):
        return pd.DataFrame(columns=EXTREME_COLUMNS)
    marks = distance[0] + units.split_m * np.arange(0, count + 1)
    split_s = np.diff(np.interp(marks, distance, moving))
    pace = split_s / 60.0  # minutes per display split = display pace
    middle = marks[:-1] + units.split_m / 2.0
    rows = []
    for kind, i in (("fastest", int(np.argmin(pace))), ("slowest", int(np.argmax(pace)))):
        rows.append({"kind": kind, "split": i + 1, "pace": float(pace[i]),
                     "lat": float(np.interp(middle[i], distance, route["lat"])),
                     "lon": float(np.interp(middle[i], distance, route["lon"]))})
    return pd.DataFrame(rows, columns=EXTREME_COLUMNS)


def near_median_share(heat: PaceHeat) -> float:
    """Share of the run's distance in the middle band, at about the median pace."""
    if heat.bands.empty:
        return float("nan")
    return float(heat.bands["share"].iloc[len(heat.bands) // 2])
