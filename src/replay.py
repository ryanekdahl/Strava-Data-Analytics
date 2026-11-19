"""Run replay: the runner's position, pace and split marks at each moment of a run.

A replay is a sequence of *frames* evenly spaced on the run's moving clock, not
on distance.  So when it is played back at a steady frame rate, the runner dot
moves across the map at a speed proportional to how fast the run actually was:
a fast stretch whips by, a slow climb crawls.  Stopped time is already left out
of ``moving_s``, so a traffic light does not freeze the replay.

Everything here works from the thinned route that :func:`src.streams.route_for`
returns (``lat, lon, distance_m, moving_s, hr, pace_min_per_km`` per point) and
interpolates linearly between its points.

Milestones are the points where the run crosses each whole split (every
kilometre, or every mile), with the moving time at which it got there and the
time that split took.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REPLAY_FRAMES = 150  # enough for a smooth dot, few enough to keep the page light

FRAME_COLUMNS = ["moving_s", "distance_m", "lat", "lon", "pace_min_per_km", "hr", "trail"]
MILESTONE_COLUMNS = ["split", "distance_m", "moving_s", "lat", "lon", "split_s",
                     "pace_min_per_km"]


def _clock(route: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance and moving time along the route, made non-decreasing for interpolation."""
    distance = np.maximum.accumulate(route["distance_m"].to_numpy(dtype="float64"))
    moving = np.maximum.accumulate(route["moving_s"].to_numpy(dtype="float64"))
    return distance, moving


def _at_distance(route: pd.DataFrame, distance: np.ndarray, targets: np.ndarray,
                 column: str) -> np.ndarray:
    """A route column interpolated at ``targets`` metres, NaN where it has no data."""
    values = route[column].to_numpy(dtype="float64")
    known = np.isfinite(values)
    if known.sum() < 2:
        return np.full(len(targets), np.nan)
    return np.interp(targets, distance[known], values[known])


def replay_frames(route: pd.DataFrame, frames: int = REPLAY_FRAMES) -> pd.DataFrame:
    """Where the runner was at ``frames`` evenly spaced moments of moving time.

    Returns ``DataFrame[moving_s, distance_m, lat, lon, pace_min_per_km, hr,
    trail]``, where ``trail`` is how many route points had been passed by then
    (the slice of the route to draw as "already run").  Empty when the route
    has fewer than two points or no moving time.
    """
    if len(route) < 2:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    distance, moving = _clock(route)
    if moving[-1] <= moving[0] or distance[-1] <= distance[0]:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    clock = np.linspace(moving[0], moving[-1], max(int(frames), 2))
    at = np.interp(clock, moving, distance)
    out = pd.DataFrame({"moving_s": clock - moving[0], "distance_m": at})
    for column in ("lat", "lon", "pace_min_per_km", "hr"):
        out[column] = _at_distance(route, distance, at, column)
    out["trail"] = np.searchsorted(distance, at, side="right")
    return out[FRAME_COLUMNS]


def milestones(route: pd.DataFrame, split_m: float = 1000.0) -> pd.DataFrame:
    """Each whole split the run passed: where, when, and how long that split took.

    Returns ``DataFrame[split, distance_m, moving_s, lat, lon, split_s,
    pace_min_per_km]`` with one row per completed kilometre (or mile, with
    ``split_m`` = 1609.344); ``moving_s`` is counted from the start of the run.
    """
    if len(route) < 2 or split_m <= 0:
        return pd.DataFrame(columns=MILESTONE_COLUMNS)
    distance, moving = _clock(route)
    count = int((distance[-1] - distance[0]) // split_m)
    if count < 1:
        return pd.DataFrame(columns=MILESTONE_COLUMNS)
    marks = distance[0] + split_m * np.arange(1, count + 1)
    at = np.interp(marks, distance, moving) - moving[0]
    split_s = np.diff(np.concatenate([[0.0], at]))
    return pd.DataFrame({
        "split": np.arange(1, count + 1),
        "distance_m": marks - distance[0],
        "moving_s": at,
        "lat": _at_distance(route, distance, marks, "lat"),
        "lon": _at_distance(route, distance, marks, "lon"),
        "split_s": split_s,
        "pace_min_per_km": split_s / 60.0 / (split_m / 1000.0),
    }, columns=MILESTONE_COLUMNS)
