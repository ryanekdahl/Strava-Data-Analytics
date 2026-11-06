"""Climbs and descents: how the ground under a run moved its pace and heart rate.

A run's route (:func:`src.streams.route_for`, 300 points spaced evenly by
distance) is cut into **climbs**, **descents** and **flat** stretches at the
turning points of its elevation profile.

Turning points with hysteresis
------------------------------
Altitude is first smoothed over :data:`SMOOTH_POINTS` route points.  Walking
along the run, the lowest point so far becomes the foot of a climb once the
road has risen :data:`MIN_CHANGE_M` above it, and the highest point so far
becomes the top once it has fallen :data:`MIN_CHANGE_M` below.  Wobbles smaller
than that never start or end a climb, so GPS altitude noise and kerbs do not
chop a hill into pieces.  What is left over at either end of the run, rising or
falling less than :data:`MIN_CHANGE_M`, is flat.

Turning points sit at the lowest and highest points, so a climb found this way
would also swallow the flat road leading up to it.  Each climb and descent is
therefore trimmed at both ends to where the road has actually started moving --
within :data:`SHOULDER_FRACTION` of its rise (at most :data:`SHOULDER_MAX_M`) of
its start and end height -- and the trimmed shoulders count as flat.

Per stretch
-----------
Distance, moving time, pace, grade-adjusted pace (moving time over equivalent
flat distance, with :func:`src.streams.grade_cost`), time-weighted average
heart rate, and the heart rate at the start and end of the stretch.  Heart
rate trails effort by half a minute or so, so start and end are read
:data:`HR_LAG_S` seconds later -- otherwise a climb's rise in heart rate shows
up in the descent after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.streams import grade_cost

MIN_CHANGE_M = 5.0  # a climb or descent must rise or fall at least this much
SMOOTH_POINTS = 5  # altitude is smoothed over this many route points
SHOULDER_FRACTION = 0.02  # flat shoulders are trimmed off climbs and descents...
SHOULDER_MAX_M = 0.5  # ...up to this far from the start and end height
HR_LAG_S = 30.0  # heart rate is read this long after a stretch starts and ends
KINDS = ("climb", "flat", "descent")

SEGMENT_COLUMNS = [
    "segment", "kind", "start_m", "end_m", "distance_m", "time_s", "change_m", "grade",
    "pace_min_per_km", "gap_pace_min_per_km", "flat_distance_m", "avg_hr",
    "hr_start", "hr_end",
]


def smoothed_altitude(route: pd.DataFrame) -> np.ndarray:
    """Route altitude smoothed over :data:`SMOOTH_POINTS` points (NaN gaps filled)."""
    altitude = route["altitude_m"].astype("float64").interpolate(limit_direction="both")
    return altitude.rolling(SMOOTH_POINTS, center=True, min_periods=1).mean().to_numpy()


def turning_points(altitude: np.ndarray, min_change_m: float = MIN_CHANGE_M) -> list[int]:
    """Indices where the profile turns, always including the first and last point."""
    n = len(altitude)
    if n < 2:
        return list(range(n))
    turns = [0]
    direction = 0
    lo = hi = cand = 0
    for i in range(1, n):
        a = altitude[i]
        if direction == 0:
            if a < altitude[lo]:
                lo = i
            if a > altitude[hi]:
                hi = i
            if a - altitude[lo] >= min_change_m:
                direction, cand = 1, i
                if lo != turns[-1]:
                    turns.append(lo)
            elif altitude[hi] - a >= min_change_m:
                direction, cand = -1, i
                if hi != turns[-1]:
                    turns.append(hi)
        elif direction == 1:
            if a > altitude[cand]:
                cand = i
            elif altitude[cand] - a >= min_change_m:
                turns.append(cand)
                direction, cand = -1, i
        else:
            if a < altitude[cand]:
                cand = i
            elif a - altitude[cand] >= min_change_m:
                turns.append(cand)
                direction, cand = 1, i
    if direction != 0 and cand != turns[-1]:
        turns.append(cand)
    if turns[-1] != n - 1:
        turns.append(n - 1)
    return turns


def _pieces(altitude: np.ndarray, turns: list[int], min_change_m: float
            ) -> list[tuple[str, int, int]]:
    """``(kind, start index, end index)`` per stretch, shoulders trimmed, flats merged."""
    raw = []
    for a, b in zip(turns[:-1], turns[1:]):
        change = altitude[b] - altitude[a]
        if abs(change) < min_change_m:
            raw.append(("flat", a, b))
            continue
        sign = 1.0 if change > 0 else -1.0
        tol = min(SHOULDER_MAX_M, SHOULDER_FRACTION * abs(change))
        start, end = a, b
        while start < end - 1 and sign * (altitude[start + 1] - altitude[a]) < tol:
            start += 1
        while end > start + 1 and sign * (altitude[b] - altitude[end - 1]) < tol:
            end -= 1
        raw += [("flat", a, start), ("climb" if sign > 0 else "descent", start, end),
                ("flat", end, b)]
    pieces: list[tuple[str, int, int]] = []
    for kind, a, b in raw:
        if b <= a:
            continue
        if pieces and pieces[-1][0] == kind == "flat":
            pieces[-1] = (kind, pieces[-1][1], b)
        else:
            pieces.append((kind, a, b))
    return pieces


def _hr_at(moving_s: np.ndarray, hr: np.ndarray, at: float) -> float:
    valid = ~np.isnan(hr)
    if valid.sum() < 2:
        return np.nan
    return float(np.interp(min(at, moving_s[valid][-1]), moving_s[valid], hr[valid]))


def segments(route: pd.DataFrame, min_change_m: float = MIN_CHANGE_M) -> pd.DataFrame:
    """The run cut into climbs, descents and flat stretches (see the module docstring).

    ``route`` needs ``distance_m``, ``moving_s``, ``hr``, ``altitude_m`` and
    ``grade`` (as :func:`src.streams.route_for` returns).  Empty when the
    route has no altitude.  ``change_m`` is the smoothed rise (negative for a
    drop); ``grade`` is the stretch's average, as a fraction.
    """
    if len(route) < 2 or route["altitude_m"].notna().sum() < 2:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    altitude = smoothed_altitude(route)
    d = route["distance_m"].to_numpy(dtype="float64")
    t = route["moving_s"].to_numpy(dtype="float64")
    hr = route["hr"].to_numpy(dtype="float64")
    grade = route["grade"].fillna(0.0).to_numpy(dtype="float64")
    step_d, step_t = np.diff(d), np.diff(t)
    step_flat = step_d * grade_cost((grade[:-1] + grade[1:]) / 2.0)
    step_hr = (hr[:-1] + hr[1:]) / 2.0

    pieces = _pieces(altitude, turning_points(altitude, min_change_m), min_change_m)

    rows = []
    for number, (kind, a, b) in enumerate(pieces, start=1):
        distance, seconds = d[b] - d[a], t[b] - t[a]
        flat = float(step_flat[a:b].sum())
        hr_mask = ~np.isnan(step_hr[a:b]) & (step_t[a:b] > 0)
        rows.append({
            "segment": number, "kind": kind, "start_m": d[a], "end_m": d[b],
            "distance_m": distance, "time_s": seconds,
            "change_m": float(altitude[b] - altitude[a]),
            "grade": float((altitude[b] - altitude[a]) / distance) if distance > 0 else np.nan,
            "pace_min_per_km": seconds / 60.0 / (distance / 1000.0) if distance > 0 else np.nan,
            "gap_pace_min_per_km": seconds / 60.0 / (flat / 1000.0) if flat > 0 else np.nan,
            "flat_distance_m": flat,
            "avg_hr": (float(np.average(step_hr[a:b][hr_mask], weights=step_t[a:b][hr_mask]))
                       if hr_mask.any() else np.nan),
            "hr_start": _hr_at(t, hr, t[a] + HR_LAG_S),
            "hr_end": _hr_at(t, hr, t[b] + HR_LAG_S),
        })
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def terrain_summary(segs: pd.DataFrame) -> pd.DataFrame:
    """Climbs, flat and descents totalled: one row per kind present, in that order.

    Returns ``DataFrame[kind, count, distance_m, time_s, change_m, pace_min_per_km,
    gap_pace_min_per_km, avg_hr, time_share]``; heart rate is weighted by time.
    """
    columns = ["kind", "count", "distance_m", "time_s", "change_m", "pace_min_per_km",
               "gap_pace_min_per_km", "avg_hr", "time_share"]
    if segs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    total_time = float(segs["time_s"].sum())
    for kind in KINDS:
        part = segs[segs["kind"] == kind]
        if part.empty:
            continue
        distance, seconds = float(part["distance_m"].sum()), float(part["time_s"].sum())
        flat = float(part["flat_distance_m"].sum())
        with_hr = part.dropna(subset=["avg_hr"])
        rows.append({
            "kind": kind, "count": len(part), "distance_m": distance, "time_s": seconds,
            "change_m": float(part["change_m"].sum()),
            "pace_min_per_km": seconds / 60.0 / (distance / 1000.0) if distance > 0 else np.nan,
            "gap_pace_min_per_km": seconds / 60.0 / (flat / 1000.0) if flat > 0 else np.nan,
            "avg_hr": (float(np.average(with_hr["avg_hr"], weights=with_hr["time_s"]))
                       if not with_hr.empty and with_hr["time_s"].sum() > 0 else np.nan),
            "time_share": seconds / total_time if total_time > 0 else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)
