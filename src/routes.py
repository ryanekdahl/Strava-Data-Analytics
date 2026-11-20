"""Most frequent routes: which runs went the same way, and the fastest on each.

Strava only compares runs on *segments* someone has drawn.  Here the routes are
found from the GPS tracks themselves.

Matching two runs
-----------------
Each track is resampled to :data:`SIGNATURE_POINTS` points evenly spaced by
distance, in metres on a local flat projection.  Two runs are on the same
route when all of these hold:

* their distances differ by at most :data:`MAX_LENGTH_DIFF` (8%);
* their starts are within :data:`MAX_ENDPOINT_M` of each other, and so are
  their finishes;
* every point of each track is, on average, within :data:`MAX_MEAN_GAP_M` of
  the other track, and 90% of points are within :data:`MAX_P90_GAP_M`.  So a
  shared route with a detour of a few hundred metres still fails, but GPS
  wobble and running the other side of the road do not.

The nearest-point test does not care about direction, so a loop run the other
way round counts as the same route (same start, finish and distance).

Grouping
--------
Matching is not transitive (A ~ B and B ~ C does not make A ~ C), so runs are
not chained together.  Instead the run that matches the most others becomes a
route's reference, it and every run matching it form the route, and the step
repeats on what is left.  Each route's runs all match its reference directly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

SIGNATURE_POINTS = 60
MAX_LENGTH_DIFF = 0.08
MAX_ENDPOINT_M = 250.0
MAX_MEAN_GAP_M = 60.0
MAX_P90_GAP_M = 150.0
LOOP_GAP_M = 250.0      # start and finish this close: a loop
RETRACE_GAP_M = 60.0    # way back this close to the way out: an out-and-back
MIN_RUNS = 3

GROUP_COLUMNS = ["activity_id", "route_id", "is_reference"]
ROUTE_COLUMNS = ["route_id", "name", "shape", "runs", "distance_km", "reference_id",
                 "pr_id", "pr_date", "pr_time_s", "pr_pace_min_per_km", "median_pace_min_per_km",
                 "last_date"]
EFFORT_COLUMNS = ["activity_id", "date", "name", "distance_km", "moving_time_s",
                  "pace_min_per_km", "is_pr", "best_so_far"]
_EARTH_M_PER_DEG = 6_371_008.8 * np.pi / 180.0

# Strava's default titles say when, not where, so they never name a route.
GENERIC_NAMES = {f"{when} {sport}" for when in ("Morning", "Lunch", "Afternoon", "Evening",
                                                  "Night") for sport in ("Run", "Trail Run")}


def signature(track: pd.DataFrame, lat0: float, points: int = SIGNATURE_POINTS) -> np.ndarray:
    """``points`` x 2 east/north metres along the track, evenly spaced by distance."""
    distance = np.maximum.accumulate(track["distance_m"].to_numpy(dtype="float64"))
    lat = track["lat"].to_numpy(dtype="float64")
    lon = track["lon"].to_numpy(dtype="float64")
    targets = np.linspace(distance[0], distance[-1], points)
    east = np.interp(targets, distance, lon) * _EARTH_M_PER_DEG * np.cos(np.radians(lat0))
    north = np.interp(targets, distance, lat) * _EARTH_M_PER_DEG
    return np.column_stack([east, north])


def track_gaps(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point of ``a``, its distance to the nearest point of ``b``, and vice versa."""
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return d.min(axis=1), d.min(axis=0)


def same_route(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two signatures trace the same route (see the module docstring)."""
    ab, ba = track_gaps(a, b)
    return all(gaps.mean() <= MAX_MEAN_GAP_M and np.quantile(gaps, 0.9) <= MAX_P90_GAP_M
               for gaps in (ab, ba))


def _signatures(tracks: dict[int, pd.DataFrame]) -> tuple[list[int], np.ndarray, np.ndarray]:
    ids = sorted(i for i, t in tracks.items() if len(t) >= 2 and t["distance_m"].iloc[-1] > 0)
    if not ids:
        return [], np.empty((0, SIGNATURE_POINTS, 2)), np.empty(0)
    lat0 = float(np.median([tracks[i]["lat"].iloc[0] for i in ids]))
    sigs = np.stack([signature(tracks[i], lat0) for i in ids])
    lengths = np.array([float(tracks[i]["distance_m"].max() - tracks[i]["distance_m"].iloc[0])
                        for i in ids])
    return ids, sigs, lengths


def group_routes(tracks: dict[int, pd.DataFrame], min_runs: int = MIN_RUNS) -> pd.DataFrame:
    """Assign runs to routes: ``DataFrame[activity_id, route_id, is_reference]``.

    Only routes run at least ``min_runs`` times are kept.  ``route_id`` counts
    from 1, the route with the most runs first (ties: the earliest reference
    activity id first).
    """
    ids, sigs, lengths = _signatures(tracks)
    n = len(ids)
    if n < max(min_runs, 2):
        return pd.DataFrame({"activity_id": pd.Series(dtype="int64"),
                             "route_id": pd.Series(dtype="int64"),
                             "is_reference": pd.Series(dtype="bool")})

    # Cheap tests on every pair first, then the full track comparison on survivors.
    longer = np.maximum(lengths[:, None], lengths[None, :])
    close = np.abs(lengths[:, None] - lengths[None, :]) <= MAX_LENGTH_DIFF * longer
    for end in (0, -1):
        p = sigs[:, end, :]
        close &= np.hypot(p[:, None, 0] - p[None, :, 0],
                          p[:, None, 1] - p[None, :, 1]) <= MAX_ENDPOINT_M
    neighbours: list[set[int]] = [set() for _ in range(n)]
    for i, j in zip(*np.nonzero(np.triu(close, k=1))):
        if same_route(sigs[i], sigs[j]):
            neighbours[i].add(j)
            neighbours[j].add(i)

    unassigned = set(range(n))
    groups: list[tuple[int, list[int]]] = []
    while unassigned:
        # The reference: the run matching the most still-unassigned runs.
        ref = max(unassigned, key=lambda i: (len(neighbours[i] & unassigned), -ids[i]))
        members = sorted({ref} | (neighbours[ref] & unassigned))
        if len(members) < min_runs:
            break  # every later reference has no more matches than this one
        groups.append((ref, members))
        unassigned -= set(members)

    groups.sort(key=lambda g: (-len(g[1]), ids[g[0]]))
    rows = [(ids[m], route_id, m == ref)
            for route_id, (ref, members) in enumerate(groups, start=1) for m in members]
    return pd.DataFrame(rows, columns=GROUP_COLUMNS).astype(
        {"activity_id": "int64", "route_id": "int64", "is_reference": "bool"})


def route_shape(track: pd.DataFrame) -> str:
    """``"loop"``, ``"out-and-back"`` or ``"point-to-point"``, from one run's track."""
    sig = signature(track, float(track["lat"].iloc[0]))
    if np.hypot(*(sig[0] - sig[-1])) > LOOP_GAP_M:
        return "point-to-point"
    half = len(sig) // 2
    out_leg, back_leg = sig[: half + 1], sig[half:][::-1]
    gaps, _ = track_gaps(back_leg, out_leg)
    return "out-and-back" if gaps.mean() <= RETRACE_GAP_M else "loop"


def route_name(names: pd.Series, distance_km: float, shape: str, distance_text: str) -> str:
    """The title used on at least a third of the route's runs, if not a Strava default.

    Otherwise the route is described, e.g. ``"6.0 km loop"``.
    """
    counts = Counter(str(n).strip() for n in names.dropna() if str(n).strip())
    if counts:
        title, count = counts.most_common(1)[0]
        if title not in GENERIC_NAMES and count >= max(2, len(names) / 3):
            return title
    return f"{distance_text} {shape}"


def route_efforts(groups: pd.DataFrame, activities: pd.DataFrame, route_id: int) -> pd.DataFrame:
    """Every run on one route, oldest first, with the PR and the PR as it stood each time.

    The PR is the fastest average pace (so a run a few hundred metres longer is
    not penalised for the extra distance).  ``best_so_far`` is the fastest
    pace up to and including each run, the route's PR progression.
    """
    ids = groups.loc[groups["route_id"] == route_id, "activity_id"]
    runs = activities[activities["activity_id"].isin(ids)].sort_values("date")
    runs = runs[runs["distance_km"] > 0]
    if runs.empty:
        return pd.DataFrame(columns=EFFORT_COLUMNS)
    out = runs[["activity_id", "date", "name", "distance_km", "moving_time_s"]].copy()
    out["pace_min_per_km"] = out["moving_time_s"] / 60.0 / out["distance_km"]
    out["best_so_far"] = out["pace_min_per_km"].cummin()
    out["is_pr"] = out["activity_id"] == out.loc[out["pace_min_per_km"].idxmin(), "activity_id"]
    return out[EFFORT_COLUMNS].reset_index(drop=True)


def route_table(groups: pd.DataFrame, activities: pd.DataFrame, tracks: dict[int, pd.DataFrame],
                distance_text=lambda km: f"{km:.1f} km", min_runs: int = MIN_RUNS) -> pd.DataFrame:
    """One row per route that ``activities`` ran at least ``min_runs`` times.

    ``activities`` may be a period's runs: routes are found on the whole
    history (so their numbering is stable), then counted and their PRs taken
    within the runs given.  ``pr_time_s`` is the PR pace carried over the
    route's median distance.  Sorted by number of runs, most first.
    """
    references = groups.loc[groups["is_reference"]].set_index("route_id")["activity_id"]
    rows = []
    for route_id in sorted(groups["route_id"].unique()):
        efforts = route_efforts(groups, activities, route_id)
        if len(efforts) < min_runs:
            continue
        pr = efforts[efforts["is_pr"]].iloc[0]
        distance_km = float(efforts["distance_km"].median())
        reference = int(references.get(route_id, pr["activity_id"]))
        shape = route_shape(tracks[reference]) if reference in tracks else "route"
        rows.append({
            "route_id": int(route_id),
            "name": route_name(efforts["name"], distance_km, shape, distance_text(distance_km)),
            "shape": shape, "runs": len(efforts), "distance_km": distance_km,
            "reference_id": reference, "pr_id": int(pr["activity_id"]), "pr_date": pr["date"],
            "pr_time_s": float(pr["pace_min_per_km"]) * 60.0 * distance_km,
            "pr_pace_min_per_km": float(pr["pace_min_per_km"]),
            "median_pace_min_per_km": float(efforts["pace_min_per_km"].median()),
            "last_date": efforts["date"].max(),
        })
    table = pd.DataFrame(rows, columns=ROUTE_COLUMNS)
    return table.sort_values(["runs", "route_id"], ascending=[False, True]).reset_index(drop=True)
