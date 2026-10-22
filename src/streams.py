"""Second-by-second analysis from the GPS files inside a Strava export.

``activities.csv`` holds one summary row per run; the export's ``activities/``
folder holds the recording itself -- a ``.gpx``, ``.tcx`` or ``.fit`` file per
activity, usually gzipped, named in the CSV's ``Filename`` column.  Those files
turn three activity-level approximations into real measurements:

* **True best efforts.**  The fastest 1 km, 5 km or half marathon *anywhere
  inside* any run, found with a sliding window over the cumulative-distance
  stream, instead of the fastest whole run that was at least that long.
* **True time in zone.**  Every sample's heart rate is credited to its own
  zone, instead of the whole run going to the zone of its average.
* **Splits and drift.**  Per-kilometre (and per-mile) pace, heart rate and
  elevation, and aerobic decoupling -- how much the speed-to-heart-rate ratio
  fades from the first half of a run to the second.
* **Route maps.**  A thinned copy of the GPS track, one point per equal slice
  of distance, for drawing the run with its pace or heart rate along the way.

Parsing is the slow part (a FIT file takes a fraction of a second in pure
Python, and an export can hold thousands), so each run is reduced once to a
small JSON summary in a ``.stream_cache`` folder next to the export and only
re-parsed when its source file changes (the cache file's name carries the
source's size and modification time).

Sliding-window best efforts
---------------------------
Between samples, distance and time are linearly interpolated.  For a target
distance ``D`` the window is anchored on every sample twice -- once as its start
(the end time is interpolated at ``d_i + D``) and once as its end (the start is
interpolated at ``d_j - D``).  For a piecewise-linear distance curve the fastest
window always has one end on a sample, so this search is exact, not a grid
approximation.

Moving time, not elapsed
------------------------
Pace in ``activities.csv`` is moving time over distance, so every stream
computation runs on a *moving clock* too -- otherwise a two-minute wait at a
crossing would make the true in-run best slower than the whole-run estimate it
is meant to improve on.  A step slower than :data:`STOP_SPEED_M_S` is a stop.
It is credited only with the time its few metres would take at the run's
median moving speed, so a pause logged as one long gap between two GPS fixes
costs no more than the running inside it.

GPS glitches are the main threat to a best effort -- one bad fix can teleport a
runner 200 m in a second.  Any step implying more than
:data:`MAX_SPEED_M_S` (faster than any human sprints) is clipped to that speed
before the search, so a glitch can shorten a best effort only negligibly.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CACHE_DIR_NAME = ".stream_cache"
CACHE_VERSION = 2  # bump when the summary format or any computation changes

STREAM_COLUMNS = ["t_s", "distance_m", "hr", "cadence_spm", "altitude_m", "lat", "lon"]

MAX_SPEED_M_S = 11.0  # ~40 km/h; faster steps are GPS glitches, not running
STOP_SPEED_M_S = 0.5  # slower than a shuffle: standing at a light, not running
MIN_DECOUPLING_MINUTES = 20.0
MILE_M = 1609.344
ROUTE_POINTS = 300  # points kept per route: smooth on a map, ~10 KB per run in the cache
ROUTE_COLUMNS = ["lat", "lon", "distance_m", "moving_s", "hr"]
_EARTH_RADIUS_M = 6_371_008.8

# Named distances shown in the best-effort table.
NAMED_DISTANCES_M: dict[str, float] = {
    "400 m": 400.0,
    "1K": 1000.0,
    "1 mile": 1609.344,
    "3K": 3000.0,
    "5K": 5000.0,
    "10K": 10000.0,
    "15K": 15000.0,
    "Half Marathon": 21097.5,
    "30K": 30000.0,
    "Marathon": 42195.0,
}

# The pace curve is the named distances plus a log-spaced grid between them.
CURVE_DISTANCES_M: tuple[float, ...] = tuple(sorted(
    set(NAMED_DISTANCES_M.values())
    | {round(float(d), 1) for d in np.geomspace(400.0, 42195.0, 28)}
))

_PER_LEG_CADENCE_THRESHOLD = 120.0


class StreamData(NamedTuple):
    """Everything the dashboard reads from the cached stream summaries.

    ``best``        long ``DataFrame[activity_id, date, distance_m, time_s]``
    ``hr_seconds``  long ``DataFrame[activity_id, date, bpm, seconds]``
    ``splits``      ``DataFrame[activity_id, split, distance_m, time_s,
                    pace_min_per_km, avg_hr, cadence_spm, elevation_change_m]``
    ``mile_splits`` the same, one row per mile (pace still in min/km)
    ``runs``        ``DataFrame[activity_id, duration_s, moving_s, distance_m,
                    decoupling_pct, has_hr]`` -- one row per parsed file
    ``status``      counts of ``parsed``, ``failed`` and ``missing`` files
    """

    best: pd.DataFrame
    hr_seconds: pd.DataFrame
    splits: pd.DataFrame
    mile_splits: pd.DataFrame
    runs: pd.DataFrame
    status: dict[str, int]


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------

def _open_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() == ".gz" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def _base_suffix(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def _local(tag: str) -> str:
    """Element tag without its XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _float(text: str | None) -> float:
    try:
        return float(text) if text is not None else math.nan
    except ValueError:
        return math.nan


def _haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance along a track, in meters."""
    if len(lat) < 2:
        return np.zeros(len(lat))
    phi = np.radians(lat)
    dphi = np.diff(phi)
    dlmb = np.radians(np.diff(lon))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(dlmb / 2) ** 2
    step = 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    step = np.nan_to_num(step, nan=0.0)
    return np.concatenate([[0.0], np.cumsum(step)])


def _frame(times: list, distance: Iterable[float], hr, cadence, altitude,
           lat=None, lon=None) -> pd.DataFrame:
    """Assemble a stream from parallel lists, with time as seconds from the start."""
    missing = np.full(len(times), np.nan)
    stamps = pd.to_datetime(pd.Series(times, dtype="object"), utc=True,
                            format="ISO8601", errors="coerce")
    frame = pd.DataFrame({
        "time": stamps,
        "distance_m": np.asarray(list(distance), dtype="float64"),
        "hr": np.asarray(hr, dtype="float64"),
        "cadence_spm": np.asarray(cadence, dtype="float64"),
        "altitude_m": np.asarray(altitude, dtype="float64"),
        "lat": missing if lat is None else np.asarray(lat, dtype="float64"),
        "lon": missing if lon is None else np.asarray(lon, dtype="float64"),
    })
    return _finish(frame)


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop untimed samples, and express time as seconds from the start."""
    frame = frame.dropna(subset=["time"]).sort_values("time", kind="stable")
    frame = frame.drop_duplicates("time", keep="last").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=STREAM_COLUMNS, dtype="float64")
    frame["t_s"] = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds()
    frame["distance_m"] = frame["distance_m"].ffill().fillna(0.0).cummax()
    frame["hr"] = frame["hr"].where(frame["hr"].between(30, 250))
    cadence = frame["cadence_spm"].where(frame["cadence_spm"] > 0)
    if cadence.notna().any() and cadence.median() < _PER_LEG_CADENCE_THRESHOLD:
        cadence = cadence * 2.0  # per-leg -> steps per minute
    frame["cadence_spm"] = cadence
    for column in ("lat", "lon"):
        if column not in frame:
            frame[column] = np.nan
    return frame[STREAM_COLUMNS].astype("float64")


def _read_gpx(data: bytes) -> pd.DataFrame:
    times, lat, lon, hr, cad, ele = [], [], [], [], [], []
    for _, element in ET.iterparse(io.BytesIO(data)):
        if _local(element.tag) != "trkpt":
            continue
        values = {"time": None, "ele": None, "hr": None, "cad": None}
        for child in element.iter():
            name = _local(child.tag)
            if name in values and values[name] is None:
                values[name] = child.text
        times.append(values["time"])
        lat.append(_float(element.get("lat")))
        lon.append(_float(element.get("lon")))
        hr.append(_float(values["hr"]))
        cad.append(_float(values["cad"]))
        ele.append(_float(values["ele"]))
        element.clear()
    lat_a, lon_a = np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64")
    located = ~(np.isnan(lat_a) | np.isnan(lon_a))
    distance = np.full(len(lat_a), np.nan)
    distance[located] = _haversine_m(lat_a[located], lon_a[located])
    return _frame(times, distance, hr, cad, ele, lat_a, lon_a)


def _read_tcx(data: bytes) -> pd.DataFrame:
    times, dist, hr, cad, ele, lat, lon = [], [], [], [], [], [], []
    for _, element in ET.iterparse(io.BytesIO(data)):
        if _local(element.tag) != "Trackpoint":
            continue
        values: dict[str, str | None] = {}
        for child in element.iter():
            name = _local(child.tag)
            if name == "HeartRateBpm":
                values["hr"] = next((v.text for v in child if _local(v.tag) == "Value"), None)
            elif name in ("Time", "DistanceMeters", "AltitudeMeters", "Cadence", "RunCadence",
                          "LatitudeDegrees", "LongitudeDegrees"):
                values.setdefault(name, child.text)
        times.append(values.get("Time"))
        dist.append(_float(values.get("DistanceMeters")))
        hr.append(_float(values.get("hr")))
        cad.append(_float(values.get("RunCadence") or values.get("Cadence")))
        ele.append(_float(values.get("AltitudeMeters")))
        lat.append(_float(values.get("LatitudeDegrees")))
        lon.append(_float(values.get("LongitudeDegrees")))
        element.clear()
    return _frame(times, dist, hr, cad, ele, lat, lon)


def _read_fit(data: bytes) -> pd.DataFrame:
    try:
        import fitdecode
    except ImportError as err:  # pragma: no cover - exercised only without the package
        raise RuntimeError("reading .fit files needs the 'fitdecode' package") from err

    rows = []
    with fitdecode.FitReader(io.BytesIO(data)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage) or frame.name != "record":
                continue

            def value(*names):
                for name in names:
                    if frame.has_field(name):
                        found = frame.get_value(name)
                        if found is not None:
                            return found
                return None

            stamp = value("timestamp")
            if stamp is None:
                continue
            lat, lon = value("position_lat"), value("position_long")
            rows.append({
                "time": pd.Timestamp(stamp),
                "distance_m": value("distance"),
                "hr": value("heart_rate"),
                "cadence_spm": value("cadence"),
                "altitude_m": value("enhanced_altitude", "altitude"),
                # FIT stores positions as semicircles.
                "lat": None if lat is None else lat * 180.0 / 2**31,
                "lon": None if lon is None else lon * 180.0 / 2**31,
            })
    frame = pd.DataFrame(rows, columns=["time", "distance_m", "hr", "cadence_spm",
                                        "altitude_m", "lat", "lon"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    for column in ("distance_m", "hr", "cadence_spm", "altitude_m", "lat", "lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["distance_m"].isna().all() and frame["lat"].notna().any():
        located = frame[["lat", "lon"]].notna().all(axis=1)
        frame.loc[located, "distance_m"] = _haversine_m(
            frame.loc[located, "lat"].to_numpy(), frame.loc[located, "lon"].to_numpy())
    return _finish(frame)


_READERS: dict[str, Callable[[bytes], pd.DataFrame]] = {
    ".gpx": _read_gpx, ".tcx": _read_tcx, ".fit": _read_fit,
}


def read_stream(path: str | Path) -> pd.DataFrame:
    """Read one activity file into ``DataFrame[t_s, distance_m, hr, cadence_spm, altitude_m,
    lat, lon]``.

    ``t_s`` is seconds since the first sample, ``distance_m`` is cumulative and
    never decreases, and cadence is in steps per minute; ``lat`` and ``lon`` are
    ``NaN`` where the file has no position (a treadmill run, say).  Raises
    ``ValueError`` for a format other than GPX, TCX or FIT.
    """
    path = Path(path)
    reader = _READERS.get(_base_suffix(path))
    if reader is None:
        raise ValueError(f"unsupported activity file type: {path.name}")
    return reader(_open_bytes(path))


# --------------------------------------------------------------------------
# Computing from a stream
# --------------------------------------------------------------------------

def clean_distance(t_s: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    """Cumulative distance with every step capped at :data:`MAX_SPEED_M_S`."""
    if len(distance_m) < 2:
        return np.asarray(distance_m, dtype="float64")
    steps = np.diff(distance_m)
    caps = np.maximum(np.diff(t_s), 0.0) * MAX_SPEED_M_S
    steps = np.clip(steps, 0.0, caps)
    return np.concatenate([[distance_m[0]], distance_m[0] + np.cumsum(steps)])


def moving_clock(t_s: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    """Cumulative moving time at each sample (see "Moving time, not elapsed").

    ``distance_m`` should already be glitch-free (:func:`clean_distance`).
    """
    t_s = np.asarray(t_s, dtype="float64")
    if len(t_s) < 2:
        return np.zeros(len(t_s))
    dt = np.maximum(np.diff(t_s), 0.0)
    dd = np.maximum(np.diff(distance_m), 0.0)
    speed = np.divide(dd, dt, out=np.zeros_like(dd), where=dt > 0)
    moving = speed >= STOP_SPEED_M_S
    typical = float(np.median(speed[moving])) if moving.any() else 0.0
    stopped = np.minimum(dt, dd / typical) if typical > 0 else np.zeros_like(dt)
    return np.concatenate([[0.0], np.cumsum(np.where(moving, dt, stopped))])


def _moving(stream: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(moving time, cleaned distance)`` arrays for a stream."""
    t = stream["t_s"].to_numpy(dtype="float64")
    d = clean_distance(t, stream["distance_m"].to_numpy(dtype="float64"))
    return moving_clock(t, d), d


def _distance_index(t_s: np.ndarray, distance_m: np.ndarray):
    """Unique distances with the first and last time the runner was at each.

    A pause shows up as several samples at one distance; keeping both ends of
    it lets :func:`_time_at` place the pause correctly instead of smearing it
    across the next stretch of running.
    """
    d = np.asarray(distance_m, dtype="float64")
    moved = np.diff(d) > 0
    first = np.concatenate([[True], moved])
    last = np.concatenate([moved, [True]])
    return d[first], t_s[first], t_s[last]


def _time_at(x: np.ndarray, u: np.ndarray, t_first: np.ndarray, t_last: np.ndarray,
             arriving: bool) -> np.ndarray:
    """Time at distance ``x`` on the piecewise-linear track.

    Between samples the runner moves from the *last* moment at one distance to
    the *first* moment at the next.  Exactly at a sampled distance the answer
    is the first moment there when ``arriving`` (a window's end) and the last
    when leaving (a window's start).
    """
    x = np.asarray(x, dtype="float64")
    k = np.clip(np.searchsorted(u, x, side="right") - 1, 0, len(u) - 1)
    nxt = np.minimum(k + 1, len(u) - 1)
    span = u[nxt] - u[k]
    frac = np.where(span > 0, (x - u[k]) / np.where(span > 0, span, 1.0), 0.0)
    between = t_last[k] + frac * (t_first[nxt] - t_last[k])
    exact = np.isclose(x, u[k], rtol=0.0, atol=1e-6)
    return np.where(exact, t_first[k] if arriving else t_last[k], between)


def best_times(stream: pd.DataFrame, distances_m: Iterable[float] = CURVE_DISTANCES_M) -> np.ndarray:
    """Fastest moving time, in seconds, to cover each distance anywhere in the run.

    Distances longer than the run are ``NaN``.  See the module docstring for
    why anchoring windows on samples from both ends makes the search exact.
    """
    distances_m = np.asarray(list(distances_m), dtype="float64")
    result = np.full(len(distances_m), np.nan)
    if len(stream) < 2:
        return result
    u, t_first, t_last = _distance_index(*_moving(stream))
    for k, target in enumerate(distances_m):
        if target <= 0 or target > u[-1] - u[0] + 1e-6:
            continue
        starts = u[u + target <= u[-1] + 1e-6]
        from_start = (_time_at(starts + target, u, t_first, t_last, arriving=True)
                      - _time_at(starts, u, t_first, t_last, arriving=False))
        ends = u[u - target >= u[0] - 1e-6]
        from_end = (_time_at(ends, u, t_first, t_last, arriving=True)
                    - _time_at(ends - target, u, t_first, t_last, arriving=False))
        result[k] = float(np.concatenate([from_start, from_end]).min())
    return result


def _sample_seconds(moving_t: np.ndarray) -> np.ndarray:
    """Moving seconds each sample represents: the moving time until the next one."""
    if len(moving_t) == 0:
        return np.zeros(0)
    return np.diff(moving_t, append=moving_t[-1])


def hr_histogram(stream: pd.DataFrame) -> dict[int, float]:
    """Moving seconds spent at each whole-number heart rate.

    Storing the histogram rather than zone totals means zones can be redrawn
    for any resting and max heart rate without re-reading a single file.
    """
    hr = stream["hr"].to_numpy(dtype="float64")
    seconds = _sample_seconds(_moving(stream)[0])
    valid = ~np.isnan(hr) & (seconds > 0)
    if not valid.any():
        return {}
    totals = pd.Series(seconds[valid]).groupby(np.rint(hr[valid]).astype(int)).sum()
    return {int(bpm): float(sec) for bpm, sec in totals.items()}


def km_splits(stream: pd.DataFrame, split_m: float = 1000.0) -> pd.DataFrame:
    """Per-kilometre moving time, pace, heart rate, cadence and elevation change.

    The final partial split keeps its own distance, so its pace stays honest;
    a sliver under 100 m is folded into the split before it instead.
    """
    columns = ["split", "distance_m", "time_s", "pace_min_per_km", "avg_hr",
               "cadence_spm", "elevation_change_m"]
    if len(stream) < 2:
        return pd.DataFrame(columns=columns)
    t, d = _moving(stream)
    if d[-1] - d[0] < 100.0:
        return pd.DataFrame(columns=columns)
    edges = list(np.arange(d[0], d[-1], split_m)) + [d[-1]]
    if len(edges) > 2 and edges[-1] - edges[-2] < 100.0:
        edges.pop(-2)  # fold a sliver of a split into the one before it
    u, t_first, t_last = _distance_index(t, d)
    edge_times = _time_at(np.asarray(edges), u, t_first, t_last, arriving=True)
    altitude = stream["altitude_m"].to_numpy(dtype="float64")
    has_alt = ~np.isnan(altitude)
    edge_alt = (np.interp(edges, d[has_alt], altitude[has_alt])
                if has_alt.sum() >= 2 else np.full(len(edges), np.nan))
    seconds = _sample_seconds(t)
    rows = []
    for i in range(len(edges) - 1):
        span = edges[i + 1] - edges[i]
        duration = edge_times[i + 1] - edge_times[i]
        inside = (t >= edge_times[i]) & (t < edge_times[i + 1])
        rows.append({
            "split": i + 1,
            "distance_m": float(span),
            "time_s": float(duration),
            "pace_min_per_km": float(duration / 60.0 / (span / 1000.0)) if span > 0 else math.nan,
            "avg_hr": _weighted_mean(stream["hr"].to_numpy()[inside], seconds[inside]),
            "cadence_spm": _weighted_mean(stream["cadence_spm"].to_numpy()[inside],
                                          seconds[inside]),
            "elevation_change_m": float(edge_alt[i + 1] - edge_alt[i]),
        })
    return pd.DataFrame(rows, columns=columns)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype="float64")
    valid = ~np.isnan(values) & (weights > 0)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def aerobic_decoupling(stream: pd.DataFrame) -> float:
    """Percent drop in speed per heartbeat from the first half of a run to the second.

    Joe Friel's Pa:Hr decoupling: ``EF = speed / HR`` for each half by moving
    time, and ``(EF_first - EF_second) / EF_first * 100``.  Under about 5% on a
    steady run means the aerobic base held up; a large value means the heart
    was working harder for the same speed by the end.  ``NaN`` for runs shorter
    than 20 minutes or without heart rate.  It is only meaningful for steady
    runs -- an interval session or a fast finish decouples by design.
    """
    hr = stream["hr"].to_numpy(dtype="float64")
    if len(hr) < 10 or np.isnan(hr).mean() > 0.5:
        return math.nan
    t, d = _moving(stream)
    if t[-1] < MIN_DECOUPLING_MINUTES * 60.0:
        return math.nan
    step_t, step_d = np.diff(t), np.diff(d)
    step_hr = (hr[:-1] + hr[1:]) / 2.0
    halves = t[1:] <= t[-1] / 2.0
    ratios = []
    for half in (halves, ~halves):
        mask = half & (step_t > 0) & ~np.isnan(step_hr)
        seconds = step_t[mask].sum()
        if seconds <= 0:
            return math.nan
        speed = step_d[mask].sum() / seconds
        heart = np.average(step_hr[mask], weights=step_t[mask])
        ratios.append(speed / heart)
    return float((ratios[0] - ratios[1]) / ratios[0] * 100.0)


def route_track(stream: pd.DataFrame, points: int = ROUTE_POINTS) -> pd.DataFrame:
    """A thinned copy of the GPS track: ``DataFrame[lat, lon, distance_m, moving_s, hr]``.

    One sample is kept per equal slice of distance rather than of time, so a
    long stop does not crowd one spot with points and every stretch of the run
    is drawn at the same resolution.  Empty when the stream has no positions.
    """
    if not {"lat", "lon"} <= set(stream.columns):
        return pd.DataFrame(columns=ROUTE_COLUMNS, dtype="float64")
    located = stream[["lat", "lon"]].notna().all(axis=1).to_numpy()
    if located.sum() < 2:
        return pd.DataFrame(columns=ROUTE_COLUMNS, dtype="float64")
    t, d = _moving(stream)
    t, d = t[located], d[located]
    targets = np.linspace(d[0], d[-1], points)
    keep = np.unique(np.clip(np.searchsorted(d, targets, side="left"), 0, len(d) - 1))
    return pd.DataFrame({
        "lat": stream["lat"].to_numpy()[located][keep],
        "lon": stream["lon"].to_numpy()[located][keep],
        "distance_m": d[keep] - d[0],
        "moving_s": t[keep] - t[0],
        "hr": stream["hr"].to_numpy()[located][keep],
    }, columns=ROUTE_COLUMNS)


def summarize_stream(stream: pd.DataFrame) -> dict:
    """Reduce a stream to the JSON-friendly summary the cache stores."""
    best = best_times(stream)
    splits = km_splits(stream)
    mile_splits = km_splits(stream, MILE_M)
    route = route_track(stream).round({"lat": 6, "lon": 6, "distance_m": 1, "moving_s": 1,
                                       "hr": 0})
    return {
        "duration_s": float(stream["t_s"].iloc[-1]) if len(stream) else 0.0,
        "moving_s": float(_moving(stream)[0][-1]) if len(stream) else 0.0,
        "distance_m": float(stream["distance_m"].iloc[-1] - stream["distance_m"].iloc[0])
        if len(stream) else 0.0,
        "best_s": [None if np.isnan(x) else round(float(x), 2) for x in best],
        "hr_hist": {str(k): round(v, 1) for k, v in hr_histogram(stream).items()},
        "decoupling_pct": None if np.isnan(dc := aerobic_decoupling(stream)) else round(dc, 2),
        "splits": json.loads(splits.to_json(orient="records")),
        "mile_splits": json.loads(mile_splits.to_json(orient="records")),
        # Column lists rather than records keep the cache small.
        "route": {column: json.loads(route[column].to_json(orient="values"))
                  for column in ROUTE_COLUMNS},
    }


# --------------------------------------------------------------------------
# The on-disk cache
# --------------------------------------------------------------------------

def cache_dir_for(export_csv: str | Path) -> Path:
    return Path(export_csv).parent / CACHE_DIR_NAME


def _cache_path(cache_dir: Path, activity_id: int, source: Path) -> Path:
    """Cache file for one run, named after the source file's size and mtime.

    Putting the key in the name means checking whether a run is up to date is
    a single ``stat`` -- no JSON is opened on a page load that has nothing new.
    """
    stat = source.stat()
    return cache_dir / f"v{CACHE_VERSION}-{int(activity_id)}-{stat.st_size}-{stat.st_mtime_ns}.json"


def _read_cache(cache_dir: Path, activity_id: int, source: Path) -> dict | None:
    try:
        return json.loads(_cache_path(cache_dir, activity_id, source).read_text())
    except (OSError, ValueError):
        return None


def stream_files(activities: pd.DataFrame, export_csv: str | Path) -> list[tuple[int, Path]]:
    """``(activity_id, path)`` for every run whose ``filename`` points at a file."""
    if "filename" not in activities:
        return []
    root = Path(export_csv).parent
    named = activities.dropna(subset=["filename"])
    return [(int(aid), root / str(name))
            for aid, name in zip(named["activity_id"], named["filename"])]


def pending_files(activities: pd.DataFrame, export_csv: str | Path) -> list[tuple[int, Path]]:
    """Stream files that exist but have no up-to-date cached summary."""
    cache_dir = cache_dir_for(export_csv)
    return [(activity_id, path) for activity_id, path in stream_files(activities, export_csv)
            if path.is_file() and not _cache_path(cache_dir, activity_id, path).is_file()]


def cache_summary(activity_id: int, path: Path, cache_dir: Path) -> bool:
    """Parse one file and write its summary; returns ``False`` if it failed.

    Failures are cached too (with the error message), so a corrupt file is
    reported once instead of being re-parsed on every page load.
    """
    try:
        stream = read_stream(path)
        if len(stream) < 2:
            raise ValueError("fewer than two timed samples")
        record = {"file": path.name, **summarize_stream(stream)}
        ok = True
    except Exception as err:  # noqa: BLE001 - any unreadable file is skipped, not fatal
        LOGGER.warning("Could not read %s: %s", path.name, err)
        record = {"file": path.name, "error": f"{type(err).__name__}: {err}"}
        ok = False
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(cache_dir, activity_id, path)
    for stale in cache_dir.glob(f"v*-{int(activity_id)}-*.json"):
        if stale != target:  # an older version, or an earlier copy of the file
            stale.unlink(missing_ok=True)
    target.write_text(json.dumps(record))
    return ok


def build_cache(
    activities: pd.DataFrame,
    export_csv: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Parse every pending file, reporting ``progress(done, total)``; returns the count."""
    pending = pending_files(activities, export_csv)
    cache_dir = cache_dir_for(export_csv)
    for done, (activity_id, path) in enumerate(pending, start=1):
        cache_summary(activity_id, path, cache_dir)
        if progress is not None:
            progress(done, len(pending))
    return len(pending)


def load_stream_data(activities: pd.DataFrame, export_csv: str | Path) -> StreamData:
    """Gather every cached summary for these runs into long-form frames.

    Runs whose file is missing, unparsed or failed are simply absent, and
    ``status`` says how many of each there were.
    """
    cache_dir = cache_dir_for(export_csv)
    dates = activities.set_index("activity_id")["date"]
    best_rows, hr_rows, split_frames, mile_frames, run_rows = [], [], [], [], []
    status = {"parsed": 0, "failed": 0, "missing": 0}
    for activity_id, path in stream_files(activities, export_csv):
        if not path.is_file():
            status["missing"] += 1
            continue
        cached = _read_cache(cache_dir, activity_id, path)
        if cached is None or "error" in cached or len(cached.get("best_s", [])) != len(
                CURVE_DISTANCES_M):
            status["failed" if cached and "error" in cached else "missing"] += 1
            continue
        status["parsed"] += 1
        date = dates.get(activity_id)
        for distance, seconds in zip(CURVE_DISTANCES_M, cached["best_s"]):
            if seconds is not None:
                best_rows.append((activity_id, date, distance, seconds))
        for bpm, seconds in cached["hr_hist"].items():
            hr_rows.append((activity_id, date, int(bpm), seconds))
        if cached["splits"]:
            split_frames.append(pd.DataFrame(cached["splits"]).assign(activity_id=activity_id))
        if cached["mile_splits"]:
            mile_frames.append(pd.DataFrame(cached["mile_splits"])
                               .assign(activity_id=activity_id))
        run_rows.append((activity_id, cached["duration_s"], cached["moving_s"],
                         cached["distance_m"], cached["decoupling_pct"],
                         bool(cached["hr_hist"])))

    best = pd.DataFrame(best_rows, columns=["activity_id", "date", "distance_m", "time_s"])
    hr_seconds = pd.DataFrame(hr_rows, columns=["activity_id", "date", "bpm", "seconds"])
    split_columns = ["activity_id", "split", "distance_m", "time_s", "pace_min_per_km",
                     "avg_hr", "cadence_spm", "elevation_change_m"]
    splits, mile_splits = (
        pd.concat(frames, ignore_index=True)[split_columns] if frames
        else pd.DataFrame(columns=split_columns)
        for frames in (split_frames, mile_frames))
    runs = pd.DataFrame(run_rows, columns=["activity_id", "duration_s", "moving_s",
                                           "distance_m", "decoupling_pct", "has_hr"])
    for frame in (best, hr_seconds):
        frame["date"] = pd.to_datetime(frame["date"])
    runs["decoupling_pct"] = runs["decoupling_pct"].astype("float64")
    return StreamData(best, hr_seconds, splits, mile_splits, runs, status)


def restrict(data: StreamData, activity_ids) -> StreamData:
    """The same stream data, keeping only the given runs (``status`` is unchanged)."""
    ids = set(int(i) for i in activity_ids)
    return data._replace(**{
        name: getattr(data, name)[getattr(data, name)["activity_id"].isin(ids)]
        for name in ("best", "hr_seconds", "splits", "mile_splits", "runs")
    })


def route_for(activity_id: int, activities: pd.DataFrame,
              export_csv: str | Path) -> pd.DataFrame:
    """One run's cached route, with a smoothed pace at every point.

    Read on demand rather than with :func:`load_stream_data`, since only the
    run on screen needs its track.  Returns ``DataFrame[lat, lon, distance_m,
    moving_s, hr, pace_min_per_km]``, empty when the run has no GPS positions.
    """
    columns = ROUTE_COLUMNS + ["pace_min_per_km"]
    path = dict(stream_files(activities, export_csv)).get(int(activity_id))
    cached = (_read_cache(cache_dir_for(export_csv), activity_id, path)
              if path is not None and path.is_file() else None)
    if not cached or not cached.get("route", {}).get("lat"):
        return pd.DataFrame(columns=columns, dtype="float64")
    route = pd.DataFrame(cached["route"], columns=ROUTE_COLUMNS).astype("float64")
    # Pace over a few points either side: GPS jitter makes point-to-point pace unreadable.
    span = 3
    ahead, behind = route.shift(-span).bfill().ffill(), route.shift(span).ffill().bfill()
    dist = (ahead["distance_m"] - behind["distance_m"]) / 1000.0
    route["pace_min_per_km"] = ((ahead["moving_s"] - behind["moving_s"]) / 60.0
                                / dist.where(dist > 0))
    return route[columns]


# --------------------------------------------------------------------------
# Views over the cached data
# --------------------------------------------------------------------------

def best_effort_table(
    best: pd.DataFrame, activities: pd.DataFrame, since: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Fastest in-run effort at each named distance, and the run it came from.

    Returns ``DataFrame[distance, distance_km, time_s, pace_min_per_km,
    activity_id, date, name]``; distances never covered are omitted.
    """
    columns = ["distance", "distance_km", "time_s", "pace_min_per_km",
               "activity_id", "date", "name"]
    if since is not None:
        best = best[best["date"] >= since]
    names = activities.set_index("activity_id")["name"]
    rows = []
    for label, meters in NAMED_DISTANCES_M.items():
        at = best[np.isclose(best["distance_m"], meters)]
        if at.empty:
            continue
        top = at.loc[at["time_s"].idxmin()]
        rows.append({
            "distance": label, "distance_km": meters / 1000.0, "time_s": float(top["time_s"]),
            "pace_min_per_km": float(top["time_s"]) / 60.0 / (meters / 1000.0),
            "activity_id": int(top["activity_id"]), "date": top["date"],
            "name": names.get(int(top["activity_id"]), ""),
        })
    return pd.DataFrame(rows, columns=columns)


def pace_curve(best: pd.DataFrame, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Best pace at every curve distance: the running analogue of a power curve.

    Returns ``DataFrame[distance_km, time_s, pace_min_per_km, activity_id, date]``.
    """
    if since is not None:
        best = best[best["date"] >= since]
    if best.empty:
        return pd.DataFrame(columns=["distance_km", "time_s", "pace_min_per_km",
                                     "activity_id", "date"])
    top = best.loc[best.groupby("distance_m")["time_s"].idxmin()].sort_values("distance_m")
    return pd.DataFrame({
        "distance_km": top["distance_m"].to_numpy() / 1000.0,
        "time_s": top["time_s"].to_numpy(),
        "pace_min_per_km": top["time_s"].to_numpy() / 60.0 / (top["distance_m"].to_numpy() / 1000.0),
        "activity_id": top["activity_id"].to_numpy(),
        "date": top["date"].to_numpy(),
    })
