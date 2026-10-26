"""Parse a Strava ``activities.csv`` export into the project's tidy schema.

Strava's export format has drifted across versions: some columns appear twice
(``Distance`` in the athlete's display unit *and* again in meters), durations
are sometimes integer seconds and sometimes ``H:MM:SS`` strings, and numeric
fields can carry thousands separators.  Everything here is therefore coerced
defensively with :func:`pandas.to_numeric(errors="coerce")` and the tidy frame
is derived from whichever numeric columns are actually present.

The contract produced by :func:`load_activities` is one row per run with:

===================  =========================================================
``activity_id``      int   -- stable id from the export (or a generated index)
``date``             datetime64[ns], tz-naive local time
``name``             str
``distance_km``      float
``moving_time_s``    int
``elapsed_time_s``   int
``pace_min_per_km``  float -- derived from *moving* time
``avg_hr``           float, nullable
``elevation_gain_m`` float, nullable
===================  =========================================================

Recent exports carry far more than that, so the frame also has these *extra*
columns.  Every one is always present and simply empty when the export lacks it:

=======================  =====================================================
``max_hr``               float -- highest heart rate recorded on the run
``avg_cadence_spm``      float -- steps per minute (per-leg values are doubled)
``gap_pace_min_per_km``  float -- grade-adjusted pace, from Strava's
                         ``Grade Adjusted Distance``
``relative_effort``      float -- Strava's own Relative Effort score
``gear``                 str, nullable -- the shoe the run was logged with
``filename``             str, nullable -- path of the run's GPS file inside the
                         export, e.g. ``activities/123.fit.gz``
=======================  =====================================================

Everything that is *not* a run -- rides, swims, walks, strength sessions -- is
read by :func:`load_other_activities` into a smaller frame (see
:data:`OTHER_COLUMNS`), which :mod:`src.sports` scores so it can count toward
Fitness and Fatigue.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample" / "activities.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CONTRACT_COLUMNS = [
    "activity_id",
    "date",
    "name",
    "distance_km",
    "moving_time_s",
    "elapsed_time_s",
    "pace_min_per_km",
    "avg_hr",
    "elevation_gain_m",
]

EXTRA_COLUMNS = [
    "max_hr",
    "avg_cadence_spm",
    "gap_pace_min_per_km",
    "relative_effort",
    "gear",
    "filename",
]

TIDY_COLUMNS = CONTRACT_COLUMNS + EXTRA_COLUMNS

# The non-run frame: ``activity_type`` is Strava's own label ("Ride", "Weight
# Training", ...); ``distance_km`` is 0 for sessions that cover no ground, and
# the power columns are empty for anything recorded without a power meter.
OTHER_COLUMNS = [
    "activity_id",
    "date",
    "name",
    "activity_type",
    "distance_km",
    "moving_time_s",
    "avg_hr",
    "max_hr",
    "elevation_gain_m",
    "avg_watts",
    "weighted_watts",
    "relative_effort",
]

# Candidate source columns, most trustworthy first.  Strava repeats "Distance";
# the duplicate (read as "Distance.1") is the meters column, so prefer it.
_CANDIDATES = {
    "activity_id": ("Activity ID", "activity_id", "id"),
    "date": ("Activity Date", "activity_date", "date", "start_date_local"),
    "name": ("Activity Name", "activity_name", "name"),
    "activity_type": ("Activity Type", "activity_type", "type"),
    "distance": ("Distance.1", "Distance", "distance_km", "distance"),
    "moving_time": ("Moving Time", "Moving Time.1", "moving_time_s", "moving_time"),
    "elapsed_time": ("Elapsed Time", "Elapsed Time.1", "elapsed_time_s", "elapsed_time"),
    "avg_hr": ("Average Heart Rate", "Average Heart Rate.1", "avg_hr", "average_heartrate"),
    "elevation_gain": ("Elevation Gain", "Elevation Gain.1", "elevation_gain_m", "total_elevation_gain"),
    "max_hr": ("Max Heart Rate.1", "Max Heart Rate", "max_hr", "max_heartrate"),
    "cadence": ("Average Cadence", "avg_cadence_spm", "average_cadence"),
    "grade_adjusted_distance": ("Grade Adjusted Distance", "grade_adjusted_distance"),
    "relative_effort": ("Relative Effort.1", "Relative Effort", "relative_effort", "suffer_score"),
    "gear": ("Activity Gear", "gear"),
    "filename": ("Filename", "filename"),
    "avg_watts": ("Average Watts", "Average Watts.1", "average_watts"),
    "weighted_watts": ("Weighted Average Power", "weighted_average_watts"),
}

# A distance series whose typical value exceeds this is read as meters, not km.
# No sane run is 300 km, and no sane run is 300 m, so the gap is unambiguous.
_METERS_THRESHOLD = 300.0

# Strava and most watches log running cadence per leg (~80-95); a typical
# value below this is doubled to steps per minute, where 150-190 is normal.
_PER_LEG_CADENCE_THRESHOLD = 120.0

# Grade-adjusted pace further than this from flat pace is treated as a bad
# value: even a mountain run rarely moves its flat-equivalent pace by 40%.
_GAP_RATIO_LIMITS = (0.6, 1.4)


def _pick(frame: pd.DataFrame, key: str) -> pd.Series | None:
    """Return the first candidate column present for ``key``, else ``None``."""
    for column in _CANDIDATES[key]:
        if column in frame.columns:
            return frame[column]
    return None


def _coerce_numeric(series: pd.Series | None) -> pd.Series | None:
    """Coerce to float, tolerating thousands separators and stray text."""
    if series is None:
        return None
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(series, errors="coerce")


def _coerce_seconds(series: pd.Series | None) -> pd.Series | None:
    """Coerce a duration column to seconds, accepting ``H:MM:SS`` strings."""
    if series is None:
        return None
    numeric = _coerce_numeric(series)
    missing = numeric.isna() & series.notna()
    if missing.any():
        parsed = pd.to_timedelta(series[missing].astype("string"), errors="coerce")
        numeric.loc[missing] = parsed.dt.total_seconds()
    return numeric


def _to_kilometers(series: pd.Series) -> pd.Series:
    """Normalize a distance column to km, inferring meters from magnitude."""
    positive = series[series > 0]
    if not positive.empty and positive.median() > _METERS_THRESHOLD:
        return series / 1000.0
    return series


def _to_steps_per_minute(series: pd.Series) -> pd.Series:
    """Normalize cadence to steps per minute, inferring per-leg values from magnitude."""
    positive = series[series > 0]
    if not positive.empty and positive.median() < _PER_LEG_CADENCE_THRESHOLD:
        series = series * 2.0
    return series.where(series > 0)


def _optional_text(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """A text column with blanks turned into missing values."""
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    text = series.astype("string").str.strip()
    return text.mask(text == "")


def _read_export(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    return raw


def _is_run(raw: pd.DataFrame) -> pd.Series | None:
    """``True`` for runs, or ``None`` when the export has no activity-type column."""
    activity_type = _pick(raw, "activity_type")
    if activity_type is None:
        return None
    return (activity_type.astype("string").str.strip().str.casefold() == "run").fillna(False)


def _durations(raw: pd.DataFrame, path: Path) -> tuple[pd.Series, pd.Series]:
    """Moving and elapsed seconds, each standing in for the other when missing."""
    moving_time_s = _coerce_seconds(_pick(raw, "moving_time"))
    elapsed_time_s = _coerce_seconds(_pick(raw, "elapsed_time"))
    if moving_time_s is None and elapsed_time_s is None:
        raise ValueError(f"{path}: no recognizable duration column in {list(raw.columns)}")
    if moving_time_s is None:  # older exports only carry elapsed time
        moving_time_s = elapsed_time_s
    if elapsed_time_s is None:
        elapsed_time_s = moving_time_s
    return moving_time_s, elapsed_time_s


def _dates(raw: pd.DataFrame, path: Path) -> pd.Series:
    raw_date = _pick(raw, "date")
    if raw_date is None:
        raise ValueError(f"{path}: no recognizable date column in {list(raw.columns)}")
    date = pd.to_datetime(raw_date, format="mixed", errors="coerce")
    if isinstance(date.dtype, pd.DatetimeTZDtype):  # normalize to tz-naive local
        date = date.dt.tz_localize(None)
    return date


def _ids(raw: pd.DataFrame) -> pd.Series:
    activity_id = _coerce_numeric(_pick(raw, "activity_id"))
    if activity_id is None:
        activity_id = pd.Series(np.arange(len(raw), dtype="float64"), index=raw.index)
    return activity_id


def _names(raw: pd.DataFrame) -> pd.Series:
    name = _pick(raw, "name")
    name = pd.Series([""] * len(raw), index=raw.index) if name is None else name
    return name.astype("string").fillna("").astype(str)


def _numeric_or_nan(raw: pd.DataFrame, key: str) -> pd.Series:
    values = _coerce_numeric(_pick(raw, key))
    if values is None:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return values.astype("float64")


def load_activities(path: str | Path = SAMPLE_CSV) -> pd.DataFrame:
    """Read a Strava ``activities.csv`` and return the tidy run-level frame.

    Runs are kept, everything else is filtered out, unusable rows (no date,
    non-positive distance or moving time) are dropped, and the number dropped
    is logged.  The frame is sorted by date with a fresh ``RangeIndex``.
    """
    path = Path(path)
    raw = _read_export(path)
    n_read = len(raw)

    is_run = _is_run(raw)
    if is_run is not None:
        raw = raw[is_run]
        LOGGER.info("Kept %d runs out of %d activities.", len(raw), n_read)
    n_runs = len(raw)

    distance = _coerce_numeric(_pick(raw, "distance"))
    if distance is None:
        raise ValueError(f"{path}: no recognizable distance column in {list(raw.columns)}")
    distance_km = _to_kilometers(distance)

    moving_time_s, elapsed_time_s = _durations(raw, path)

    nan_column = pd.Series(np.nan, index=raw.index, dtype="float64")
    cadence = _coerce_numeric(_pick(raw, "cadence"))
    grade_adjusted = _coerce_numeric(_pick(raw, "grade_adjusted_distance"))

    tidy = pd.DataFrame(
        {
            "activity_id": _ids(raw),
            "date": _dates(raw, path),
            "name": _names(raw),
            "distance_km": distance_km.astype("float64"),
            "moving_time_s": moving_time_s,
            "elapsed_time_s": elapsed_time_s,
            "avg_hr": _numeric_or_nan(raw, "avg_hr"),
            "elevation_gain_m": _numeric_or_nan(raw, "elevation_gain"),
            "max_hr": _numeric_or_nan(raw, "max_hr"),
            "avg_cadence_spm": nan_column if cadence is None else _to_steps_per_minute(cadence),
            "grade_adjusted_km": (nan_column if grade_adjusted is None
                                  else _to_kilometers(grade_adjusted)),
            "relative_effort": _numeric_or_nan(raw, "relative_effort"),
            "gear": _optional_text(_pick(raw, "gear"), raw.index),
            "filename": _optional_text(_pick(raw, "filename"), raw.index),
        }
    )

    usable = (
        tidy["date"].notna()
        & tidy["activity_id"].notna()
        & tidy["distance_km"].gt(0)
        & tidy["moving_time_s"].gt(0)
    )
    n_dropped = int((~usable).sum())
    if n_dropped:
        LOGGER.info(
            "Dropped %d of %d runs with a missing date/id or non-positive distance or moving time.",
            n_dropped,
            n_runs,
        )
    tidy = tidy[usable].copy()

    tidy["activity_id"] = tidy["activity_id"].astype("int64")
    tidy["moving_time_s"] = tidy["moving_time_s"].round().astype("int64")
    tidy["elapsed_time_s"] = (
        tidy["elapsed_time_s"].fillna(tidy["moving_time_s"]).round().astype("int64")
    )
    tidy["pace_min_per_km"] = (tidy["moving_time_s"] / 60.0) / tidy["distance_km"]

    # Grade-adjusted pace: the same moving time spread over Strava's
    # flat-equivalent distance, so a hilly run reads faster than its raw pace.
    gap = (tidy["moving_time_s"] / 60.0) / tidy["grade_adjusted_km"].where(
        tidy["grade_adjusted_km"] > 0)
    low, high = _GAP_RATIO_LIMITS
    tidy["gap_pace_min_per_km"] = gap.where(
        (gap / tidy["pace_min_per_km"]).between(low, high)).astype("float64")
    for column in ("max_hr", "avg_cadence_spm", "relative_effort"):
        tidy[column] = tidy[column].astype("float64")

    tidy = tidy.sort_values("date").reset_index(drop=True)
    return tidy[TIDY_COLUMNS]


def load_other_activities(path: str | Path = SAMPLE_CSV) -> pd.DataFrame:
    """Every activity in the export that is not a run, in the :data:`OTHER_COLUMNS` frame.

    Unlike runs, a session with no distance is kept -- a strength session or a
    yoga class covers no ground but is still training.  Rows with no date, no
    id or no moving time are dropped.  An export without an activity-type
    column is read as all runs, so this returns an empty frame for it.
    """
    path = Path(path)
    raw = _read_export(path)
    is_run = _is_run(raw)
    raw = raw.iloc[0:0] if is_run is None else raw[~is_run]

    distance = _coerce_numeric(_pick(raw, "distance"))
    distance_km = (pd.Series(0.0, index=raw.index) if distance is None
                   else _to_kilometers(distance).fillna(0.0).clip(lower=0.0))
    moving_time_s, _ = _durations(raw, path)
    activity_type = _optional_text(_pick(raw, "activity_type"), raw.index).fillna("Other")

    other = pd.DataFrame({
        "activity_id": _ids(raw),
        "date": _dates(raw, path),
        "name": _names(raw),
        "activity_type": activity_type.astype(str),
        "distance_km": distance_km.astype("float64"),
        "moving_time_s": moving_time_s,
        "avg_hr": _numeric_or_nan(raw, "avg_hr"),
        "max_hr": _numeric_or_nan(raw, "max_hr"),
        "elevation_gain_m": _numeric_or_nan(raw, "elevation_gain"),
        "avg_watts": _numeric_or_nan(raw, "avg_watts"),
        "weighted_watts": _numeric_or_nan(raw, "weighted_watts"),
        "relative_effort": _numeric_or_nan(raw, "relative_effort"),
    })
    usable = (other["date"].notna() & other["activity_id"].notna()
              & other["moving_time_s"].gt(0))
    other = other[usable].copy()
    other["activity_id"] = other["activity_id"].astype("int64")
    other["moving_time_s"] = other["moving_time_s"].round().astype("int64")
    for column in ("avg_watts", "weighted_watts"):
        other[column] = other[column].where(other[column] > 0)
    return other.sort_values("date").reset_index(drop=True)[OTHER_COLUMNS]


def find_real_export(raw_dir: str | Path = RAW_DIR) -> Path | None:
    """Return the user's own ``activities.csv`` in ``data/raw/``, if present.

    ``recovery.csv`` (sleep/HRV, see :mod:`src.recovery`) shares the folder and
    is never mistaken for a Strava export.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return None
    preferred = raw_dir / "activities.csv"
    if preferred.is_file():
        return preferred
    others = (path for path in sorted(raw_dir.glob("*.csv")) if path.name != "recovery.csv")
    return next(others, None)
