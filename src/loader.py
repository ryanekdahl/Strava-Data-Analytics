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
}

# A distance series whose typical value exceeds this is read as meters, not km.
# No sane run is 300 km, and no sane run is 300 m, so the gap is unambiguous.
_METERS_THRESHOLD = 300.0


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


def load_activities(path: str | Path = SAMPLE_CSV) -> pd.DataFrame:
    """Read a Strava ``activities.csv`` and return the tidy run-level frame.

    Runs are kept, everything else is filtered out, unusable rows (no date,
    non-positive distance or moving time) are dropped, and the number dropped
    is logged.  The frame is sorted by date with a fresh ``RangeIndex``.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    n_read = len(raw)

    activity_type = _pick(raw, "activity_type")
    if activity_type is not None:
        is_run = activity_type.astype("string").str.strip().str.casefold() == "run"
        raw = raw[is_run.fillna(False)]
        LOGGER.info("Kept %d runs out of %d activities.", len(raw), n_read)
    n_runs = len(raw)

    distance = _coerce_numeric(_pick(raw, "distance"))
    if distance is None:
        raise ValueError(f"{path}: no recognizable distance column in {list(raw.columns)}")
    distance_km = _to_kilometers(distance)

    moving_time_s = _coerce_seconds(_pick(raw, "moving_time"))
    elapsed_time_s = _coerce_seconds(_pick(raw, "elapsed_time"))
    if moving_time_s is None and elapsed_time_s is None:
        raise ValueError(f"{path}: no recognizable duration column in {list(raw.columns)}")
    if moving_time_s is None:  # older exports only carry elapsed time
        moving_time_s = elapsed_time_s
    if elapsed_time_s is None:
        elapsed_time_s = moving_time_s

    raw_date = _pick(raw, "date")
    if raw_date is None:
        raise ValueError(f"{path}: no recognizable date column in {list(raw.columns)}")
    date = pd.to_datetime(raw_date, format="mixed", errors="coerce")
    if isinstance(date.dtype, pd.DatetimeTZDtype):  # normalize to tz-naive local
        date = date.dt.tz_localize(None)

    activity_id = _coerce_numeric(_pick(raw, "activity_id"))
    if activity_id is None:
        activity_id = pd.Series(np.arange(len(raw), dtype="float64"), index=raw.index)

    name = _pick(raw, "name")
    name = pd.Series([""] * len(raw), index=raw.index) if name is None else name

    nan_column = pd.Series(np.nan, index=raw.index, dtype="float64")
    avg_hr = _coerce_numeric(_pick(raw, "avg_hr"))
    elevation_gain_m = _coerce_numeric(_pick(raw, "elevation_gain"))

    tidy = pd.DataFrame(
        {
            "activity_id": activity_id,
            "date": date,
            "name": name.astype("string").fillna("").astype(str),
            "distance_km": distance_km.astype("float64"),
            "moving_time_s": moving_time_s,
            "elapsed_time_s": elapsed_time_s,
            "avg_hr": nan_column if avg_hr is None else avg_hr,
            "elevation_gain_m": nan_column if elevation_gain_m is None else elevation_gain_m,
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

    tidy = tidy.sort_values("date").reset_index(drop=True)
    return tidy[CONTRACT_COLUMNS]


def find_real_export(raw_dir: str | Path = RAW_DIR) -> Path | None:
    """Return the user's own ``activities.csv`` in ``data/raw/``, if present."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return None
    preferred = raw_dir / "activities.csv"
    if preferred.is_file():
        return preferred
    return next(iter(sorted(raw_dir.glob("*.csv"))), None)
