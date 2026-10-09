"""Sleep score and HRV from a wearable, lined up against training load.

Strava knows what you did; a wearable (Whoop, Oura, Garmin, Apple Watch via an
export app) knows how you slept and how recovered your nervous system looked
the next morning.  This module imports that second file and asks one question:
**does training move my recovery numbers?**

Input format
------------
Any CSV with a date column and at least one of a sleep-score or HRV column.
Column names are matched loosely (case, spaces and punctuation ignored), so the
template ``date,sleep_score,hrv_ms`` works, and so do vendor exports such as
Whoop's ``Cycle start time`` / ``Sleep performance %`` / ``Heart rate
variability (ms)``.  Several rows on one day (naps, multiple sleeps) are
averaged.

**Dates are the morning the number describes.**  A sleep score and an overnight
HRV belong to the night *before* that date.  Some exports stamp a row with the
time you fell asleep instead (e.g. ``2026-03-01 22:40``), so any timestamp at or
after 15:00 is rolled forward to the next morning.

How the pairing works
---------------------
For a recovery reading dated ``d``:

=====================  ======================================================
``load_prev_day``      training load on ``d - 1``, the day before that night
``atl_prev``           Fatigue (7-day EWMA) at the end of ``d - 1``
``tsb``                Form on ``d``, already defined from ``d - 1`` values
``load_same_day``      training load on ``d`` -- *after* the reading, so it
                       asks "do I train harder when recovered?" not the reverse
=====================  ======================================================

Correlation is Pearson's ``r`` over the days where both values exist, reported
with its sample size.  It shows association, not cause: a hard week, a late
night and a bad HRV can all share a third cause (travel, illness, stress).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import RAW_DIR

RECOVERY_CSV = RAW_DIR / "recovery.csv"
RECOVERY_COLUMNS = ["date", "sleep_score", "hrv_ms"]

RECOVERY_METRICS = {"sleep_score": "Sleep score", "hrv_ms": "HRV (ms)"}
TRAINING_METRICS = {
    "load_prev_day": "Training load the day before",
    "atl_prev": "Fatigue (7-day) going into the night",
    "tsb": "Form (TSB)",
    "load_same_day": "Training load later that day",
}

MIN_PAIRS = 10  # below this, r is noise and is reported as missing
EVENING_HOUR = 15  # timestamps at/after this hour describe the next morning
_VALID_RANGE = {"sleep_score": (0.0, 100.0), "hrv_ms": (5.0, 300.0)}

TEMPLATE_CSV = "date,sleep_score,hrv_ms\n2026-03-01,82,64\n2026-03-02,75,58\n"


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def _find_column(columns: list[str], wanted: str) -> str | None:
    """Pick the source column for ``date``, ``sleep_score`` or ``hrv_ms``.

    Rolling or baseline variants (``HRV 7d avg``, ``Baseline HRV``) are skipped
    so a smoothed column never stands in for the nightly value.
    """
    normalized = {column: _normalize(column) for column in columns}
    if wanted == "date":
        exact = ("date", "day", "calendardate", "cyclestarttime", "sleeponset", "summarydate")
        for target in exact:
            for column, norm in normalized.items():
                if norm == target:
                    return column
        return next((c for c, n in normalized.items() if "date" in n), None)

    smoothed = ("baseline", "avg7", "7d", "7day", "weekly", "week", "balance")
    candidates = [c for c, n in normalized.items() if not any(s in n for s in smoothed)]
    if wanted == "sleep_score":
        exact = [c for c in candidates if normalized[c] == "sleepscore"]
        loose = [c for c in candidates if "sleep" in normalized[c]
                 and ("score" in normalized[c] or "performance" in normalized[c])]
    else:
        exact = [c for c in candidates if normalized[c] in ("hrv", "hrvms")]
        loose = [c for c in candidates if "hrv" in normalized[c]
                 or "heartratevariability" in normalized[c]]
    return next(iter(exact + loose), None)


def _to_number(series: pd.Series) -> pd.Series:
    """Float coercion that tolerates ``85%``, ``64 ms`` and thousands separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def load_recovery(source: str | Path | bytes) -> pd.DataFrame:
    """Read a sleep/HRV CSV into ``DataFrame[date, sleep_score, hrv_ms]``, one row per day.

    ``source`` is a path or the raw bytes of an uploaded file.  Raises
    ``ValueError`` when no date column or no recognizable metric is found.
    """
    raw = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    raw.columns = [str(column).strip() for column in raw.columns]
    columns = list(raw.columns)

    date_column = _find_column(columns, "date")
    if date_column is None:
        raise ValueError(f"No date column found in {columns}.")
    metric_columns = {key: _find_column(columns, key) for key in RECOVERY_METRICS}
    if all(column is None for column in metric_columns.values()):
        raise ValueError(f"No sleep-score or HRV column found in {columns}.")

    stamp = pd.to_datetime(raw[date_column], format="mixed", errors="coerce")
    if isinstance(stamp.dtype, pd.DatetimeTZDtype):
        stamp = stamp.dt.tz_localize(None)
    evening = stamp.dt.hour >= EVENING_HOUR
    day = stamp.dt.normalize() + pd.to_timedelta(evening.astype(int), unit="D")

    tidy = pd.DataFrame({"date": day})
    for key, column in metric_columns.items():
        values = (_to_number(raw[column]) if column is not None
                  else pd.Series(np.nan, index=raw.index, dtype="float64"))
        low, high = _VALID_RANGE[key]
        tidy[key] = values.where(values.between(low, high))

    tidy = tidy.dropna(subset=["date"]).dropna(subset=list(RECOVERY_METRICS), how="all")
    daily = tidy.groupby("date", as_index=False)[list(RECOVERY_METRICS)].mean()
    return daily.sort_values("date").reset_index(drop=True)[RECOVERY_COLUMNS]


def join_recovery(recovery: pd.DataFrame, fitness: pd.DataFrame) -> pd.DataFrame:
    """Attach the training metrics in the module docstring to each recovery day.

    ``fitness`` is the frame from :func:`src.training_load.compute_fitness_freshness`.
    Only days inside the training calendar are kept, so a wearable file that
    runs past the Strava export does not pair readings with invented rest days.
    """
    daily = fitness.set_index("date")
    previous = daily.shift(1)
    training = pd.DataFrame({
        "load_prev_day": previous["load"],
        "atl_prev": previous["atl"],
        "tsb": daily["tsb"],
        "load_same_day": daily["load"],
    })
    joined = recovery.join(training, on="date", how="inner")
    return joined.reset_index(drop=True)[RECOVERY_COLUMNS + list(TRAINING_METRICS)]


def strength_label(r: float) -> str:
    """Plain-English size of a correlation coefficient."""
    if not np.isfinite(r):
        return "Not enough data"
    size = abs(r)
    if size < 0.1:
        return "No clear link"
    word = "Weak" if size < 0.3 else "Moderate" if size < 0.5 else "Strong"
    return f"{word} {'positive' if r > 0 else 'negative'}"


def correlation_table(joined: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """Pearson ``r`` for every recovery x training pair.

    Returns ``DataFrame[recovery, training, recovery_label, training_label, r, n,
    strength]``; ``r`` is NaN when fewer than ``min_pairs`` days overlap or a
    column has no variance.
    """
    rows = []
    for rec, rec_label in RECOVERY_METRICS.items():
        for train, train_label in TRAINING_METRICS.items():
            pair = joined[[rec, train]].dropna()
            n = len(pair)
            r = np.nan
            if n >= min_pairs and pair[rec].std() > 0 and pair[train].std() > 0:
                r = float(pair[rec].corr(pair[train]))
            rows.append({"recovery": rec, "training": train, "recovery_label": rec_label,
                         "training_label": train_label, "r": r, "n": n,
                         "strength": strength_label(r)})
    return pd.DataFrame(rows)


def rolling_baseline(joined: pd.DataFrame, column: str, days: int = 7) -> pd.Series:
    """Trailing ``days``-day mean of a recovery metric over calendar time."""
    series = joined.set_index("date")[column]
    return series.rolling(f"{days}D", min_periods=3).mean().reset_index(drop=True)


def synthetic_recovery(fitness: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Plausible sleep/HRV readings for the synthetic training history.

    HRV rises slowly with Fitness and dips the morning after a big day; sleep
    score dips a little after big days too.  Noise dominates, as it does in real
    wearable data, and ~8% of nights are missing.  Only ever paired with the
    synthetic runs -- never with a real export.
    """
    rng = np.random.default_rng(seed)
    previous = fitness.shift(1)
    load_prev = previous["load"].fillna(0.0).to_numpy()
    ctl_prev = previous["ctl"].fillna(0.0).to_numpy()
    n = len(fitness)

    hrv = 58.0 + 0.18 * ctl_prev - 0.07 * load_prev + rng.normal(0.0, 6.0, n)
    sleep = 80.0 - 0.05 * load_prev + rng.normal(0.0, 7.0, n)
    frame = pd.DataFrame({
        "date": fitness["date"].to_numpy(),
        "sleep_score": np.clip(np.round(sleep), 30, 100),
        "hrv_ms": np.clip(np.round(hrv), 15, 200),
    })
    missing = rng.random(n) < 0.08
    frame.loc[missing, ["sleep_score", "hrv_ms"]] = np.nan
    return frame.dropna(how="all", subset=list(RECOVERY_METRICS)).reset_index(drop=True)
