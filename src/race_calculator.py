"""Interactive race-time calculator: type a result, get Riegel predictions.

The dashboard's own predictions start from the best run in the export; this
module lets the user supply the reference effort instead -- a recent race, a
time trial, or a goal time -- and see what it implies at other distances.  The
model is the same ``T2 = T1 * (D2 / D1) ** exponent`` from :mod:`src.performance`,
with the exponent exposed so the user can see how sensitive a marathon
prediction is to Riegel's fatigue factor.
"""

from __future__ import annotations

import pandas as pd

from src.performance import RIEGEL_EXPONENT, Effort, format_duration, predict_races

# Distances the user can enter a result for, and that predictions are made at.
CALCULATOR_DISTANCES: dict[str, float] = {
    "1 mile": 1.609344,
    "3K": 3.0,
    "5K": 5.0,
    "10K": 10.0,
    "15K": 15.0,
    "Half Marathon": 21.0975,
    "Marathon": 42.195,
}

# Range offered on the exponent slider: 1.0 means pace never fades with
# distance; values near 1.15 describe runners whose endurance lags their speed.
EXPONENT_RANGE = (1.01, 1.15)


def parse_duration(text: str) -> float:
    """Parse ``H:MM:SS``, ``MM:SS`` or plain minutes into seconds.

    Every field after the first must be below 60, so ``5:75`` is rejected
    rather than silently read as 6:15.  Raises :class:`ValueError` for anything
    that is not a positive duration.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("enter a time such as 22:30 or 1:45:00")
    parts = cleaned.split(":")
    if len(parts) > 3:
        raise ValueError(f"too many ':' separators in {text!r}")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"could not read {text!r} as a time") from None
    if any(value < 0 for value in values) or any(value >= 60 for value in values[1:]):
        raise ValueError(f"minutes and seconds must be between 0 and 59 in {text!r}")

    if len(values) == 1:
        seconds = values[0] * 60.0
    else:
        seconds = 0.0
        for value in values:
            seconds = seconds * 60.0 + value
    if seconds <= 0:
        raise ValueError("time must be greater than zero")
    return seconds


def calculator_predictions(
    distance_km: float,
    time_s: float,
    exponent: float = RIEGEL_EXPONENT,
    distances: dict[str, float] = CALCULATOR_DISTANCES,
) -> pd.DataFrame:
    """Riegel predictions at every calculator distance from one entered result.

    The entered distance is kept in the table (where the prediction equals the
    input) so the user can sanity-check the pace column against what they ran.
    """
    return predict_races(Effort(distance_km, time_s), distances, exponent)


def split_times(distance_km: float, time_s: float, split_km: float = 1.0) -> pd.DataFrame:
    """Even-pace split sheet: the clock time to hit at every split marker.

    Markers fall every ``split_km`` kilometres (1.0 for kilometres, 1.609344
    for miles), and the race distance itself is always the last row, so a
    half marathon in miles ends with a 0.1 mi split.  Returns
    ``DataFrame[split, at_km, split_time_s, elapsed_s, elapsed]``.
    """
    if distance_km <= 0 or time_s <= 0 or split_km <= 0:
        raise ValueError("distance, time and split length must be positive")
    pace_s_per_km = time_s / distance_km
    rows = []
    marker, number = 0.0, 0
    while marker < distance_km - 1e-9:
        number += 1
        # Snap markers within a metre of the finish to the finish itself.
        nxt = number * split_km
        nxt = distance_km if nxt > distance_km - 1e-3 else nxt
        elapsed = nxt * pace_s_per_km
        rows.append({
            "split": number, "at_km": nxt,
            "split_time_s": (nxt - marker) * pace_s_per_km,
            "elapsed_s": elapsed, "elapsed": format_duration(elapsed),
        })
        marker = nxt
    return pd.DataFrame(rows)
