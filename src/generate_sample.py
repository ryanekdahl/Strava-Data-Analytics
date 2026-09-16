"""Generate a realistic synthetic Strava export so the dashboard is never empty.

The file written here mimics a real ``activities.csv``: Strava's column names,
distances in meters, ``H:MM:SS``-free integer seconds, a mix of activity types
and a few unusable rows.  That matters because the dashboard reads the sample
through :func:`src.loader.load_activities` -- the same code path a real export
takes -- so the loader is exercised by the sample rather than bypassed.

What the simulation models
--------------------------
* **A slow fitness trend.** Threshold pace improves linearly across the 16
  months with week-to-week noise, so "am I getting faster?" has a real answer.
* **Two multi-week gaps** (injury/holiday) followed by a decaying detraining
  penalty, which gives the Fitness/Freshness curve something to say.
* **A realistic run mix**: easy 5-8 km, tempo ~10 km, long 16-25 km and the
  occasional short 3-4 km blast, with rest weeks every fifth week.
* **Heart rate correlated with pace** (faster => higher HR) plus cardiac drift
  on long runs, with ~10% of values missing to prove nulls are handled.

Run it with ``python -m src.generate_sample``.  The RNG is seeded, so the shape
of the data is reproducible; dates are anchored to the day you run it so the
dashboard always looks current.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import SAMPLE_CSV

N_WEEKS = 70  # ~16 months
REST_WEEK_EVERY = 5
GAP_WEEKS = frozenset(range(22, 25)) | frozenset(range(47, 51))  # 3- and 4-week breaks

START_THRESHOLD_PACE = 5.30  # min/km at week 0
END_THRESHOLD_PACE = 4.75  # min/km at the final week
DETRAINING_PENALTY = 0.35  # min/km added on return from a gap, decaying weekly

RUN_NAMES = {
    "easy": ("Morning Run", "Evening Run", "Easy Miles", "Recovery Jog"),
    "tempo": ("Tempo Run", "Threshold Session", "Lunch Run"),
    "long": ("Long Run", "Sunday Long Run", "Weekend Long Run"),
    "short": ("Shakeout", "Parkrun", "Quick Blast"),
}

CSV_COLUMNS = [
    "Activity ID",
    "Activity Date",
    "Activity Name",
    "Activity Type",
    "Elapsed Time",
    "Distance",
    "Moving Time",
    "Average Heart Rate",
    "Elevation Gain",
]


def _week_plan(week: int, rng: np.random.Generator) -> list[str]:
    """Return the run types for one week of training."""
    if week in GAP_WEEKS:
        return []
    if week % REST_WEEK_EVERY == REST_WEEK_EVERY - 1:
        return ["easy", "easy"]
    plan = ["easy", "tempo", "easy"]
    plan.append("long" if rng.random() < 0.65 else "easy")
    if rng.random() < 0.25:
        plan.append("short")
    return plan


def _threshold_pace(week: int, rng: np.random.Generator, weeks_since_gap: int | None) -> float:
    """Estimated threshold pace for a week: linear gains, noise, detraining."""
    progress = week / max(N_WEEKS - 1, 1)
    pace = START_THRESHOLD_PACE + (END_THRESHOLD_PACE - START_THRESHOLD_PACE) * progress
    pace += rng.normal(0.0, 0.05)
    if weeks_since_gap is not None and weeks_since_gap < 4:
        pace += DETRAINING_PENALTY * (1.0 - weeks_since_gap / 4.0)
    return pace


def _distance_and_pace(
    run_type: str, threshold_pace: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Sample (distance_km, pace_min_per_km) for one run of a given type."""
    if run_type == "easy":
        distance = rng.uniform(5.0, 8.0)
        pace = threshold_pace + 1.00 + rng.normal(0.0, 0.12)
    elif run_type == "tempo":
        distance = float(np.clip(rng.normal(10.0, 0.8), 8.0, 12.0))
        pace = threshold_pace + 0.08 + rng.normal(0.0, 0.08)
    elif run_type == "long":
        distance = rng.uniform(16.0, 25.0)
        pace = threshold_pace + 0.75 + 0.02 * (distance - 16.0) + rng.normal(0.0, 0.12)
    else:  # short, run hard
        distance = rng.uniform(3.0, 4.5)
        pace = threshold_pace - 0.15 + rng.normal(0.0, 0.10)
    return round(distance, 2), max(pace, 3.0)


def _heart_rate(
    run_type: str, pace: float, threshold_pace: float, distance: float, rng: np.random.Generator
) -> float:
    """HR rises with relative intensity, plus cardiac drift on long runs."""
    relative_effort = (threshold_pace + 1.0) - pace  # 0 at easy pace, ~1 at threshold
    hr = 138.0 + 26.0 * relative_effort + rng.normal(0.0, 4.0)
    if run_type == "long":
        hr += 0.3 * max(distance - 16.0, 0.0)
    return float(np.clip(hr, 115.0, 192.0))


def _rows(end_date: date, rng: np.random.Generator) -> list[dict]:
    """Build every activity row, runs and decoys alike, oldest first."""
    first_monday = end_date - timedelta(days=end_date.weekday() + 7 * (N_WEEKS - 1))
    rows: list[dict] = []
    activity_id = 4_100_000_000
    weeks_since_gap: int | None = None

    for week in range(N_WEEKS):
        if week in GAP_WEEKS:
            weeks_since_gap = 0
            continue
        threshold_pace = _threshold_pace(week, rng, weeks_since_gap)
        week_start = first_monday + timedelta(weeks=week)
        plan = _week_plan(week, rng)
        for day_offset, run_type in zip(sorted(rng.choice(7, size=len(plan), replace=False)), plan):
            distance, pace = _distance_and_pace(run_type, threshold_pace, rng)
            moving_time_s = int(round(pace * distance * 60))
            elapsed_time_s = int(round(moving_time_s * (1.0 + rng.uniform(0.01, 0.08))))
            hr = _heart_rate(run_type, pace, threshold_pace, distance, rng)
            started = datetime.combine(
                week_start + timedelta(days=int(day_offset)),
                datetime.min.time(),
            ) + timedelta(hours=int(rng.integers(6, 20)), minutes=int(rng.integers(0, 60)))
            activity_id += 1
            rows.append(
                {
                    "Activity ID": activity_id,
                    "Activity Date": started,
                    "Activity Name": str(rng.choice(RUN_NAMES[run_type])),
                    "Activity Type": "Run",
                    "Elapsed Time": elapsed_time_s,
                    "Distance": round(distance * 1000, 1),  # Strava exports meters
                    "Moving Time": moving_time_s,
                    # ~10% of runs were recorded without a HR strap.
                    "Average Heart Rate": "" if rng.random() < 0.10 else round(hr),
                    "Elevation Gain": "" if rng.random() < 0.05 else round(distance * rng.uniform(4, 14)),
                }
            )
            # Occasional cross-training and a rare corrupt row, as real exports have.
            if rng.random() < 0.06:
                activity_id += 1
                rows.append(
                    {
                        "Activity ID": activity_id,
                        "Activity Date": started + timedelta(days=1, hours=2),
                        "Activity Name": "Recovery Spin",
                        "Activity Type": str(rng.choice(["Ride", "Walk", "Workout"])),
                        "Elapsed Time": int(rng.integers(1800, 5400)),
                        "Distance": round(rng.uniform(5000, 40000), 1),
                        "Moving Time": int(rng.integers(1500, 5000)),
                        "Average Heart Rate": round(rng.uniform(105, 140)),
                        "Elevation Gain": round(rng.uniform(10, 300)),
                    }
                )
            if rng.random() < 0.01:
                activity_id += 1
                rows.append(
                    {
                        "Activity ID": activity_id,
                        "Activity Date": started + timedelta(hours=3),
                        "Activity Name": "Accidental start",
                        "Activity Type": "Run",
                        "Elapsed Time": 0,
                        "Distance": 0.0,
                        "Moving Time": 0,
                        "Average Heart Rate": "",
                        "Elevation Gain": "",
                    }
                )
        weeks_since_gap = None if weeks_since_gap is None else weeks_since_gap + 1
    return rows


def generate_sample(
    out_path: str | Path = SAMPLE_CSV, seed: int = 20240921, end_date: date | None = None
) -> pd.DataFrame:
    """Write the synthetic export to ``out_path`` and return the rows written."""
    rng = np.random.default_rng(seed)
    end_date = end_date or date.today()
    frame = pd.DataFrame(_rows(end_date, rng), columns=CSV_COLUMNS)
    # The final training week can spill past today; a real export never does.
    frame = frame[frame["Activity Date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    frame = frame.sort_values("Activity Date").reset_index(drop=True)
    # Strava writes dates like "Sep 15, 2025, 7:31:04 AM".
    frame["Activity Date"] = frame["Activity Date"].dt.strftime("%b %d, %Y, %-I:%M:%S %p")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(SAMPLE_CSV), help="output CSV path")
    parser.add_argument("--seed", type=int, default=20240921, help="RNG seed")
    args = parser.parse_args()

    frame = generate_sample(args.out, seed=args.seed)
    runs = frame[frame["Activity Type"] == "Run"]
    print(f"Wrote {len(frame)} activities ({len(runs)} runs) to {args.out}")
    print(f"Date range: {frame['Activity Date'].iloc[0]} -> {frame['Activity Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
