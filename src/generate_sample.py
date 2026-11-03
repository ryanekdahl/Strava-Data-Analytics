"""Generate a realistic synthetic Strava export so the dashboard is never empty.

The file written here mimics a real ``activities.csv``: Strava's column names,
distances in meters, ``H:MM:SS``-free integer seconds, a mix of activity types
and a few unusable rows.  That matters because the dashboard reads the sample
through :func:`src.loader.load_activities` -- the same code path a real export
takes -- so the loader is exercised by the sample rather than bypassed.

What the simulation models
--------------------------
* **Fitness that follows training.** Threshold pace tracks a 42-day average of
  daily distance, and weekly volume ramps up across the 16 months, so the
  runner gets faster because they trained -- "am I getting faster?" has a real
  answer, and a model of training response has a real signal to find.
* **Two multi-week gaps** (injury/holiday).  Fitness decays through them, so
  the runner comes back slower and has to rebuild.
* **A realistic run mix**: easy 5-8 km, tempo ~10 km, long 16-25 km and the
  occasional short 3-4 km blast, with rest weeks every fifth week.
* **Heart rate correlated with pace** (faster => higher HR) plus cardiac drift
  on long runs, with ~10% of values missing to prove nulls are handled.
* **Short-term fatigue.**  Heart rate follows the *planned* effort, but the pace
  actually run is slowed when the 7-day average of distance runs ahead of the
  42-day one, so a big week makes the next runs slower at the same heart rate,
  while steady training costs nothing.  This is the fatigue term
  that :mod:`src.personal_models` tries to recover.
* **The columns recent exports carry**: max heart rate, cadence (per leg, as
  Strava writes it), grade-adjusted distance, Relative Effort, the shoe each run
  was logged in, and the ``Filename`` of its GPS recording.
* **Cross-training**: a gym session most weeks, weekend rides (some with a
  power meter, some on an indoor trainer), the odd swim and walk -- and more
  riding and swimming through the injury gaps, when the runner could not run.
  These come from their own RNG stream, so they did not change any run.
* **A GPS recording per run** (``activities/<id>.gpx.gz``, as in a real export):
  intervals hidden inside tempo sessions, fast finishes inside long runs, hills,
  a lagging heart rate and the odd stop at a crossing.  These recordings are what
  :mod:`src.streams` reads, so the sample exercises the same path a real export
  does.  Their randomness is seeded per activity, so adding them did not change a
  single value in the CSV.

Run it with ``python -m src.generate_sample``.  The RNG is seeded, so the shape
of the data is reproducible; dates are anchored to the day you run it so the
dashboard always looks current.
"""

from __future__ import annotations

import argparse
import gzip
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import SAMPLE_CSV

N_WEEKS = 70  # ~16 months
REST_WEEK_EVERY = 5
GAP_WEEKS = frozenset(range(22, 25)) | frozenset(range(47, 51))  # 3- and 4-week breaks

# Threshold pace = BASE - GAIN * (42-day average km per day): with the volume
# ramp below that runs from about 5:20 /km at the start to 4:45 /km at the end.
BASE_THRESHOLD_PACE = 6.05  # min/km for a runner with no recent training
FITNESS_GAIN = 0.18  # min/km faster per km/day of 42-day average distance
FITNESS_DAYS = 42
START_FITNESS_KM = 4.0  # km/day: the runner was already running before the export
VOLUME_RAMP = (0.85, 1.30)  # run-distance multiplier at the first and last week
FATIGUE_DAYS = 7  # time constant of the short-term fatigue average
FATIGUE_PACE_PER_KM = 0.08  # min/km slower per km/day the 7-day average exceeds the 42-day

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
    "Max Heart Rate",
    "Relative Effort",
    "Activity Gear",
    "Filename",
    "Moving Time",
    "Average Heart Rate",
    "Elevation Gain",
    "Average Cadence",
    "Grade Adjusted Distance",
    "Average Watts",
    "Weighted Average Power",
]

# Daily trainers are replaced every ~750 km; fast sessions use a racing shoe.
TRAINERS = ("Nike Pegasus 39", "Nike Pegasus 40", "Nike Pegasus 41")
TRAINER_LIFE_KM = 750.0
RACER = "Nike Vaporfly 3"

# The simulated athlete's true heart-rate settings, used for Relative Effort.
TRUE_HR_REST, TRUE_HR_MAX = 52.0, 192.0

# Cross-training: the simulated athlete's cycling FTP, and per-week chances of
# each session in a normal week and in a week off running.
TRUE_FTP = 240.0
CROSS_TRAINING = {  # sport: (chance in a training week, chance in a gap week)
    "Weight Training": (0.55, 0.70),
    "Ride": (0.40, 0.95),
    "Swim": (0.12, 0.80),
    "Walk": (0.20, 0.50),
}

STREAM_SAMPLE_S = 5.0  # seconds between GPS points, like a watch's smart recording
START_LAT, START_LON = 51.5007, -0.1246
METERS_PER_DEGREE = 6_371_008.8 * math.pi / 180.0  # same sphere the haversine uses


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


def _threshold_pace(fitness_km: float, rng: np.random.Generator) -> float:
    """Threshold pace for a week: set by recent training, plus noise."""
    return BASE_THRESHOLD_PACE - FITNESS_GAIN * fitness_km + rng.normal(0.0, 0.05)


class _Ewma:
    """A daily exponentially weighted average of distance, advanced day by day."""

    def __init__(self, days: int, start: float = 0.0) -> None:
        self.days, self.value, self.day = days, start, 0

    def at(self, day: int) -> float:
        self.value *= (1.0 - 1.0 / self.days) ** (day - self.day)
        self.day = day
        return self.value

    def add(self, km: float) -> None:
        self.value += km / self.days


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


def _extras(
    run_type: str, distance: float, moving_time_s: int, hr: float, has_hr: bool,
    gain: float | str, activity_id: int, trainer_km: float, aux: np.random.Generator,
) -> dict:
    """The newer export columns for one run, drawn from its own RNG stream."""
    hrr = (hr - TRUE_HR_REST) / (TRUE_HR_MAX - TRUE_HR_REST)
    effort = moving_time_s / 60.0 * 1.2 * math.exp(3.0 * (hrr - 0.65)) * aux.lognormal(0, 0.1)
    speed = distance * 1000.0 / moving_time_s
    steps_per_minute = 158.0 + 9.0 * (speed - 2.8) + aux.normal(0.0, 2.0)
    if run_type in ("tempo", "short") and aux.random() < 0.7:
        gear = RACER
    elif aux.random() < 0.03:
        gear = ""  # forgot to pick a shoe
    else:
        gear = TRAINERS[min(int(trainer_km // TRAINER_LIFE_KM), len(TRAINERS) - 1)]
    climb = 0.0 if gain == "" else float(gain)
    return {
        "Max Heart Rate": round(hr + aux.uniform(6, 18)) if has_hr else "",
        "Relative Effort": round(effort) if has_hr else "",
        "Activity Gear": gear,
        "Filename": f"activities/{activity_id}.gpx.gz",
        "Average Cadence": round(steps_per_minute / 2.0, 1),  # Strava logs per leg
        "Grade Adjusted Distance": round(distance * 1000.0 + 5.5 * climb, 1),
    }


def _rows(end_date: date, rng: np.random.Generator, seed: int = 0) -> list[dict]:
    """Build every activity row, runs and decoys alike, oldest first.

    Keys starting with ``_`` carry what the GPS recordings need to know (run
    type, that week's threshold pace) and never reach the CSV.  The newer
    columns draw from a per-activity RNG, leaving ``rng``'s sequence -- and so
    every original column -- exactly as it was.
    """
    first_monday = end_date - timedelta(days=end_date.weekday() + 7 * (N_WEEKS - 1))
    rows: list[dict] = []
    activity_id = 4_100_000_000
    trainer_km = 0.0
    fitness = _Ewma(FITNESS_DAYS, START_FITNESS_KM)
    fatigue = _Ewma(FATIGUE_DAYS)

    for week in range(N_WEEKS):
        if week in GAP_WEEKS:
            continue
        threshold_pace = _threshold_pace(fitness.at(week * 7), rng)
        progress = week / max(N_WEEKS - 1, 1)
        volume = VOLUME_RAMP[0] + (VOLUME_RAMP[1] - VOLUME_RAMP[0]) * progress
        week_start = first_monday + timedelta(weeks=week)
        plan = _week_plan(week, rng)
        for day_offset, run_type in zip(sorted(rng.choice(7, size=len(plan), replace=False)), plan):
            distance, pace = _distance_and_pace(run_type, threshold_pace, rng)
            distance = round(distance * volume, 2)
            day = week * 7 + int(day_offset)
            overload = fatigue.at(day) - fitness.at(day)
            actual_pace = pace + FATIGUE_PACE_PER_KM * max(overload, 0.0)
            fatigue.add(distance)
            fitness.add(distance)
            moving_time_s = int(round(actual_pace * distance * 60))
            elapsed_time_s = int(round(moving_time_s * (1.0 + rng.uniform(0.01, 0.08))))
            hr = _heart_rate(run_type, pace, threshold_pace, distance, rng)
            started = datetime.combine(
                week_start + timedelta(days=int(day_offset)),
                datetime.min.time(),
            ) + timedelta(hours=int(rng.integers(6, 20)), minutes=int(rng.integers(0, 60)))
            activity_id += 1
            row = {
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
                "_run_type": run_type,
                "_threshold_pace": threshold_pace,
            }
            aux = np.random.default_rng([seed, activity_id])
            row.update(_extras(run_type, distance, moving_time_s, hr,
                               row["Average Heart Rate"] != "", row["Elevation Gain"],
                               activity_id, trainer_km, aux))
            if row["Activity Gear"] in TRAINERS:
                trainer_km += distance
            rows.append(row)
            # Occasional cross-training and a rare corrupt row, as real exports have.
            if rng.random() < 0.06:
                activity_id += 1
                kind = str(rng.choice(["Ride", "Walk", "Workout"]))
                elapsed = int(rng.integers(1800, 5400))
                distance_m = round(rng.uniform(5000, 40000), 1)
                moving = int(rng.integers(1500, 5000))
                if kind == "Walk":  # same draws as ever, at a walking speed
                    distance_m = round(moving * 1.4, 1)
                elif kind == "Workout":
                    distance_m = 0.0
                rows.append(
                    {
                        "Activity ID": activity_id,
                        "Activity Date": started + timedelta(days=1, hours=2),
                        "Activity Name": {"Ride": "Recovery Spin", "Walk": "Recovery Walk",
                                          "Workout": "Mobility Session"}[kind],
                        "Activity Type": kind,
                        "Elapsed Time": elapsed,
                        "Distance": distance_m,
                        "Moving Time": moving,
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
    return rows


def _cross_training_rows(end_date: date, seed: int) -> list[dict]:
    """Gym sessions, rides, swims and walks, from an RNG of their own.

    More riding and swimming in the weeks the runner was not running, as an
    injured runner does.  Heart rate is sometimes missing and some rides carry
    power, so every branch of :func:`src.sports.compute_other_load` is used.
    """
    aux = np.random.default_rng([seed, 3])
    first_monday = end_date - timedelta(days=end_date.weekday() + 7 * (N_WEEKS - 1))
    activity_id = 4_200_000_000
    rows: list[dict] = []
    for week in range(N_WEEKS):
        gap = week in GAP_WEEKS
        week_start = first_monday + timedelta(weeks=week)
        for sport, chances in CROSS_TRAINING.items():
            for _ in range(2 if gap and sport in ("Ride", "Swim") else 1):
                if aux.random() >= chances[gap]:
                    continue
                activity_id += 1
                day = int(aux.choice([5, 6]) if sport == "Ride" and not gap
                          else aux.integers(0, 7))
                started = datetime.combine(week_start + timedelta(days=day),
                                           datetime.min.time()) + timedelta(
                    hours=int(aux.integers(7, 19)), minutes=int(aux.integers(0, 60)))
                rows.append({"Activity ID": activity_id, "Activity Date": started,
                             **_cross_session(sport, gap, aux)})
    return rows


def _cross_session(sport: str, gap: bool, aux: np.random.Generator) -> dict:
    """The CSV columns for one cross-training session of ``sport``."""
    hr_mean, watts = None, None
    if sport == "Weight Training":
        moving = int(aux.integers(40, 70)) * 60
        distance, climb, name = 0.0, "", str(aux.choice(["Gym - Legs", "Gym - Upper Body",
                                                         "Strength & Core"]))
        hr_mean = aux.uniform(100, 120)
    elif sport == "Ride":
        indoor = aux.random() < 0.3
        moving = int(aux.integers(45, 90) if indoor else aux.integers(60, 180)) * 60
        intensity = aux.uniform(0.62, 0.82) if not gap else aux.uniform(0.65, 0.88)
        speed_kmh = 20.0 + 14.0 * intensity + aux.normal(0.0, 1.5)
        distance = moving / 3600.0 * speed_kmh * 1000.0
        climb = 0.0 if indoor else round(distance / 1000.0 * aux.uniform(4, 15))
        name = "Zwift - Watopia" if indoor else str(aux.choice(["Morning Ride", "Weekend Ride",
                                                                "Coffee Ride"]))
        hr_mean = 112.0 + 55.0 * (intensity - 0.55)
        if indoor or aux.random() < 0.5:
            watts = TRUE_FTP * intensity
        sport = "Virtual Ride" if indoor else sport
    elif sport == "Swim":
        moving = int(aux.integers(30, 55)) * 60
        distance = moving / 60.0 * aux.uniform(40, 52)  # metres a minute
        climb, name = "", "Pool Swim"
        hr_mean = aux.uniform(120, 145) if aux.random() < 0.3 else None  # wrist HR in water
    else:  # Walk
        moving = int(aux.integers(30, 75)) * 60
        distance = moving * aux.uniform(1.25, 1.5)
        climb, name = round(distance / 1000.0 * aux.uniform(2, 10)), "Lunch Walk"
        hr_mean = aux.uniform(90, 108)
    has_hr = hr_mean is not None and aux.random() > 0.1
    return {
        "Activity Name": name,
        "Activity Type": sport,
        "Elapsed Time": int(moving * aux.uniform(1.0, 1.15)),
        "Distance": round(distance, 1),
        "Moving Time": moving,
        "Average Heart Rate": round(hr_mean) if has_hr else "",
        "Max Heart Rate": round(hr_mean + aux.uniform(15, 35)) if has_hr else "",
        "Elevation Gain": climb,
        "Average Watts": round(watts * aux.uniform(0.88, 0.95)) if watts else "",
        "Weighted Average Power": round(watts) if watts else "",
    }


def sample_is_current(path: str | Path = SAMPLE_CSV) -> bool:
    """Whether ``path`` was written by this version of the generator (same columns)."""
    try:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline().strip()
    except OSError:
        return False
    return header == ",".join(CSV_COLUMNS)


def generate_sample(
    out_path: str | Path = SAMPLE_CSV,
    seed: int = 20240921,
    end_date: date | None = None,
    write_streams: bool = True,
) -> pd.DataFrame:
    """Write the synthetic export to ``out_path`` and return the rows written.

    With ``write_streams`` the GPS recordings are written to an ``activities/``
    folder beside the CSV, exactly where a real export keeps them.
    """
    rng = np.random.default_rng(seed)
    end_date = end_date or date.today()
    frame = pd.DataFrame(_rows(end_date, rng, seed) + _cross_training_rows(end_date, seed))
    # The final training week can spill past today; a real export never does.
    frame = frame[frame["Activity Date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    frame = frame.sort_values("Activity Date").reset_index(drop=True)
    extras = ["Max Heart Rate", "Relative Effort", "Activity Gear", "Filename",
              "Average Cadence", "Grade Adjusted Distance", "Average Watts",
              "Weighted Average Power"]
    frame[extras] = frame[extras].astype(object).fillna("")  # decoys never got them
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if write_streams:
        _write_streams(frame, out_path.parent, seed)

    # Strava writes dates like "Sep 15, 2025, 7:31:04 AM".
    frame["Activity Date"] = frame["Activity Date"].dt.strftime("%b %d, %Y, %-I:%M:%S %p")
    frame = frame[CSV_COLUMNS]
    frame.to_csv(out_path, index=False)
    return frame


# --------------------------------------------------------------------------
# Friends, for the Head to head tab
# --------------------------------------------------------------------------

# name: (seed, pace factor, volume factor, share of your long runs they join).
# Alex is quicker on less volume; Sam is slower but runs further.
SAMPLE_FRIENDS = {"Alex": (7, 0.93, 0.80, 0.35), "Sam": (11, 1.07, 1.20, 0.20)}
_GROUP_RUN_MIN_M = 14_000.0
_DATE_FORMAT = "%b %d, %Y, %-I:%M:%S %p"
_SCALED_BY_VOLUME = ("Distance", "Grade Adjusted Distance", "Elevation Gain")
_SCALED_BY_TIME = ("Moving Time", "Elapsed Time")


def friends_are_current(you_csv: str | Path = SAMPLE_CSV) -> bool:
    """Whether every sample friend's export exists and is newer than the sample itself."""
    you_csv = Path(you_csv)
    paths = [you_csv.parent / "friends" / f"{name.lower()}.csv" for name in SAMPLE_FRIENDS]
    return you_csv.is_file() and all(
        path.is_file() and path.stat().st_mtime >= you_csv.stat().st_mtime for path in paths)


def generate_friends(you_csv: str | Path = SAMPLE_CSV,
                     out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write a synthetic export for each of :data:`SAMPLE_FRIENDS` and return their paths.

    Each friend is the sample runner re-simulated with their own seed, then
    made faster or slower (``pace factor`` scales every run's time) and bigger
    or smaller in volume.  They also join a share of your long runs -- same
    start within a few minutes, much the same distance, at a pace pulled
    toward yours -- so the tab's "runs together" view has something to find.
    No GPS recordings are written for friends.
    """
    you_csv = Path(you_csv)
    out_dir = Path(out_dir) if out_dir is not None else you_csv.parent / "friends"
    you = pd.read_csv(you_csv)
    you["Activity Date"] = pd.to_datetime(you["Activity Date"], format="mixed")
    long_runs = you[(you["Activity Type"] == "Run") & (you["Distance"] >= _GROUP_RUN_MIN_M)]
    paths = {}
    for name, (seed, pace, volume, join) in SAMPLE_FRIENDS.items():
        path = out_dir / f"{name.lower()}.csv"
        frame = generate_sample(path, seed=seed, end_date=you["Activity Date"].max().date(),
                                write_streams=False)
        frame["Activity Date"] = pd.to_datetime(frame["Activity Date"], format="mixed")
        frame["Activity ID"] += seed * 10_000_000
        runs = frame["Activity Type"] == "Run"
        for column in _SCALED_BY_VOLUME + _SCALED_BY_TIME:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        frame.loc[runs, list(_SCALED_BY_VOLUME)] *= volume
        frame.loc[runs, list(_SCALED_BY_TIME)] *= volume * pace
        frame[list(_SCALED_BY_TIME)] = frame[list(_SCALED_BY_TIME)].round()
        frame["Filename"] = ""

        rng = np.random.default_rng([seed, 1])
        joined = long_runs[rng.random(len(long_runs)) < join].copy()
        stretch = rng.uniform(0.97, 1.03, len(joined))
        together = 1.0 + 0.3 * (pace - 1.0) + rng.normal(0.0, 0.01, len(joined))
        joined["Activity ID"] = frame["Activity ID"].max() + 1 + np.arange(len(joined))
        joined["Activity Date"] += pd.to_timedelta(rng.integers(-4, 5, len(joined)), unit="min")
        joined["Activity Name"] = "Group Long Run"
        for column in _SCALED_BY_VOLUME:
            joined[column] = pd.to_numeric(joined[column], errors="coerce") * stretch
        for column in _SCALED_BY_TIME:
            joined[column] = (joined[column] * stretch * together).round()
        joined[["Activity Gear", "Filename"]] = ""

        frame = pd.concat([frame, joined[CSV_COLUMNS]]).sort_values("Activity Date")
        frame["Activity Date"] = frame["Activity Date"].dt.strftime(_DATE_FORMAT)
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


# --------------------------------------------------------------------------
# GPS recordings
# --------------------------------------------------------------------------

def _speed_profile(run_type: str, distance_km: float, aux: np.random.Generator):
    """Relative speed as a function of km covered, plus a label for the shape.

    Absolute speeds are set later so the whole run takes exactly its moving
    time; only the *shape* is decided here.
    """
    wobble_phase = aux.uniform(0, 2 * math.pi)

    def wobble(x):
        return 1.0 + 0.025 * math.sin(2 * math.pi * x / 1.7 + wobble_phase)

    if run_type == "tempo" and aux.random() < 0.5:
        rep, jog = 0.8, 0.4  # 800 m reps with 400 m recoveries

        def profile(x):
            if x < 2.0 or x > distance_km - 1.5:
                return 0.86 * wobble(x)
            return 1.18 if (x - 2.0) % (rep + jog) < rep else 0.78
        return profile
    if run_type == "tempo":
        def profile(x):
            return 0.86 * wobble(x) if x < 2.0 or x > distance_km - 1.5 else 1.05
        return profile
    if run_type == "long":
        finish = min(5.0, distance_km / 3.0) if aux.random() < 0.6 else 0.0

        def profile(x):
            return 1.13 if x > distance_km - finish else wobble(x)
        return profile
    if run_type == "short":
        return lambda x: 0.97 + 0.06 * x / max(distance_km, 1.0)
    return wobble


def _simulate(row: pd.Series, aux: np.random.Generator) -> dict[str, np.ndarray]:
    """Time, position, elevation, heart rate and cadence samples for one run."""
    distance_m = float(row["Distance"])
    moving_s = float(row["Moving Time"])
    distance_km = distance_m / 1000.0
    profile = _speed_profile(row["_run_type"], distance_km, aux)

    # Scale so that integrating 1 / speed over the route gives the moving time.
    grid = np.linspace(0.0, distance_km, 2000)
    inverse = np.array([1.0 / profile(x) for x in grid])
    v0 = distance_m * float(np.mean(inverse)) / moving_s

    t, x = [0.0], [0.0]
    while x[-1] < distance_m:
        step = v0 * profile(x[-1] / 1000.0) * STREAM_SAMPLE_S
        if x[-1] + step >= distance_m:
            t.append(t[-1] + STREAM_SAMPLE_S * (distance_m - x[-1]) / step)
            x.append(distance_m)
        else:
            t.append(t[-1] + STREAM_SAMPLE_S)
            x.append(x[-1] + step)
    t_arr, x_arr = np.array(t), np.array(x)
    speed = np.gradient(x_arr, t_arr)

    # One or two stops (traffic lights) account for elapsed minus moving time.
    stopped = float(row["Elapsed Time"]) - moving_s
    if stopped > 5 and len(t_arr) > 20:
        stops = sorted(aux.choice(np.arange(5, len(t_arr) - 5), size=2 if stopped > 90 else 1,
                              replace=False))
        shift = np.zeros(len(t_arr))
        for i, at in enumerate(stops):
            shift[at:] += stopped / len(stops)
        t_arr = t_arr + shift

    # Rolling hills sized so the climb matches the CSV's elevation gain.
    gain = pd.to_numeric(row["Elevation Gain"], errors="coerce")
    gain = distance_km * 8.0 if pd.isna(gain) else float(gain)
    wavelength = aux.uniform(2.0, 4.0)
    amplitude = gain / max(2.0 * distance_km / wavelength, 1.0) / 2.0
    phase = aux.uniform(0, 2 * math.pi)
    altitude = 30.0 + amplitude * (1 + np.sin(2 * math.pi * x_arr / 1000.0 / wavelength + phase))

    # A wandering route: heading drifts, position follows the distance covered.
    heading = np.cumsum(aux.normal(0.0, 0.08, len(x_arr))) + aux.uniform(0, 2 * math.pi)
    step = np.diff(x_arr, prepend=0.0)
    lat = START_LAT + np.cumsum(step * np.cos(heading)) / METERS_PER_DEGREE
    lon = START_LON + np.cumsum(step * np.sin(heading)) / (
        METERS_PER_DEGREE * np.cos(np.radians(lat)))

    # Heart rate chases a pace-driven target with a ~25 s lag, then is shifted
    # so its time-weighted mean is exactly the CSV's average heart rate.
    hr = None
    avg_hr = pd.to_numeric(row["Average Heart Rate"], errors="coerce")
    if not pd.isna(avg_hr):
        threshold = float(row["_threshold_pace"])
        pace = 1000.0 / 60.0 / np.maximum(speed, 0.5)
        target = 138.0 + 26.0 * ((threshold + 1.0) - pace)
        if row["_run_type"] == "long":
            target += 0.3 * np.maximum(x_arr / 1000.0 - 16.0, 0.0)
        target += 0.4 * x_arr / 1000.0  # gentle cardiac drift on every run
        hr = np.empty(len(target))
        hr[0] = 95.0
        for i in range(1, len(target)):
            dt = t_arr[i] - t_arr[i - 1]
            hr[i] = hr[i - 1] + (target[i] - hr[i - 1]) * (1.0 - math.exp(-dt / 25.0))
        hr += aux.normal(0.0, 1.0, len(hr))
        weights = np.clip(np.diff(t_arr, append=t_arr[-1]), 0, 10)
        hr = np.clip(hr + (avg_hr - np.average(hr, weights=weights)), 80, 205)

    cadence = (158.0 + 9.0 * (speed - 2.8) + aux.normal(0.0, 2.0, len(speed))) / 2.0
    return {"t": t_arr, "lat": lat, "lon": lon, "ele": altitude, "hr": hr, "cad": cadence}


def _gpx(start: datetime, name: str, samples: dict[str, np.ndarray]) -> str:
    """Render samples as a GPX 1.1 track with Garmin's heart-rate extension."""
    points = []
    for i, offset in enumerate(samples["t"]):
        stamp = (start + timedelta(seconds=float(offset))).strftime("%Y-%m-%dT%H:%M:%SZ")
        extension = f"<gpxtpx:cad>{samples['cad'][i]:.0f}</gpxtpx:cad>"
        if samples["hr"] is not None:
            extension = f"<gpxtpx:hr>{samples['hr'][i]:.0f}</gpxtpx:hr>" + extension
        points.append(
            f'<trkpt lat="{samples["lat"][i]:.7f}" lon="{samples["lon"][i]:.7f}">'
            f"<ele>{samples['ele'][i]:.1f}</ele><time>{stamp}</time>"
            f"<extensions><gpxtpx:TrackPointExtension>{extension}"
            f"</gpxtpx:TrackPointExtension></extensions></trkpt>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="strava-run-analytics sample" '
        'xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
        f"<trk><name>{name}</name><type>running</type><trkseg>"
        + "".join(points) + "</trkseg></trk></gpx>\n"
    )


def _write_streams(frame: pd.DataFrame, out_dir: Path, seed: int) -> None:
    """Write ``activities/<id>.gpx.gz`` for every run that names one."""
    runs = frame[(frame["Activity Type"] == "Run") & (frame["Filename"] != "")]
    (out_dir / "activities").mkdir(parents=True, exist_ok=True)
    for _, row in runs.iterrows():
        if float(row["Distance"]) <= 0:
            continue
        # A different stream from the CSV extras' RNG, but just as reproducible.
        aux = np.random.default_rng([seed, int(row["Activity ID"]), 1])
        samples = _simulate(row, aux)
        text = _gpx(row["Activity Date"].to_pydatetime(), row["Activity Name"], samples)
        path = out_dir / row["Filename"]
        path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=6, mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(SAMPLE_CSV), help="output CSV path")
    parser.add_argument("--seed", type=int, default=20240921, help="RNG seed")
    parser.add_argument("--no-streams", action="store_true",
                        help="skip writing the per-run GPS recordings")
    args = parser.parse_args()

    frame = generate_sample(args.out, seed=args.seed, write_streams=not args.no_streams)
    runs = frame[frame["Activity Type"] == "Run"]
    print(f"Wrote {len(frame)} activities ({len(runs)} runs, "
          f"{len(frame) - len(runs)} other sports) to {args.out}")
    print(f"Date range: {frame['Activity Date'].iloc[0]} -> {frame['Activity Date'].iloc[-1]}")
    if Path(args.out).resolve() == SAMPLE_CSV.resolve():
        friends = generate_friends(args.out)
        print(f"Wrote {len(friends)} sample friends ({', '.join(friends)}) for Head to head")


if __name__ == "__main__":
    main()
