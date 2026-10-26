"""Scoring non-run sessions on the runs' load axis, and the per-sport roll-ups."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.generate_sample import generate_sample
from src.loader import OTHER_COLUMNS, load_activities, load_other_activities
from src.sports import (
    DEFAULT_INTENSITY, METHOD_DEFAULT, METHOD_HR, METHOD_HR_POWER, METHOD_POWER, all_sessions,
    compute_other_load, sport_family, sport_summary, weekly_sport_hours,
)
from src.training_load import compute_activity_load, trimp


def _sessions(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    base = {
        "activity_id": np.arange(n), "date": pd.date_range("2025-01-06", periods=n),
        "name": [""] * n, "activity_type": ["Ride"] * n, "distance_km": [20.0] * n,
        "moving_time_s": [3600] * n, "avg_hr": [np.nan] * n, "max_hr": [np.nan] * n,
        "elevation_gain_m": [np.nan] * n, "avg_watts": [np.nan] * n,
        "weighted_watts": [np.nan] * n, "relative_effort": [np.nan] * n,
    }
    return pd.DataFrame({**base, **columns})[OTHER_COLUMNS]


def test_strava_types_fold_into_families():
    types = pd.Series(["Virtual Ride", "gravel ride ", "Weight Training", "Swim", "Kitesurf"])
    assert sport_family(types).tolist() == ["Ride", "Ride", "Strength", "Swim", "Other"]


def test_an_hour_with_no_hr_or_power_scores_the_default_intensity():
    scored = compute_other_load(_sessions(activity_type=["Ride", "Walk"]))
    assert scored["load"].tolist() == pytest.approx(
        [DEFAULT_INTENSITY["Ride"] ** 2 * 100, DEFAULT_INTENSITY["Walk"] ** 2 * 100])
    assert (scored["load_method"] == METHOD_DEFAULT).all()


def test_heart_rate_scores_with_the_same_trimp_as_runs():
    scored = compute_other_load(_sessions(avg_hr=[150.0]), hr_rest=50, hr_max=190)
    expected = trimp(pd.Series([3600]), pd.Series([150.0]), 50, 190).iloc[0]
    assert scored["load"].iloc[0] == pytest.approx(expected)
    assert scored["load_method"].iloc[0] == METHOD_HR


def test_an_hour_at_ftp_is_100_points_and_blends_with_heart_rate():
    sessions = _sessions(weighted_watts=[250.0, 250.0], avg_watts=[230.0, 230.0],
                         avg_hr=[np.nan, 160.0])
    scored = compute_other_load(sessions, ftp=250.0)
    assert scored["load"].iloc[0] == pytest.approx(100.0)
    assert scored["load"].iloc[1] == pytest.approx(0.5 * 100.0 + 0.5 * scored["load_hr"].iloc[1])
    assert scored["load_method"].tolist() == [METHOD_POWER, METHOD_HR_POWER]


def test_power_is_ignored_without_an_ftp_or_outside_cycling():
    no_ftp = compute_other_load(_sessions(weighted_watts=[250.0]))
    assert no_ftp["load_method"].iloc[0] == METHOD_DEFAULT
    rowing = compute_other_load(_sessions(activity_type=["Rowing"], avg_watts=[200.0]), ftp=250)
    assert rowing["load_method"].iloc[0] == METHOD_DEFAULT
    assert math.isnan(rowing["load_power"].iloc[0])


def test_summary_and_weekly_roll_ups_add_up():
    other = compute_other_load(_sessions(activity_type=["Ride", "Swim", "Ride"],
                                         moving_time_s=[3600, 1800, 7200]))
    runs = compute_activity_load(pd.DataFrame({
        "activity_id": [100], "date": [pd.Timestamp("2025-01-07")], "name": ["Run"],
        "distance_km": [10.0], "moving_time_s": [3000], "pace_min_per_km": [5.0],
        "gap_pace_min_per_km": [np.nan], "avg_hr": [np.nan],
    }))
    sessions = all_sessions(runs, other)
    assert sessions["sport"].tolist() == ["Ride", "Run", "Swim", "Ride"]
    summary = sport_summary(sessions).set_index("sport")
    assert summary.loc["Ride", "hours"] == pytest.approx(3.0)
    assert summary["load_share"].sum() == pytest.approx(1.0)
    weekly = weekly_sport_hours(sessions)
    assert weekly["hours"].sum() == pytest.approx(sessions["moving_time_s"].sum() / 3600)
    assert (pd.DatetimeIndex(weekly["week_start"]).dayofweek == 0).all()
    assert len(weekly) == weekly["week_start"].nunique() * 3  # zero-filled per sport


def test_empty_frames_score_and_roll_up_cleanly():
    scored = compute_other_load(_sessions(activity_type=[]))
    assert scored.empty and "load" in scored
    assert sport_summary(scored.assign(sport=[])).empty


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("sports") / "activities.csv"
    generate_sample(path, seed=1234, write_streams=False)
    return path


def test_the_sample_has_cross_training_of_every_scoring_kind(sample_csv):
    other = compute_other_load(load_other_activities(sample_csv), ftp=240.0)
    assert {"Ride", "Swim", "Walk", "Strength"} <= set(other["sport"])
    assert {METHOD_DEFAULT, METHOD_HR, METHOD_HR_POWER} <= set(other["load_method"])
    assert (other.loc[other["sport"] == "Strength", "distance_km"] == 0).all()
    runs = load_activities(sample_csv)
    assert not set(runs["activity_id"]) & set(other["activity_id"])
