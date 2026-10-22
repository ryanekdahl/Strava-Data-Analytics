"""Tests for the Strava CSV loader and the tidy data contract."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.generate_sample import generate_sample
from src.loader import CONTRACT_COLUMNS, TIDY_COLUMNS, find_real_export, load_activities

HEADER = ("Activity ID,Activity Date,Activity Name,Activity Type,Elapsed Time,"
          "Distance,Moving Time,Average Heart Rate,Elevation Gain\n")


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> object:
    """A freshly generated synthetic export, independent of ``data/sample``."""
    path = tmp_path_factory.mktemp("sample") / "activities.csv"
    generate_sample(path, seed=1234, write_streams=False)
    return path


@pytest.fixture(scope="module")
def sample_runs(sample_csv) -> pd.DataFrame:
    return load_activities(sample_csv)


def write_csv(path, rows: list[str]):
    path.write_text(HEADER + "".join(row + "\n" for row in rows))
    return path


def test_synthetic_csv_loads_and_has_the_contract_columns(sample_runs):
    assert list(sample_runs.columns) == TIDY_COLUMNS
    assert TIDY_COLUMNS[:len(CONTRACT_COLUMNS)] == CONTRACT_COLUMNS
    assert len(sample_runs) > 200  # ~16 months at ~4 runs/week


def test_contract_dtypes(sample_runs):
    types = pd.api.types
    assert types.is_integer_dtype(sample_runs["activity_id"])
    assert types.is_datetime64_any_dtype(sample_runs["date"])
    assert sample_runs["date"].dt.tz is None  # tz-naive local time
    assert types.is_string_dtype(sample_runs["name"])
    assert types.is_integer_dtype(sample_runs["moving_time_s"])
    assert types.is_integer_dtype(sample_runs["elapsed_time_s"])
    for column in ("distance_km", "pace_min_per_km", "avg_hr", "elevation_gain_m"):
        assert types.is_float_dtype(sample_runs[column]), column


def test_synthetic_data_is_plausible_and_sorted(sample_runs):
    assert sample_runs["date"].is_monotonic_increasing
    assert sample_runs["distance_km"].between(1, 50).all()
    assert sample_runs["pace_min_per_km"].between(2.5, 10).all()
    assert (sample_runs["elapsed_time_s"] >= sample_runs["moving_time_s"]).all()
    # Pace is derived from moving time, not elapsed time.
    expected = (sample_runs["moving_time_s"] / 60.0) / sample_runs["distance_km"]
    pd.testing.assert_series_equal(sample_runs["pace_min_per_km"], expected,
                                   check_names=False)
    # ~16 months of history with the two planned multi-week gaps.
    span_days = (sample_runs["date"].max() - sample_runs["date"].min()).days
    assert 450 <= span_days <= 500


def test_synthetic_data_exercises_null_heart_rates(sample_runs):
    missing_share = sample_runs["avg_hr"].isna().mean()
    assert 0.03 < missing_share < 0.20
    assert sample_runs["avg_hr"].dropna().between(100, 200).all()


def test_zero_and_negative_distance_rows_are_dropped(tmp_path, caplog):
    path = write_csv(tmp_path / "activities.csv", [
        '1,"Jan 01, 2024, 7:00:00 AM",Good run,Run,3100,10000,3000,150,50',
        '2,"Jan 02, 2024, 7:00:00 AM",Accidental start,Run,0,0,0,,',
        '3,"Jan 03, 2024, 7:00:00 AM",Corrupt,Run,600,-5000,600,,',
        '4,"Jan 04, 2024, 7:00:00 AM",No moving time,Run,1800,5000,0,,',
    ])
    with caplog.at_level(logging.INFO, logger="src.loader"):
        runs = load_activities(path)

    assert list(runs["activity_id"]) == [1]
    assert "Dropped 3 of 4 runs" in caplog.text


def test_non_run_activities_are_filtered_out(tmp_path):
    path = write_csv(tmp_path / "activities.csv", [
        '1,"Jan 01, 2024, 7:00:00 AM",Morning run,Run,3100,10000,3000,150,50',
        '2,"Jan 02, 2024, 7:00:00 AM",Commute,Ride,3100,30000,3000,120,50',
        '3,"Jan 03, 2024, 7:00:00 AM",Dog walk,Walk,3100,4000,3000,95,10',
        '4,"Jan 04, 2024, 7:00:00 AM",Shouty caps,RUN,1200,4000,1200,,',
    ])
    runs = load_activities(path)
    assert list(runs["activity_id"]) == [1, 4]  # the type filter ignores case


def test_distances_in_meters_and_kilometres_both_land_in_km(tmp_path):
    meters = write_csv(tmp_path / "meters.csv", [
        '1,"Jan 01, 2024, 7:00:00 AM",Metres,Run,3100,"10,000",3000,150,50',
    ])
    kilometres = write_csv(tmp_path / "km.csv", [
        '1,"Jan 01, 2024, 7:00:00 AM",Kilometres,Run,3100,10.0,3000,150,50',
    ])
    assert load_activities(meters)["distance_km"].iloc[0] == pytest.approx(10.0)
    assert load_activities(kilometres)["distance_km"].iloc[0] == pytest.approx(10.0)


def test_durations_written_as_clock_strings_are_parsed(tmp_path):
    path = write_csv(tmp_path / "activities.csv", [
        '1,"Jan 01, 2024, 7:00:00 AM",Clock format,Run,0:52:00,10000,0:50:00,150,50',
    ])
    runs = load_activities(path)
    assert runs["moving_time_s"].iloc[0] == 3000
    assert runs["elapsed_time_s"].iloc[0] == 3120
    assert runs["pace_min_per_km"].iloc[0] == pytest.approx(5.0)


def test_missing_optional_columns_still_satisfy_the_contract(tmp_path):
    path = tmp_path / "minimal.csv"
    path.write_text(
        "Activity ID,Activity Date,Activity Type,Distance,Elapsed Time\n"
        '1,"Jan 01, 2024, 7:00:00 AM",Run,10000,3000\n'
    )
    runs = load_activities(path)

    assert list(runs.columns) == TIDY_COLUMNS
    assert runs["avg_hr"].isna().all()
    for column in ("max_hr", "avg_cadence_spm", "gap_pace_min_per_km", "relative_effort",
                   "gear", "filename"):
        assert runs[column].isna().all(), column
    assert runs["elevation_gain_m"].isna().all()
    # With no moving-time column, elapsed time stands in for it.
    assert runs["moving_time_s"].iloc[0] == 3000


def test_unreadable_export_raises_a_useful_error(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("Some Column,Another\n1,2\n")
    with pytest.raises(ValueError, match="no recognizable distance column"):
        load_activities(path)


def test_find_real_export(tmp_path):
    assert find_real_export(tmp_path) is None
    (tmp_path / "other.csv").write_text("x\n")
    assert find_real_export(tmp_path).name == "other.csv"
    (tmp_path / "activities.csv").write_text("x\n")
    assert find_real_export(tmp_path).name == "activities.csv"


EXTRA_HEADER = ("Activity ID,Activity Date,Activity Type,Distance,Moving Time,"
                "Max Heart Rate,Average Cadence,Grade Adjusted Distance,Relative Effort,"
                "Activity Gear,Filename\n")


def test_extra_export_columns_are_parsed(tmp_path):
    path = tmp_path / "activities.csv"
    path.write_text(EXTRA_HEADER + "".join([
        '1,"Jan 01, 2024, 7:00:00 AM",Run,10000,3000,171,84.5,10500,62,Pegasus 40,'
        'activities/1.fit.gz\n',
        '2,"Jan 02, 2024, 7:00:00 AM",Run,10000,3000,,86.0,,,,\n',
    ]))
    runs = load_activities(path)

    first, second = runs.iloc[0], runs.iloc[1]
    assert first["max_hr"] == 171
    assert first["avg_cadence_spm"] == pytest.approx(169.0)  # per-leg value doubled
    # 50 minutes over 10.5 grade-adjusted km.
    assert first["gap_pace_min_per_km"] == pytest.approx(50.0 / 10.5)
    assert first["relative_effort"] == 62
    assert first["gear"] == "Pegasus 40"
    assert first["filename"] == "activities/1.fit.gz"
    assert pd.isna(second["gap_pace_min_per_km"])
    assert pd.isna(second["gear"]) and pd.isna(second["filename"])  # blanks become missing


def test_cadence_already_in_steps_per_minute_is_left_alone(tmp_path):
    path = tmp_path / "activities.csv"
    path.write_text(EXTRA_HEADER
                    + '1,"Jan 01, 2024, 7:00:00 AM",Run,10000,3000,,172,,,,\n')
    assert load_activities(path)["avg_cadence_spm"].iloc[0] == pytest.approx(172.0)


def test_implausible_grade_adjusted_distance_is_ignored(tmp_path):
    path = tmp_path / "activities.csv"
    path.write_text(EXTRA_HEADER
                    + '1,"Jan 01, 2024, 7:00:00 AM",Run,10000,3000,,,30000,,,\n')
    assert pd.isna(load_activities(path)["gap_pace_min_per_km"].iloc[0])


def test_synthetic_sample_carries_the_extra_columns(sample_runs):
    assert sample_runs["filename"].notna().all()
    assert sample_runs["gear"].notna().mean() > 0.9
    assert sample_runs["avg_cadence_spm"].between(140, 200).all()
    # Hills make grade-adjusted pace faster than raw pace, never slower.
    assert (sample_runs["gap_pace_min_per_km"] <= sample_runs["pace_min_per_km"]).all()
