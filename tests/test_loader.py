"""Tests for the Strava CSV loader and the tidy data contract."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.generate_sample import generate_sample
from src.loader import CONTRACT_COLUMNS, find_real_export, load_activities

HEADER = ("Activity ID,Activity Date,Activity Name,Activity Type,Elapsed Time,"
          "Distance,Moving Time,Average Heart Rate,Elevation Gain\n")


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> object:
    """A freshly generated synthetic export, independent of ``data/sample``."""
    path = tmp_path_factory.mktemp("sample") / "activities.csv"
    generate_sample(path, seed=1234)
    return path


@pytest.fixture(scope="module")
def sample_runs(sample_csv) -> pd.DataFrame:
    return load_activities(sample_csv)


def write_csv(path, rows: list[str]):
    path.write_text(HEADER + "".join(row + "\n" for row in rows))
    return path


def test_synthetic_csv_loads_and_has_the_contract_columns(sample_runs):
    assert list(sample_runs.columns) == CONTRACT_COLUMNS
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

    assert list(runs.columns) == CONTRACT_COLUMNS
    assert runs["avg_hr"].isna().all()
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
