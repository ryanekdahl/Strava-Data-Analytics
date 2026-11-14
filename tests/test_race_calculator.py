"""Tests for the interactive race-time calculator."""

from __future__ import annotations

import pytest

from src.performance import riegel_time
from src.race_calculator import (
    CALCULATOR_DISTANCES, calculator_predictions, parse_duration, split_times,
)


@pytest.mark.parametrize("text, seconds", [
    ("22:30", 1350.0),
    ("1:45:00", 6300.0),
    (" 3:05:09 ", 11109.0),
    ("45", 2700.0),          # bare number is minutes
    ("19:59.5", 1199.5),
])
def test_parse_duration_accepts_common_formats(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "5:75", "1:60:00", "0:00", "1:2:3:4", "-5:00"])
def test_parse_duration_rejects_invalid_input(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_calculator_matches_riegel_and_echoes_the_input_distance():
    table = calculator_predictions(5.0, 1200.0).set_index("distance")
    assert list(table.index) == list(CALCULATOR_DISTANCES)
    assert table.loc["5K", "predicted_time_s"] == pytest.approx(1200.0)
    assert table.loc["10K", "predicted_time_s"] == pytest.approx(riegel_time(1200.0, 5.0, 10.0))
    assert table["predicted_time_s"].is_monotonic_increasing


def test_a_higher_exponent_slows_longer_predictions_and_speeds_shorter_ones():
    gentle = calculator_predictions(10.0, 2700.0, exponent=1.03).set_index("distance")
    steep = calculator_predictions(10.0, 2700.0, exponent=1.10).set_index("distance")
    assert steep.loc["Marathon", "predicted_time_s"] > gentle.loc["Marathon", "predicted_time_s"]
    assert steep.loc["5K", "predicted_time_s"] < gentle.loc["5K", "predicted_time_s"]


def test_split_sheet_is_even_paced_and_ends_at_the_finish():
    sheet = split_times(10.0, 2400.0)
    assert len(sheet) == 10
    assert sheet["split_time_s"].tolist() == pytest.approx([240.0] * 10)
    assert sheet["elapsed_s"].iloc[-1] == pytest.approx(2400.0)
    assert sheet["elapsed"].iloc[4] == "20:00"


def test_split_sheet_in_miles_ends_with_a_partial_split():
    half = CALCULATOR_DISTANCES["Half Marathon"]
    sheet = split_times(half, 5400.0, split_km=1.609344)
    assert len(sheet) == 14                      # 13 full miles + 0.11 mi
    assert sheet["at_km"].iloc[-1] == pytest.approx(half)
    assert sheet["split_time_s"].iloc[-1] < sheet["split_time_s"].iloc[0]
    assert sheet["elapsed_s"].iloc[-1] == pytest.approx(5400.0)


def test_split_sheet_for_a_mile_race_in_miles_is_one_row():
    sheet = split_times(1.609344, 360.0, split_km=1.609344)
    assert len(sheet) == 1 and sheet["elapsed"].iloc[0] == "6:00"


def test_split_sheet_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        split_times(0.0, 100.0)
