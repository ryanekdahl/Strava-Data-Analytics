"""Tests for the interactive race-time calculator."""

from __future__ import annotations

import pytest

from src.performance import riegel_time
from src.race_calculator import CALCULATOR_DISTANCES, calculator_predictions, parse_duration


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
