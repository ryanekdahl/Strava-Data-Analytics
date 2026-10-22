"""Tests for kilometre / mile display conversion."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.units import IMPERIAL, KM_PER_MILE, METRIC, UNIT_SYSTEMS


def test_metric_is_the_identity():
    assert METRIC.distance(10.0) == 10.0
    assert METRIC.pace(5.0) == 5.0
    assert METRIC.format_pace(5.0) == "5:00 /km"
    assert METRIC.format_distance(12.345) == "12.3 km"
    assert METRIC.format_elevation(12.4) == "+12 m"


def test_miles_convert_distance_and_pace():
    assert IMPERIAL.distance(KM_PER_MILE) == pytest.approx(1.0)
    assert IMPERIAL.format_distance(42.195) == "26.2 mi"
    # 5:00 /km is 8:02.7 per mile, which rounds to 8:03.
    assert IMPERIAL.pace(5.0) == pytest.approx(5.0 * KM_PER_MILE)
    assert IMPERIAL.format_pace(5.0) == "8:03 /mi"
    assert IMPERIAL.format_elevation(-30.48) == "-100 ft"


def test_conversions_work_on_series_and_missing_values():
    paces = IMPERIAL.pace(pd.Series([4.0, 6.0]))
    assert paces.tolist() == pytest.approx([4.0 * KM_PER_MILE, 6.0 * KM_PER_MILE])
    for units in UNIT_SYSTEMS.values():
        assert units.format_pace(math.nan) == "--"
        assert units.format_pace(None) == "--"
        assert units.format_distance(math.nan) == "--"
        assert units.format_elevation(math.nan) == "--"


def test_split_length_matches_the_unit():
    assert METRIC.split_m == 1000.0
    assert IMPERIAL.split_m == pytest.approx(1609.344)
