"""Tests for heart-rate zone assignment and the zone distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heart_rate import (
    assign_zone, hr_coverage, monthly_zone_hours, zone_distribution, zone_table,
)


def make_runs(specs: list[tuple[str, int, float]]) -> pd.DataFrame:
    """Build a tidy activities frame from ``(date, moving_time_s, avg_hr)``."""
    return pd.DataFrame([
        {"activity_id": i + 1, "date": pd.Timestamp(day), "distance_km": 10.0,
         "moving_time_s": seconds, "avg_hr": hr}
        for i, (day, seconds, hr) in enumerate(specs)
    ])


def test_zone_boundaries_follow_heart_rate_reserve():
    """With rest 60 / max 160 the reserve is 100 bpm, so zone floors are 120/130/140/150."""
    hr = pd.Series([100.0, 119.9, 120.0, 135.0, 149.9, 150.0, 175.0, np.nan])
    zones = assign_zone(hr, hr_rest=60, hr_max=160)
    assert zones.tolist()[:-1] == [1, 1, 2, 3, 4, 5, 5]
    assert pd.isna(zones.iloc[-1])
    assert zone_table(60, 160)["bpm"].tolist() == [
        "below 120", "120-130", "130-140", "140-150", "150+"]


def test_invalid_hr_settings_are_rejected():
    with pytest.raises(ValueError):
        assign_zone(pd.Series([150.0]), hr_rest=180, hr_max=170)


def test_distribution_sums_time_per_zone_and_ignores_runs_without_hr():
    runs = make_runs([
        ("2026-01-05", 3600, 125.0),   # Z2, 1 h
        ("2026-01-07", 1800, 126.0),   # Z2, 0.5 h
        ("2026-01-09", 1800, 155.0),   # Z5, 0.5 h
        ("2026-01-11", 7200, np.nan),  # no HR: excluded, not counted as easy
    ])
    dist = zone_distribution(runs, hr_rest=60, hr_max=160).set_index("zone")
    assert list(dist.index) == [1, 2, 3, 4, 5]
    assert dist.loc[2, "hours"] == pytest.approx(1.5)
    assert dist.loc[5, "runs"] == 1
    assert dist.loc[3, "hours"] == 0.0
    assert dist["share"].sum() == pytest.approx(1.0)
    assert dist.loc[2, "share"] == pytest.approx(0.75)
    assert hr_coverage(runs) == pytest.approx(0.75)


def test_monthly_zone_hours_zero_fills_empty_months_and_zones():
    runs = make_runs([("2026-01-05", 3600, 125.0), ("2026-03-05", 3600, 145.0)])
    monthly = monthly_zone_hours(runs, hr_rest=60, hr_max=160)
    assert len(monthly) == 3 * 5
    february = monthly[monthly["month_start"] == pd.Timestamp("2026-02-01")]
    assert february["hours"].sum() == 0.0
    march = monthly[monthly["month_start"] == pd.Timestamp("2026-03-01")].set_index("zone")
    assert march.loc[4, "hours"] == pytest.approx(1.0)
