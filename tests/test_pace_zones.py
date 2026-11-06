"""Tests for pace zones: boundaries, whole-run counting and second-by-second counting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import pace_zones
from src.units import IMPERIAL

THRESHOLD = 4.0  # min/km, i.e. 4.1667 m/s


def make_runs(specs: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    """Tidy runs from ``(date, moving_time_s, pace, gap_pace)``."""
    return pd.DataFrame([
        {"activity_id": i + 1, "date": pd.Timestamp(day), "distance_km": seconds / 60 / pace,
         "moving_time_s": seconds, "pace_min_per_km": pace, "gap_pace_min_per_km": gap,
         "name": f"Run {i + 1}"}
        for i, (day, seconds, pace, gap) in enumerate(specs)
    ])


def test_zone_boundaries_are_fractions_of_threshold_speed():
    # Speed fractions 0.70, 0.80, 0.90, 1.00, 1.10 of threshold.
    paces = pd.Series([THRESHOLD / f for f in (0.70, 0.80, 0.90, 1.00, 1.10)] + [np.nan, 0.0])
    zones = pace_zones.assign_zone(paces, THRESHOLD)
    assert zones.tolist()[:5] == [1, 2, 3, 4, 5]
    assert zones.iloc[5:].isna().all()
    # Exactly on a floor belongs to the faster zone.
    assert pace_zones.assign_zone(pd.Series([THRESHOLD / 0.95]), THRESHOLD).iloc[0] == 4


def test_zone_table_lists_pace_ranges_slow_to_fast():
    table = pace_zones.zone_table(THRESHOLD)
    assert table["label"].tolist() == ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo",
                                       "Z4 Threshold", "Z5 VO2 max"]
    assert table["pace"].iloc[0].startswith("slower than")
    assert table["pace"].iloc[-1].startswith("faster than")
    assert table.loc[3, "slow_min_per_km"] == pytest.approx(THRESHOLD / 0.95)
    assert "/mi" in pace_zones.zone_table(THRESHOLD, IMPERIAL)["pace"].iloc[2]


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        pace_zones.assign_zone(pd.Series([5.0]), 0.0)


def test_whole_run_zones_use_grade_adjusted_pace():
    """A hilly run slow on the clock but threshold effort by GAP counts as Z4."""
    runs = make_runs([("2024-01-01", 3600, 5.5, 4.0),        # hilly: GAP at threshold
                      ("2024-01-02", 1800, 5.5, np.nan)])    # no GAP: raw pace, Z1/Z2
    dist = pace_zones.zone_distribution(runs, THRESHOLD).set_index("zone")
    assert dist.loc[4, "hours"] == pytest.approx(1.0)
    assert dist.loc[1, "hours"] == pytest.approx(0.5)  # 4.0/5.5 = 0.73 of threshold speed
    assert dist["share"].sum() == pytest.approx(1.0)
    assert len(dist) == 5


def test_monthly_and_per_run_breakdowns_fill_every_zone():
    runs = make_runs([("2024-01-05", 3600, 5.0, 5.0), ("2024-03-05", 1200, 3.8, 3.8)])
    monthly = pace_zones.monthly_zone_hours(runs, THRESHOLD)
    assert len(monthly) == 3 * 5  # Jan-Mar, five zones each
    assert monthly["hours"].sum() == pytest.approx(4800 / 3600)
    per_run = pace_zones.run_zone_breakdown(runs, THRESHOLD)
    assert per_run.groupby("activity_id")["share"].sum().tolist() == pytest.approx([1.0, 1.0])
    assert (per_run.groupby("activity_id")["share"].max() == 1.0).all()


def test_stream_zones_split_an_interval_session():
    """Seconds from the speed histogram land in their own zones, not the average's."""
    threshold_speed = 1000.0 / 60.0 / THRESHOLD
    gap_seconds = pd.DataFrame({
        "activity_id": [1, 1, 2],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
        "speed_m_s": [round(threshold_speed * 0.7, 1), round(threshold_speed * 1.15, 1),
                      round(threshold_speed * 0.83, 1)],
        "seconds": [1200.0, 600.0, 1800.0],
    })
    dist = pace_zones.stream_zone_distribution(gap_seconds, THRESHOLD).set_index("zone")
    assert dist.loc[1, "hours"] == pytest.approx(1200 / 3600)
    assert dist.loc[5, "hours"] == pytest.approx(600 / 3600)
    assert dist.loc[2, "runs"] == 1
    per_run = pace_zones.stream_run_zone_breakdown(gap_seconds, THRESHOLD)
    first = per_run[per_run["activity_id"] == 1].set_index("zone")["share"]
    assert first[1] == pytest.approx(2 / 3) and first[5] == pytest.approx(1 / 3)
    monthly = pace_zones.stream_monthly_zone_hours(gap_seconds, THRESHOLD)
    assert monthly["hours"].sum() == pytest.approx(1.0)


def test_empty_stream_data_gives_empty_zones():
    empty = pd.DataFrame(columns=["activity_id", "date", "speed_m_s", "seconds"])
    assert pace_zones.stream_zone_distribution(empty, THRESHOLD)["hours"].sum() == 0
    assert pace_zones.stream_monthly_zone_hours(empty, THRESHOLD).empty
    assert pace_zones.stream_run_zone_breakdown(empty, THRESHOLD).empty
