"""Tests for the near-PR alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.near_pr import (
    GETTING_CLOSE, NEW_PR, NO_RECENT, OFF_PACE, WITHIN_REACH, chances, classify,
    near_pr_table, whole_run_efforts,
)

FIVE_K = 5000.0
TEN_K = 10000.0


def efforts(rows: list[tuple[int, str, float, float]]) -> pd.DataFrame:
    """Long best-effort frame from ``(activity_id, date, distance_m, time_s)``."""
    frame = pd.DataFrame(rows, columns=["activity_id", "date", "distance_m", "time_s"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def row(table: pd.DataFrame, distance: str) -> pd.Series:
    return table.set_index("distance").loc[distance]


def test_gap_and_status_against_an_old_pr():
    table = near_pr_table(efforts([
        (1, "2025-01-10", FIVE_K, 1200.0),   # PR, long ago
        (2, "2025-11-01", FIVE_K, 1230.0),   # recent best, 2.5% off
        (3, "2025-11-20", FIVE_K, 1260.0),
    ]))
    five = row(table, "5K")
    assert five["pr_time_s"] == 1200.0 and five["recent_time_s"] == 1230.0
    assert five["gap_s"] == pytest.approx(30.0)
    assert five["gap_pct"] == pytest.approx(2.5)
    assert five["status"] == WITHIN_REACH
    assert five["pr_age_days"] == (pd.Timestamp("2025-11-20") - pd.Timestamp("2025-01-10")).days
    # One second under the PR, as an even pace over 5 km.
    assert five["target_pace_min_per_km"] == pytest.approx(1199.0 / 60.0 / 5.0)


def test_a_pr_set_in_the_window_is_new():
    table = near_pr_table(efforts([
        (1, "2025-01-10", FIVE_K, 1250.0),
        (2, "2025-11-01", FIVE_K, 1200.0),
    ]))
    assert row(table, "5K")["status"] == NEW_PR


def test_no_effort_in_the_window():
    table = near_pr_table(efforts([
        (1, "2025-01-10", TEN_K, 2500.0),
        (2, "2025-11-01", FIVE_K, 1200.0),
    ]))
    ten = row(table, "10K")
    assert ten["status"] == NO_RECENT
    assert np.isnan(ten["gap_pct"])


def test_distances_never_covered_are_left_out_and_order_is_by_distance():
    table = near_pr_table(efforts([
        (1, "2025-11-01", TEN_K, 2500.0),
        (1, "2025-11-01", FIVE_K, 1200.0),
    ]))
    assert table["distance"].tolist() == ["5K", "10K"]


@pytest.mark.parametrize("gap, recent, expected", [
    (0.0, False, WITHIN_REACH), (3.0, False, WITHIN_REACH), (3.1, False, GETTING_CLOSE),
    (8.0, False, GETTING_CLOSE), (8.1, False, OFF_PACE), (np.nan, False, NO_RECENT),
    (5.0, True, NEW_PR),
])
def test_classify_thresholds(gap, recent, expected):
    assert classify(gap, recent) == expected


def test_chances_lists_only_within_reach_closest_first():
    table = near_pr_table(efforts([
        (1, "2025-01-01", FIVE_K, 1200.0), (2, "2025-11-01", FIVE_K, 1230.0),   # 2.5%
        (1, "2025-01-01", TEN_K, 2500.0), (2, "2025-11-01", TEN_K, 2510.0),     # 0.4%
        (1, "2025-01-01", 1000.0, 200.0), (2, "2025-11-01", 1000.0, 230.0),     # 15%
    ]))
    assert chances(table)["distance"].tolist() == ["10K", "5K"]


def test_window_counts_back_from_as_of():
    best = efforts([(1, "2025-01-10", FIVE_K, 1200.0), (2, "2025-06-01", FIVE_K, 1220.0)])
    assert row(near_pr_table(best, as_of=pd.Timestamp("2025-07-01")), "5K")["status"] \
        == WITHIN_REACH
    assert row(near_pr_table(best, as_of=pd.Timestamp("2026-01-01")), "5K")["status"] \
        == NO_RECENT


def test_whole_run_efforts_cover_each_distance_at_average_pace():
    runs = pd.DataFrame({"activity_id": [7], "date": [pd.Timestamp("2025-03-01")],
                         "distance_km": [10.5], "moving_time_s": [3150.0]})
    long = whole_run_efforts(runs).set_index("distance_m")["time_s"]
    assert long[FIVE_K] == pytest.approx(1500.0)
    assert long[TEN_K] == pytest.approx(3000.0)
    assert 15000.0 not in long.index


def test_empty_input():
    assert near_pr_table(efforts([])).empty
