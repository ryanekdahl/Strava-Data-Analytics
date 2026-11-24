"""Tests for lining up two exports over a fair comparison window."""

from __future__ import annotations

import pandas as pd
import pytest

from src.compare import (
    SHARED_DATES, comparison_windows, cumulative, distinct_names, fitness_lines, is_aligned,
    runs_in_windows, shared_window, weekly, window_choices,
)


def make_runs(days: list[str], km: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame({
        "activity_id": range(1, len(days) + 1), "date": pd.to_datetime(days),
        "name": "Run", "distance_km": km, "moving_time_s": km * 300.0,
        "elevation_gain_m": 10.0,
    })


A = make_runs(["2024-01-01", "2024-03-01", "2024-06-30"])
B = make_runs(["2024-02-15", "2024-04-01", "2024-09-01"], km=10.0)
OLD = make_runs(["2020-01-06", "2020-02-03"])


def test_overlapping_exports_share_their_common_dates():
    assert shared_window({"A": A, "B": B}) == (pd.Timestamp("2024-02-15"),
                                               pd.Timestamp("2024-07-01"))
    assert window_choices({"A": A, "B": B})[0] == SHARED_DATES
    windows = comparison_windows({"A": A, "B": B}, SHARED_DATES)
    assert is_aligned(windows)
    cut = runs_in_windows({"A": A, "B": B}, windows)
    assert len(cut["A"]) == 2 and len(cut["B"]) == 2


def test_exports_from_different_years_offer_no_shared_window():
    runs = {"Now": A, "Then": OLD}
    assert shared_window(runs) is None
    assert SHARED_DATES not in window_choices(runs)
    with pytest.raises(ValueError):
        comparison_windows(runs, SHARED_DATES)


def test_relative_windows_count_back_from_each_exports_own_last_run():
    windows = comparison_windows({"Now": A, "Then": OLD}, "Last 4 weeks")
    assert windows["Now"] == (pd.Timestamp("2024-06-03"), pd.Timestamp("2024-07-01"))
    assert windows["Then"] == (pd.Timestamp("2020-01-07"), pd.Timestamp("2020-02-04"))
    assert not is_aligned(windows)
    everything = comparison_windows({"Now": A, "Then": OLD}, "All time")
    assert everything["Then"] == (pd.Timestamp("2020-01-06"), pd.Timestamp("2020-02-04"))


def test_unaligned_charts_count_days_and_weeks_from_each_window():
    runs = {"Now": A, "Then": OLD}
    windows = comparison_windows(runs, "All time")
    cut = runs_in_windows(runs, windows)
    totals = cumulative(cut, windows)
    then = totals[totals["athlete"] == "Then"]
    assert then["date"].iloc[0] == 1 and then["distance_km"].iloc[-1] == 10.0
    weeks = weekly(cut, windows)
    assert weeks["week_start"].iloc[0] == 1
    assert weeks.loc[0, "Then"] == 5.0 and weeks.loc[0, "Now"] == 5.0
    assert weeks["Then"].isna().any()  # the shorter export runs out of weeks


def test_aligned_charts_keep_calendar_dates():
    runs = {"A": A, "B": B}
    windows = comparison_windows(runs, SHARED_DATES)
    totals = cumulative(runs_in_windows(runs, windows), windows)
    assert totals["date"].iloc[0] == pd.Timestamp("2024-02-15")
    dates = pd.date_range("2023-01-01", "2024-12-31")
    fitness = pd.DataFrame({"date": dates, "ctl": 1.0, "atl": 1.0, "tsb": 0.0})
    lines = fitness_lines({"A": fitness, "B": fitness}, windows)
    assert lines["date"].min() == pd.Timestamp("2024-02-15")


def test_names_fall_back_and_stay_distinct():
    assert distinct_names(["", "  "], ["Athlete 1", "Athlete 2"]) == ["Athlete 1", "Athlete 2"]
    assert distinct_names(["Sam", "Sam"], ["A", "B"]) == ["Sam", "Sam (2)"]
