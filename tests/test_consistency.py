"""Tests for volume aggregation, streaks and the acute:chronic workload ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.consistency import (
    ACUTE_DAYS, CHRONIC_DAYS, SWEET_SPOT, acwr_status, compute_acwr,
    monthly_volume, summary_kpis, weekly_streaks, weekly_volume,
)


def make_runs(specs: list[tuple[str, float, int]]) -> pd.DataFrame:
    """Build a tidy activities frame from ``(date, distance_km, moving_time_s)``."""
    frame = pd.DataFrame(
        [
            {"activity_id": i + 1, "date": pd.Timestamp(day), "name": f"Run {i + 1}",
             "distance_km": km, "moving_time_s": seconds, "elapsed_time_s": seconds,
             "avg_hr": np.nan, "elevation_gain_m": np.nan}
            for i, (day, km, seconds) in enumerate(specs)
        ],
        columns=["activity_id", "date", "name", "distance_km", "moving_time_s",
                 "elapsed_time_s", "avg_hr", "elevation_gain_m"],
    )
    frame["pace_min_per_km"] = (frame["moving_time_s"] / 60.0) / frame["distance_km"]
    return frame


def daily(value: float, days: int, start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=days, freq="D", name="date")
    return pd.Series(float(value), index=index, dtype="float64", name="load")


def test_weekly_volume_sums_into_monday_weeks():
    runs = make_runs([
        ("2024-01-01 07:00", 10.0, 3000),   # Monday
        ("2024-01-03 07:00", 5.0, 1800),    # Wednesday, same week
        ("2024-01-08 07:00", 8.0, 2400),    # the following Monday
    ])
    weekly = weekly_volume(runs)

    assert list(weekly.columns) == ["week_start", "distance_km", "runs", "moving_time_h"]
    assert weekly["week_start"].iloc[0] == pd.Timestamp("2024-01-01")
    assert (weekly["week_start"].dt.dayofweek == 0).all()  # every bucket starts Monday
    assert weekly["distance_km"].iloc[0] == pytest.approx(15.0)
    assert weekly["runs"].iloc[0] == 2
    assert weekly["moving_time_h"].iloc[0] == pytest.approx(4800 / 3600)
    assert weekly["distance_km"].iloc[1] == pytest.approx(8.0)


def test_empty_weeks_are_kept_as_zeroes_not_dropped():
    """Time off must show up as a gap, not be quietly closed up."""
    runs = make_runs([("2024-01-01", 10.0, 3000), ("2024-02-05", 10.0, 3000)])
    weekly = weekly_volume(runs)

    assert len(weekly) == 6  # six calendar weeks spanned, not two rows
    assert (weekly["runs"] == 0).sum() == 4
    assert weekly["distance_km"].iloc[1:5].eq(0.0).all()


def test_monthly_volume_rolls_up_by_calendar_month():
    runs = make_runs([
        ("2024-01-05", 10.0, 3000),
        ("2024-01-25", 12.0, 3600),
        ("2024-03-02", 8.0, 2400),   # February is empty
    ])
    monthly = monthly_volume(runs)

    assert list(monthly["month_start"]) == [
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    assert monthly["distance_km"].iloc[0] == pytest.approx(22.0)
    assert monthly["runs"].iloc[1] == 0
    assert monthly["distance_km"].iloc[2] == pytest.approx(8.0)


def test_streaks_count_consecutive_weeks_and_break_on_a_rest_week():
    weekly = pd.DataFrame({
        "week_start": pd.date_range("2024-01-01", periods=7, freq="W-MON"),
        "distance_km": [10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 0.0],
        "runs": [2, 3, 1, 0, 2, 2, 0],
        "moving_time_h": [1.0] * 7,
    })
    streaks = weekly_streaks(weekly)

    assert streaks.longest == 3          # the opening three weeks
    assert streaks.current == 0          # the data ends on an empty week
    assert weekly_streaks(weekly.iloc[:-1]).current == 2


def test_streaks_on_an_empty_history():
    assert weekly_streaks(weekly_volume(make_runs([]))) == (0, 0)


def test_steady_training_scores_an_acwr_of_one():
    """Means, not sums: a perfectly steady block is 1.0 whatever the windows."""
    result = compute_acwr(daily(50.0, days=60))
    settled = result["acwr"].dropna()

    assert settled.iloc[-1] == pytest.approx(1.0)
    assert np.allclose(settled.to_numpy(), 1.0)
    assert SWEET_SPOT[0] < settled.iloc[-1] < SWEET_SPOT[1]


def test_acwr_is_undefined_until_the_chronic_window_is_full():
    result = compute_acwr(daily(50.0, days=60))

    assert result["acwr"].iloc[: CHRONIC_DAYS - 1].isna().all()
    assert result["acwr"].iloc[CHRONIC_DAYS - 1:].notna().all()
    assert result["acute"].iloc[: ACUTE_DAYS - 1].isna().all()


def test_a_comeback_week_after_a_layoff_spikes_the_ratio():
    """Nothing for three weeks, then a full week back: the classic spike."""
    load = daily(0.0, days=CHRONIC_DAYS + ACUTE_DAYS)
    load.iloc[-ACUTE_DAYS:] = 70.0
    result = compute_acwr(load)
    latest = float(result["acwr"].iloc[-1])

    assert latest > SWEET_SPOT[1]
    # 28 days of history holding only these 7 loaded days caps the ratio at 4.
    assert latest == pytest.approx(CHRONIC_DAYS / ACUTE_DAYS)
    assert acwr_status(latest) == "Spiking -- elevated injury risk"


def test_a_total_layoff_leaves_the_ratio_undefined_rather_than_infinite():
    load = daily(0.0, days=60)
    load.iloc[:CHRONIC_DAYS] = 50.0
    result = compute_acwr(load)

    assert result["acwr"].iloc[CHRONIC_DAYS - 1] == pytest.approx(1.0)
    assert np.isnan(result["acwr"].iloc[-1])  # chronic leg is zero, not a division by it
    assert not np.isinf(result["acwr"].to_numpy()).any()


def test_acwr_status_reads_each_band():
    low, high = SWEET_SPOT
    assert acwr_status(low - 0.2).startswith("Detraining")
    assert acwr_status((low + high) / 2).startswith("Sweet spot")
    assert acwr_status(high + 0.1).startswith("Ramping up")
    assert acwr_status(2.0).startswith("Spiking")
    assert acwr_status(float("nan")) == "Not enough history"
    assert acwr_status(None) == "Not enough history"


def test_summary_kpis_totals():
    runs = make_runs([("2024-01-01", 10.0, 3600), ("2024-01-15", 20.0, 7200)])
    kpis = summary_kpis(runs)

    assert kpis["total_runs"] == 2
    assert kpis["total_distance_km"] == pytest.approx(30.0)
    assert kpis["total_hours"] == pytest.approx(3.0)
    assert kpis["avg_distance_km"] == pytest.approx(15.0)
    assert kpis["weeks"] == 2
    assert summary_kpis(make_runs([]))["total_runs"] == 0
