"""Tests for the training-load model and the CTL/ATL/TSB engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training_load import (
    ATL_DAYS, CTL_DAYS, compute_activity_load, compute_fitness_freshness,
    daily_load_series, estimate_threshold_pace,
)


def constant_load(value: float = 50.0, days: int = 400) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=days, freq="D", name="date")
    return pd.Series(value, index=index, dtype="float64", name="load")


def make_runs(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal tidy activities frame from partial row dicts."""
    defaults = {"activity_id": 1, "date": pd.Timestamp("2024-01-01"), "name": "Run",
                "distance_km": 10.0, "moving_time_s": 3000, "elapsed_time_s": 3000,
                "avg_hr": np.nan, "elevation_gain_m": np.nan}
    frame = pd.DataFrame([{**defaults, **row} for row in rows], columns=list(defaults))
    frame["pace_min_per_km"] = (frame["moving_time_s"] / 60.0) / frame["distance_km"]
    return frame


def test_constant_load_converges_to_that_load():
    """A steady diet of 50 points/day means Fitness and Fatigue both reach 50."""
    # 1000 days is many CTL time constants, so the residual is far below 1e-6.
    result = compute_fitness_freshness(constant_load(50.0, days=1000))
    assert result["ctl"].iloc[-1] == pytest.approx(50.0, abs=1e-6)
    assert result["atl"].iloc[-1] == pytest.approx(50.0, abs=1e-6)
    # Seeded at zero, both build up from below and never overshoot.
    assert result["ctl"].iloc[0] == pytest.approx(50.0 / CTL_DAYS)
    assert (result["ctl"] <= 50.0 + 1e-9).all()
    assert result["ctl"].is_monotonic_increasing
    # Fatigue is the faster average, so it leads Fitness the whole way up.
    assert (result["atl"] >= result["ctl"] - 1e-9).all()


def test_ewma_hits_the_expected_fraction_after_one_time_constant():
    """After N days of load, an alpha = 1/N average sits at 1 - (1 - 1/N)**N."""
    result = compute_fitness_freshness(constant_load(100.0, days=CTL_DAYS))
    expected = 100.0 * (1.0 - (1.0 - 1.0 / CTL_DAYS) ** CTL_DAYS)
    assert result["ctl"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_tsb_is_ctl_minus_atl_shifted_one_day():
    """Form is the balance you finished *yesterday* with."""
    rng = np.random.default_rng(0)
    load = constant_load(0.0, days=120)
    load.iloc[:] = rng.uniform(0, 120, size=len(load))
    result = compute_fitness_freshness(load)

    balance = result["ctl"] - result["atl"]
    assert result["tsb"].iloc[0] == pytest.approx(0.0)
    np.testing.assert_allclose(result["tsb"].to_numpy()[1:], balance.to_numpy()[:-1], rtol=1e-12)


def test_a_load_spike_raises_fatigue_faster_than_fitness():
    """The 7-day average must react harder to one big day than the 42-day one."""
    # Both averages must be fully converged before the spike, otherwise the
    # 42:7 ratio below is only approximate.
    load = constant_load(40.0, days=1000)
    load.iloc[-2] = 400.0  # a monster session, then an ordinary day
    result = compute_fitness_freshness(load)

    atl_jump = result["atl"].iloc[-2] - result["atl"].iloc[-3]
    ctl_jump = result["ctl"].iloc[-2] - result["ctl"].iloc[-3]
    assert atl_jump > ctl_jump > 0
    assert atl_jump == pytest.approx(ctl_jump * (CTL_DAYS / ATL_DAYS), rel=1e-9)
    # Steady training leaves Form near zero; the spike digs it into a hole the
    # following day, because Form lags by one day.
    assert result["tsb"].iloc[-2] == pytest.approx(0.0, abs=0.5)
    assert result["tsb"].iloc[-1] < -10.0


def test_rest_days_are_zero_filled_and_decay_the_averages():
    runs = make_runs([
        {"activity_id": 1, "date": pd.Timestamp("2024-03-01 07:00")},
        {"activity_id": 2, "date": pd.Timestamp("2024-03-01 18:00")},
        {"activity_id": 3, "date": pd.Timestamp("2024-03-11 07:00")},
    ])
    daily = daily_load_series(compute_activity_load(runs))

    assert len(daily) == 11  # a gap-free calendar, not three rows
    assert (daily.iloc[1:10] == 0.0).all()
    # Two runs on one day accumulate.
    assert daily.iloc[0] == pytest.approx(2 * daily.iloc[-1], rel=1e-9)

    result = compute_fitness_freshness(daily)
    assert result["atl"].iloc[9] < result["atl"].iloc[0]


def test_faster_running_scores_more_load_than_easy_running():
    """Same duration, higher intensity => strictly more load."""
    runs = make_runs([
        {"activity_id": 1, "distance_km": 10.0, "moving_time_s": 3000},  # 5:00/km
        {"activity_id": 2, "distance_km": 8.0, "moving_time_s": 3000},   # 6:15/km
    ])
    scored = compute_activity_load(runs, threshold_pace=5.0)

    assert scored["intensity_factor"].iloc[0] == pytest.approx(1.0)
    assert scored["load_pace"].iloc[0] == pytest.approx(3000 / 3600 * 100.0)
    assert scored["load_pace"].iloc[0] > scored["load_pace"].iloc[1]


def test_missing_heart_rate_falls_back_to_the_pace_score():
    runs = make_runs([
        {"activity_id": 1, "avg_hr": np.nan},
        {"activity_id": 2, "avg_hr": 160.0},
    ])
    scored = compute_activity_load(runs, threshold_pace=5.0, hr_weight=0.5)

    assert scored["load"].iloc[0] == pytest.approx(scored["load_pace"].iloc[0])
    assert np.isnan(scored["load_hr"].iloc[0])
    blended = 0.5 * scored["load_hr"].iloc[1] + 0.5 * scored["load_pace"].iloc[1]
    assert scored["load"].iloc[1] == pytest.approx(blended)
    assert scored["load"].notna().all()


def test_threshold_pace_ignores_short_runs():
    """A blazing 2 km cannot masquerade as sustainable threshold pace."""
    runs = make_runs([
        {"activity_id": 1, "distance_km": 2.0, "moving_time_s": 420},    # 3:30/km
        {"activity_id": 2, "distance_km": 10.0, "moving_time_s": 2700},  # 4:30/km
        {"activity_id": 3, "distance_km": 12.0, "moving_time_s": 3600},  # 5:00/km
    ])
    assert estimate_threshold_pace(runs, quantile=0.0) == pytest.approx(4.5)


def test_empty_history_is_handled():
    empty = make_runs([]).iloc[0:0]
    scored = compute_activity_load(empty)
    assert scored.empty and "load" in scored.columns
    assert daily_load_series(scored).empty
