"""Tests for the personal Riegel exponent and the fitted Banister model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.performance import best_efforts
from src.personal_models import (
    fit_banister, fit_riegel, modelled_performance, performance_index, riegel_efforts, t_critical,
)
from src.training_load import _ewma


def riegel_points(exponent: float, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    km = np.array([1.0, 3.0, 5.0, 10.0, 21.0975])
    seconds = 240.0 * km ** exponent * np.exp(rng.normal(0.0, noise, len(km)))
    return pd.DataFrame({"distance_km": km, "time_s": seconds})


def test_riegel_fit_recovers_an_exact_exponent():
    fit = fit_riegel(riegel_points(1.09))
    assert fit.exponent == pytest.approx(1.09)
    assert fit.coefficient == pytest.approx(240.0)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.predict(42.195) == pytest.approx(240.0 * 42.195 ** 1.09)
    assert fit.differs_from_default


def test_riegel_interval_covers_the_truth_and_widens_with_noise():
    tight = fit_riegel(riegel_points(1.06, noise=0.005))
    loose = fit_riegel(riegel_points(1.06, noise=0.03))
    assert tight.ci_low < 1.06 < tight.ci_high
    assert (loose.ci_high - loose.ci_low) > (tight.ci_high - tight.ci_low)
    assert not tight.differs_from_default


def test_riegel_needs_three_distances():
    assert fit_riegel(riegel_points(1.06).iloc[:2]) is None


def test_t_critical_values():
    assert t_critical(1) == pytest.approx(12.706)
    assert t_critical(10) == pytest.approx(2.228)
    assert t_critical(27) == pytest.approx(2.060)  # conservative between table rows
    assert t_critical(1000) == pytest.approx(1.960)


def test_riegel_efforts_fall_back_to_whole_runs_and_respect_the_range():
    runs = pd.DataFrame({
        "activity_id": [1, 2, 3], "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "name": "Run", "distance_km": [5.0, 10.0, 21.1], "pace_min_per_km": [4.5, 4.7, 5.0],
    })
    efforts = riegel_efforts(runs, None, min_km=4.0, max_km=12.0)
    assert list(efforts["distance"]) == ["5K", "10K"]
    expected = best_efforts(runs).set_index("distance").loc["5K", "implied_time_s"]
    assert efforts.set_index("distance").loc["5K", "time_s"] == pytest.approx(expected)
    recent = riegel_efforts(runs, None, since=pd.Timestamp("2024-02-15"))
    assert set(recent["distance"]) <= {"3K", "5K", "10K", "15K", "Half Marathon"}


def simulated_training(tau1=30, tau2=9, head_start=40.0, days=500, n_perf=300, seed=1):
    """Daily load with a build-up and taper, and performances from a known model."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    ramp = np.r_[np.linspace(0.5, 1.5, days // 2), np.linspace(1.5, 0.3, days - days // 2)]
    load = pd.Series(np.where(rng.random(days) < 0.6, rng.gamma(4, 20, days), 0.0) * ramp,
                     index=index)
    fitness = np.r_[0.0, _ewma(load.to_numpy(), tau1)[:-1]] \
        + head_start * (1 - 1 / tau1) ** np.arange(days)
    fatigue = np.r_[0.0, _ewma(load.to_numpy(), tau2)[:-1]]
    when = np.sort(rng.choice(np.arange(1, days), n_perf, replace=False))
    perf = pd.DataFrame({"date": index[when],
                         "performance": 0.2 * fitness[when] - 0.25 * fatigue[when]
                         + rng.normal(0.0, 1.0, n_perf)})
    return load, perf


def test_banister_fit_recovers_known_time_constants():
    load, perf = simulated_training()
    fit = fit_banister(load, perf, n_bootstrap=60)
    assert abs(fit.tau1 - 30) <= 6
    assert abs(fit.tau2 - 9) <= 2
    assert fit.k1 > 0 and fit.k2 > 0
    assert fit.carry == pytest.approx(0.2 * 40.0, rel=0.3)  # the head start is estimated
    assert fit.tau1_interval[0] <= fit.tau1 <= fit.tau1_interval[1]
    assert fit.r2 >= fit.r2_default  # the default pair is inside the search grid
    assert fit.peak_day is not None and fit.positive_day <= fit.peak_day


def test_banister_surface_marks_inadmissible_pairs_blank():
    load, perf = simulated_training()
    fit = fit_banister(load, perf, n_bootstrap=5)
    inadmissible = fit.surface[fit.surface["tau2"] >= fit.surface["tau1"]]
    assert inadmissible["r2"].isna().all()
    assert fit.surface["r2"].max() == pytest.approx(fit.r2)


def test_banister_needs_enough_performances():
    load, perf = simulated_training()
    assert fit_banister(load, perf.iloc[:10]) is None


def test_modelled_performance_tracks_the_data():
    load, perf = simulated_training()
    fit = fit_banister(load, perf, n_bootstrap=5)
    modelled = modelled_performance(load, fit).set_index("date")["predicted"]
    residual = perf["performance"].to_numpy() - modelled.loc[perf["date"]].to_numpy()
    assert np.std(residual) == pytest.approx(1.0, rel=0.15)  # just the noise is left


def test_performance_index_rewards_speed_at_the_same_heart_rate():
    rng = np.random.default_rng(3)
    n = 60
    hr = rng.uniform(130, 170, n)
    pace = 7.0 - 0.02 * (hr - 130) + rng.normal(0, 0.05, n)
    pace[-1] -= 0.5  # the last run was much faster than its heart rate predicts
    runs = pd.DataFrame({"activity_id": range(n), "date": pd.date_range("2024-01-01", periods=n),
                         "distance_km": 8.0, "pace_min_per_km": pace, "avg_hr": hr})
    index = performance_index(runs)
    assert len(index) == n
    assert index["performance"].iloc[-1] == index["performance"].max()
    assert index["performance"].mean() == pytest.approx(0.0, abs=1e-9)
    assert performance_index(runs.iloc[:10]).empty
