"""Tests for best efforts, the Riegel model and the PR tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.performance import (
    Effort, RIEGEL_EXPONENT, best_effort_frontier, best_efforts, best_recent_effort,
    format_duration, format_pace, predict_races, recent_vs_all_time, riegel_time,
    rolling_pace,
)


def make_runs(specs: list[tuple[str, float, int]]) -> pd.DataFrame:
    """Build a tidy activities frame from ``(date, distance_km, moving_time_s)``."""
    frame = pd.DataFrame(
        [
            {"activity_id": i + 1, "date": pd.Timestamp(day), "name": f"Run {i + 1}",
             "distance_km": km, "moving_time_s": seconds, "elapsed_time_s": seconds,
             "avg_hr": np.nan, "elevation_gain_m": np.nan}
            for i, (day, km, seconds) in enumerate(specs)
        ]
    )
    frame["pace_min_per_km"] = (frame["moving_time_s"] / 60.0) / frame["distance_km"]
    return frame


def test_riegel_matches_a_hand_computed_value():
    """A 20:00 5K predicts 10 km in 20:00 x 2^1.06 = 41:42."""
    predicted = riegel_time(time_s=1200.0, from_km=5.0, to_km=10.0)
    expected = 1200.0 * 2.0 ** 1.06
    assert predicted == pytest.approx(expected, rel=1e-12)
    assert predicted == pytest.approx(2502.0, abs=1.0)
    assert format_duration(predicted) == "41:42"


def test_riegel_predicts_longer_times_for_longer_distances():
    effort = Effort(distance_km=5.0, time_s=1200.0)
    table = predict_races(effort)

    ordered = table.sort_values("distance_km")
    assert ordered["predicted_time_s"].is_monotonic_increasing
    # The same distance back predicts the same time.
    assert riegel_time(1200.0, 5.0, 5.0) == pytest.approx(1200.0)
    # ...and pace decays with distance, which is the whole point of the model.
    pace = ordered["predicted_time_s"] / ordered["distance_km"]
    assert pace.is_monotonic_increasing


def test_riegel_exponent_of_one_would_mean_no_fatigue():
    """Sanity check on the exponent's role: 1.0 is a flat-pace extrapolation."""
    assert riegel_time(1200.0, 5.0, 10.0, exponent=1.0) == pytest.approx(2400.0)
    assert riegel_time(1200.0, 5.0, 10.0, exponent=RIEGEL_EXPONENT) > 2400.0


def test_riegel_rejects_impossible_distances():
    with pytest.raises(ValueError):
        riegel_time(1200.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        riegel_time(1200.0, 5.0, -1.0)


def test_best_effort_returns_the_fastest_qualifying_run():
    runs = make_runs([
        ("2024-01-01", 10.0, 3000),   # 5:00/km
        ("2024-02-01", 12.0, 3360),   # 4:40/km -- fastest run over 10 km
        ("2024-03-01", 4.0, 1000),    # 4:10/km but far too short to qualify for 10K
        ("2024-04-01", 10.5, 3255),   # 5:10/km
    ])
    records = best_efforts(runs).set_index("distance")

    assert records.loc["10K", "activity_id"] == 2
    assert records.loc["10K", "pace_min_per_km"] == pytest.approx(3360 / 60 / 12)
    assert records.loc["10K", "implied_time_s"] == pytest.approx(3360 / 12 * 10)
    # The 4 km run wins the 3K record, where it does qualify.
    assert records.loc["3K", "activity_id"] == 3
    # Nobody ran a half, so that row is absent rather than fabricated.
    assert "Half Marathon" not in records.index


def test_best_efforts_are_monotonic_across_distances():
    """Longer standard distances can never show a faster best pace."""
    rng = np.random.default_rng(7)
    runs = make_runs([
        (f"2024-01-{day:02d}", km, int(km * pace * 60))
        for day, (km, pace) in enumerate(
            zip(rng.uniform(3, 25, 25), rng.uniform(4.2, 6.0, 25)), start=1)
    ])
    records = best_efforts(runs).sort_values("distance_km")
    assert records["pace_min_per_km"].is_monotonic_increasing


def test_frontier_contains_only_undominated_runs():
    runs = make_runs([
        ("2024-01-01", 5.0, 1500),    # 5:00/km
        ("2024-01-08", 10.0, 2700),   # 4:30/km -- longer AND faster, dominates the 5 km
        ("2024-01-15", 21.0, 6930),   # 5:30/km -- longest, so it survives
        ("2024-01-22", 8.0, 2640),    # 5:30/km -- dominated by the 10 km run
    ])
    frontier = best_effort_frontier(runs)

    assert list(frontier["activity_id"]) == [2, 3]
    assert frontier["distance_km"].is_monotonic_increasing
    assert frontier["pace_min_per_km"].is_monotonic_increasing


def test_best_recent_effort_prefers_the_recent_window():
    runs = make_runs([
        ("2024-01-01", 10.0, 2400),   # 4:00/km, a year ago: brilliant but stale
        ("2024-12-01", 10.0, 3000),   # 5:00/km, recent
        ("2024-12-20", 10.0, 2880),   # 4:48/km, recent and fastest of the two
    ])
    effort = best_recent_effort(runs, days=90)

    assert effort == Effort(10.0, 2880.0)
    # Widening the window to everything brings the old flyer back.
    assert best_recent_effort(runs, days=3650).time_s == 2400.0


def test_recent_vs_all_time_flags_an_old_pr():
    runs = make_runs([
        ("2024-01-01", 10.0, 2400),   # all-time best, long ago
        ("2024-12-01", 10.0, 3000),   # current shape: 25% slower
    ])
    comparison = recent_vs_all_time(runs, recent_days=90).set_index("distance")

    assert comparison.loc["10K", "all_time_best"] == "40:00"
    assert comparison.loc["10K", "recent_best"] == "50:00"
    assert comparison.loc["10K", "gap_pct"] == pytest.approx(25.0)
    assert comparison.loc["10K", "status"] == "off PR pace"


def test_rolling_pace_weights_by_distance():
    """A 20 km long run must move the average more than a 2 km shakeout."""
    runs = make_runs([("2024-01-01", 20.0, 6000), ("2024-01-02", 2.0, 720)])
    rolled = rolling_pace(runs, window_days=30)

    assert rolled["rolling_pace"].iloc[0] == pytest.approx(5.0)
    assert rolled["rolling_pace"].iloc[1] == pytest.approx((6000 + 720) / 60 / 22)
    assert rolled["rolling_pace"].iloc[1] < 5.2  # nowhere near the 6:00/km shakeout


def test_formatters_cover_hours_and_missing_values():
    assert format_duration(3661) == "1:01:01"
    assert format_duration(59.6) == "1:00"
    assert format_duration(float("nan")) == "--"
    assert format_pace(4.5) == "4:30 /km"
    assert format_pace(None) == "--"
