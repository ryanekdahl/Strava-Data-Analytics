"""Tests for the sleep/HRV import and its pairing with training load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.loader import find_real_export
from src.recovery import (
    MIN_PAIRS, correlation_table, join_recovery, load_recovery, strength_label,
    synthetic_recovery,
)
from src.training_load import compute_fitness_freshness


def fitness_from(loads: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(loads), freq="D", name="date")
    return compute_fitness_freshness(pd.Series(loads, index=index, dtype="float64"))


def test_template_columns_load_one_row_per_day():
    csv = b"date,sleep_score,hrv_ms\n2026-01-02,80,60\n2026-01-03,70,50\n2026-01-03,90,70\n"
    recovery = load_recovery(csv)
    assert list(recovery.columns) == ["date", "sleep_score", "hrv_ms"]
    assert len(recovery) == 2  # two rows on 3 Jan are averaged
    assert recovery.loc[1, "sleep_score"] == pytest.approx(80.0)
    assert recovery.loc[1, "hrv_ms"] == pytest.approx(60.0)


def test_vendor_columns_percent_signs_and_evening_timestamps():
    csv = (b"Cycle start time,Recovery score %,Sleep performance %,"
           b"Heart rate variability (ms),HRV 7d avg\n"
           b"2026-01-01 22:40:00,55%,88%,61,99\n"
           b"2026-01-03 06:10:00,70%,74%,52,99\n")
    recovery = load_recovery(csv)
    # The 22:40 sleep describes the morning of 2 Jan; the 06:10 row stays on 3 Jan.
    assert recovery["date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert recovery["sleep_score"].tolist() == [88.0, 74.0]
    assert recovery["hrv_ms"].tolist() == [61.0, 52.0]  # nightly, not the 7-day average


def test_only_one_metric_is_enough_and_out_of_range_values_are_dropped():
    recovery = load_recovery(b"Date,HRV\n2026-01-02,60\n2026-01-03,900\n2026-01-04,55\n")
    assert recovery["sleep_score"].isna().all()
    assert recovery["hrv_ms"].tolist() == [60.0, 55.0]


def test_files_without_a_metric_or_date_are_rejected():
    with pytest.raises(ValueError, match="sleep-score or HRV"):
        load_recovery(b"date,steps\n2026-01-02,9000\n")
    with pytest.raises(ValueError, match="date"):
        load_recovery(b"hrv_ms\n60\n")


def test_join_pairs_each_morning_with_the_previous_days_load():
    fitness = fitness_from([0.0, 100.0, 0.0, 50.0])
    recovery = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-04", "2026-02-01"]),
        "sleep_score": [80.0, 70.0, 75.0, 90.0],
        "hrv_ms": [60.0, 50.0, 58.0, 65.0],
    })
    joined = join_recovery(recovery, fitness).set_index("date")
    assert len(joined) == 3  # 1 Feb is past the training calendar
    assert np.isnan(joined.loc["2026-01-01", "load_prev_day"])
    assert joined.loc["2026-01-03", "load_prev_day"] == 100.0  # 2 Jan's run
    assert joined.loc["2026-01-03", "load_same_day"] == 0.0
    assert joined.loc["2026-01-04", "load_same_day"] == 50.0
    day_before = fitness.set_index("date").loc["2026-01-02"]
    assert joined.loc["2026-01-03", "atl_prev"] == pytest.approx(day_before["atl"])


def test_correlation_finds_a_planted_negative_link_and_reports_sample_size():
    rng = np.random.default_rng(0)
    loads = rng.uniform(0, 150, 60)
    fitness = fitness_from(list(loads))
    dates = fitness["date"].iloc[1:]
    recovery = pd.DataFrame({
        "date": dates.to_numpy(),
        "sleep_score": np.nan,
        "hrv_ms": 80.0 - 0.2 * loads[:-1],  # driven entirely by yesterday's load
    })
    table = correlation_table(join_recovery(recovery, fitness)).set_index(
        ["recovery", "training"])
    link = table.loc[("hrv_ms", "load_prev_day")]
    assert link["r"] == pytest.approx(-1.0)
    assert link["n"] == 59
    assert link["strength"] == "Strong negative"
    assert np.isnan(table.loc[("sleep_score", "load_prev_day"), "r"])


def test_too_few_days_gives_no_correlation():
    days = MIN_PAIRS - 1
    fitness = fitness_from([10.0 * i for i in range(days)])
    recovery = pd.DataFrame({"date": fitness["date"], "sleep_score": range(days),
                             "hrv_ms": np.nan})
    table = correlation_table(join_recovery(recovery, fitness))
    assert table["r"].isna().all()
    assert table["n"].max() == days


def test_strength_labels():
    assert strength_label(0.05) == "No clear link"
    assert strength_label(-0.2) == "Weak negative"
    assert strength_label(0.4) == "Moderate positive"
    assert strength_label(float("nan")) == "Not enough data"


def test_synthetic_recovery_shows_hrv_dipping_after_big_days():
    rng = np.random.default_rng(3)
    fitness = fitness_from(list(rng.choice([0.0, 40.0, 80.0, 160.0], 300)))
    recovery = synthetic_recovery(fitness)
    assert recovery["sleep_score"].between(30, 100).all()
    table = correlation_table(join_recovery(recovery, fitness)).set_index(
        ["recovery", "training"])
    assert table.loc[("hrv_ms", "load_prev_day"), "r"] < -0.3


def test_recovery_file_is_not_mistaken_for_a_strava_export(tmp_path):
    (tmp_path / "recovery.csv").write_text("date,hrv_ms\n2026-01-01,60\n")
    assert find_real_export(tmp_path) is None
    (tmp_path / "my_runs.csv").write_text("x\n")
    assert find_real_export(tmp_path).name == "my_runs.csv"
