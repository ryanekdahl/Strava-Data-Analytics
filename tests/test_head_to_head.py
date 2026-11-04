"""Tests for the head-to-head comparisons between you and friends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.generate_sample import SAMPLE_FRIENDS, generate_friends, generate_sample
from src.head_to_head import (
    METRICS, TIE, athlete_profile, cumulative_totals, find_friend_exports, friend_name,
    head_to_head_table, leader, period_weeks, prediction_comparison, record_comparison,
    remove_friend_export, safe_file_stem, save_friend_export, scoreboard, shared_runs, weekly_battles,
)
from src.loader import load_activities

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15")  # two Monday weeks


def make_runs(specs: list[tuple[str, float, int]]) -> pd.DataFrame:
    """A tidy runs frame from ``(date, distance_km, moving_time_s)``."""
    frame = pd.DataFrame(
        [{"activity_id": i + 1, "date": pd.Timestamp(day), "name": f"Run {i + 1}",
          "distance_km": km, "moving_time_s": seconds, "elevation_gain_m": 10.0}
         for i, (day, km, seconds) in enumerate(specs)],
        columns=["activity_id", "date", "name", "distance_km", "moving_time_s",
                 "elevation_gain_m"])
    frame["pace_min_per_km"] = frame["moving_time_s"] / 60.0 / frame["distance_km"]
    return frame


def flat_fitness(ctl: float) -> pd.DataFrame:
    dates = pd.date_range("2023-12-01", "2024-01-31", freq="D")
    return pd.DataFrame({"date": dates, "ctl": ctl, "atl": ctl, "tsb": 0.0})


def test_friend_names_come_from_files_and_folders(tmp_path):
    (tmp_path / "jamie_lee.csv").write_text("x")
    (tmp_path / "pat").mkdir()
    (tmp_path / "pat" / "activities.csv").write_text("x")
    found = find_friend_exports(tmp_path)
    assert set(found) == {"Jamie Lee", "Pat"}
    assert friend_name(tmp_path / "pat" / "activities.csv") == "Pat"
    assert find_friend_exports(tmp_path / "missing") == {}


def test_saving_a_friend_sanitises_the_name(tmp_path):
    path = save_friend_export(b"a,b\n", "Jo / ../../etc", tmp_path)
    assert path.parent == tmp_path and path.name == "jo_etc.csv"
    assert safe_file_stem("Ana María") == "ana_mar_a"
    with pytest.raises(ValueError):
        safe_file_stem("../")


def test_period_weeks_span_the_whole_period_even_with_no_runs():
    weeks = period_weeks(make_runs([("2024-01-03", 5.0, 1500)]), START, END)
    assert list(weeks["week_start"]) == [START, START + pd.Timedelta(days=7)]
    assert list(weeks["distance_km"]) == [5.0, 0.0]
    assert period_weeks(make_runs([]), START, END)["runs"].sum() == 0


def test_profile_counts_only_the_period():
    runs = make_runs([("2023-12-20", 30.0, 9000), ("2024-01-02", 10.0, 3000),
                      ("2024-01-09", 5.0, 1500), ("2024-01-20", 40.0, 12000)])
    profile = athlete_profile(runs, flat_fitness(42.0), START, END)
    assert profile["distance_km"] == 15.0 and profile["runs"] == 2
    assert profile["weekly_km"] == 7.5
    assert profile["active_share"] == 1.0 and profile["longest_streak"] == 2
    assert profile["avg_pace"] == pytest.approx(5.0)
    assert profile["fitness"] == 42.0
    assert set(profile) == {key for key, _, _ in METRICS}


def test_leader_respects_direction_ties_and_missing():
    values = pd.Series({"A": 3.0, "B": 5.0, "C": np.nan})
    assert leader(values) == "B"
    assert leader(values, higher_is_better=False) == "A"
    assert leader(pd.Series({"A": 1.0, "B": 1.0})) == TIE
    assert leader(pd.Series({"A": np.nan})) is None


def test_head_to_head_table_and_scoreboard():
    fast = make_runs([("2024-01-02", 10.0, 2400)])
    far = make_runs([("2024-01-02", 20.0, 7200), ("2024-01-09", 10.0, 3600)])
    profiles = {"Me": athlete_profile(fast, flat_fitness(30.0), START, END),
                "Kim": athlete_profile(far, flat_fitness(50.0), START, END)}
    table = head_to_head_table(profiles).set_index("metric")
    assert table.loc["avg_pace", "leader"] == "Me"
    assert table.loc["distance_km", "leader"] == "Kim"
    assert table.loc["fitness", "leader"] == "Kim"
    wins = scoreboard(table.reset_index(), ["Me", "Kim"])
    assert wins["Me"] + wins["Kim"] <= len(METRICS) and wins["Kim"] > wins["Me"]


def test_record_comparison_skips_distances_nobody_ran():
    runs = {"Me": make_runs([("2024-01-02", 5.0, 1500)]),
            "Kim": make_runs([("2024-01-02", 10.0, 3300)])}
    records = record_comparison(runs).set_index("distance")
    assert "Marathon" not in records.index
    assert records.loc["5K", "leader"] == "Me"
    assert records.loc["5K", "Me"] == pytest.approx(1500.0)
    assert records.loc["5K", "Kim"] == pytest.approx(1650.0)
    assert np.isnan(records.loc["10K", "Me"]) and records.loc["10K", "leader"] == "Kim"


def test_prediction_comparison_orders_runners():
    runs = {"Me": make_runs([("2024-01-02", 5.0, 1200)]),
            "Kim": make_runs([("2024-01-02", 5.0, 1500)]), "Lou": make_runs([])}
    table = prediction_comparison(runs).set_index("distance")
    assert (table["leader"] == "Me").all()
    assert table["Lou"].isna().all()


def test_weekly_battles_and_cumulative_totals():
    runs = {"Me": make_runs([("2024-01-02", 10.0, 3000)]),
            "Kim": make_runs([("2024-01-03", 4.0, 1200), ("2024-01-10", 6.0, 1800)])}
    battles = weekly_battles(runs, START, END)
    assert list(battles["leader"]) == ["Me", "Kim"]
    total = cumulative_totals(runs, START, END)
    assert total.groupby("athlete")["distance_km"].max().to_dict() == {"Me": 10.0, "Kim": 10.0}
    assert len(total) == 2 * 14


def test_battles_can_be_fought_on_other_measures():
    runs = {"Me": make_runs([("2024-01-02", 10.0, 3000)]),
            "Kim": make_runs([("2024-01-03", 4.0, 1200), ("2024-01-04", 4.0, 1200)])}
    for measure in ("runs", "elevation_m"):
        leaders = weekly_battles(runs, START, END, measure)["leader"]
        assert leaders.iloc[0] == "Kim" and pd.isna(leaders.iloc[1])
    hours = cumulative_totals(runs, START, END, "hours")
    assert hours.groupby("athlete")["hours"].max().to_dict() == pytest.approx(
        {"Me": 3000 / 3600, "Kim": 2400 / 3600})


def test_removing_a_friend_stays_inside_the_folder(tmp_path):
    friends = tmp_path / "friends"
    path = save_friend_export(b"x", "Jo", friends)
    (friends / "pat").mkdir()
    (friends / "pat" / "activities.csv").write_text("x")
    remove_friend_export(path, friends)
    remove_friend_export(friends / "pat" / "activities.csv", friends)
    assert find_friend_exports(friends) == {}
    assert not (friends / "pat").exists()
    outside = tmp_path / "mine.csv"
    outside.write_text("x")
    with pytest.raises(ValueError):
        remove_friend_export(outside, friends)
    assert outside.exists()


def test_shared_runs_match_on_start_time_and_distance():
    mine = make_runs([("2024-01-06 08:00", 20.0, 6000),   # together
                      ("2024-01-07 08:00", 10.0, 3000),   # same time, far shorter for them
                      ("2024-01-08 08:00", 10.0, 3000)])  # they started an hour later
    theirs = make_runs([("2024-01-06 08:05", 19.5, 6240),
                        ("2024-01-07 08:00", 5.0, 1500),
                        ("2024-01-08 09:00", 10.0, 3000)])
    shared = shared_runs(mine, theirs)
    assert len(shared) == 1
    row = shared.iloc[0]
    assert row["date"] == pd.Timestamp("2024-01-06 08:00")
    assert row["pace_gap_s"] == pytest.approx((6240 / 19.5 - 300.0))  # I was faster
    assert shared_runs(mine, make_runs([])).empty


def test_sample_friends_load_and_run_with_you(tmp_path):
    you = tmp_path / "activities.csv"
    generate_sample(you, write_streams=False)
    paths = generate_friends(you)
    assert set(paths) == set(SAMPLE_FRIENDS)
    mine = load_activities(you)
    for name, path in paths.items():
        theirs = load_activities(path)
        assert len(theirs) > 100
        assert not shared_runs(mine, theirs).empty, name
