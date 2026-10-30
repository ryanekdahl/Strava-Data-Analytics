"""Tests for the race planner's weeks, template and forward projection."""

from __future__ import annotations

import pandas as pd
import pytest

from src.race_plan import (
    RACE_READY_FORM, TAPER_SHAPES, plan_daily_load, plan_weeks, project_fitness,
    race_day_advice, race_day_summary, recent_weekly_load, template_plan,
)
from src.training_load import compute_fitness_freshness


def constant_load(value: float = 50.0, days: int = 200) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=days, freq="D", name="date")
    return pd.Series(value, index=index, dtype="float64", name="load")


def test_weeks_count_back_from_race_day():
    """The last block is the full week before the race; only the first can be short."""
    weeks = plan_weeks(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-20"))
    assert weeks["days"].tolist() == [5, 7, 7]
    assert weeks["days"].sum() == 19
    assert weeks["weeks_out"].tolist() == [3, 2, 1]
    assert weeks["week_start"].iloc[0] == pd.Timestamp("2024-03-01")
    assert weeks["week_start"].iloc[-1] == pd.Timestamp("2024-03-13")


def test_no_weeks_when_the_race_is_not_ahead():
    assert plan_weeks(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01")).empty


def test_template_builds_then_tapers():
    weeks = plan_weeks(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01") + pd.Timedelta(weeks=6))
    plan = template_plan(weeks, 400.0, weekly_increase=0.10, taper_weeks=2)
    loads = plan["load"].tolist()
    assert loads[:4] == [400.0, 440.0, 484.0, 532.0]
    peak = 400.0 * 1.1 ** 3
    assert loads[4:] == [round(peak * share) for share in TAPER_SHAPES[2]]


def test_template_scales_a_short_first_week():
    weeks = plan_weeks(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-20"))
    plan = template_plan(weeks, 700.0, taper_weeks=0)
    assert plan["load"].tolist() == [500.0, 700.0, 700.0]


def test_template_taper_longer_than_the_plan_keeps_its_last_weeks():
    weeks = plan_weeks(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-08"))
    plan = template_plan(weeks, 400.0, taper_weeks=3)
    assert plan["load"].tolist() == [round(400.0 * TAPER_SHAPES[3][-1])]


def test_daily_load_spreads_each_week_evenly():
    weeks = plan_weeks(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-20"))
    daily = plan_daily_load(weeks.assign(load=[100.0, 350.0, 700.0]))
    assert len(daily) == 19
    assert daily.sum() == pytest.approx(1150.0)
    assert daily.iloc[0] == pytest.approx(20.0)
    assert daily.iloc[-1] == pytest.approx(100.0)


def test_projection_continues_the_recorded_history():
    """A plan that repeats the history's load gives the same curve as a longer history."""
    history = constant_load(50.0, days=200)
    race = history.index[-1] + pd.Timedelta(days=29)
    weeks = plan_weeks(history.index[-1] + pd.Timedelta(days=1), race)
    plan = weeks.assign(load=50.0 * weeks["days"])
    projection = project_fitness(history, plan, race)

    longer = constant_load(50.0, days=228)
    expected = compute_fitness_freshness(longer)
    assert len(projection) == 229
    assert projection["projected"].sum() == 29
    assert projection["ctl"].iloc[:228].to_numpy() == pytest.approx(expected["ctl"].to_numpy())
    # The race day itself carries no load.
    assert projection["load"].iloc[-1] == 0.0


def test_a_taper_raises_race_day_form():
    history = constant_load(60.0, days=200)
    race = history.index[-1] + pd.Timedelta(weeks=6, days=1)
    weeks = plan_weeks(history.index[-1] + pd.Timedelta(days=1), race)
    flat = project_fitness(history, template_plan(weeks, 420.0, taper_weeks=0), race)
    tapered = project_fitness(history, template_plan(weeks, 420.0, taper_weeks=2), race)
    assert race_day_summary(tapered)["tsb"] > race_day_summary(flat)["tsb"] + 5
    assert race_day_summary(tapered)["ctl"] < race_day_summary(flat)["ctl"]


def test_summary_reports_change_and_ramp():
    history = constant_load(30.0, days=200)
    race = history.index[-1] + pd.Timedelta(weeks=4, days=1)
    weeks = plan_weeks(history.index[-1] + pd.Timedelta(days=1), race)
    projection = project_fitness(history, weeks.assign(load=700.0), race)
    summary = race_day_summary(projection)
    start = float(projection.loc[~projection["projected"], "ctl"].iloc[-1])
    assert summary["ctl_change"] == pytest.approx(summary["ctl"] - start)
    assert summary["max_ramp"] > 0
    assert summary["min_tsb"] < 0


def test_projection_honours_other_time_constants():
    history = constant_load(50.0, days=200)
    race = history.index[-1] + pd.Timedelta(weeks=2, days=1)
    weeks = plan_weeks(history.index[-1] + pd.Timedelta(days=1), race)
    plan = weeks.assign(load=0.0)
    slow = project_fitness(history, plan, race, ctl_days=60, atl_days=10)
    fast = project_fitness(history, plan, race, ctl_days=30, atl_days=5)
    # After two weeks of rest a slower Fitness constant has lost less.
    assert slow["ctl"].iloc[-1] / slow["ctl"].iloc[-16] > fast["ctl"].iloc[-1] / fast["ctl"].iloc[-16]


def test_recent_weekly_load():
    assert recent_weekly_load(constant_load(20.0)) == pytest.approx(140.0)
    assert recent_weekly_load(pd.Series(dtype="float64")) == 0.0


@pytest.mark.parametrize("tsb, word", [(RACE_READY_FORM[0] - 1, "below"),
                                       (sum(RACE_READY_FORM) / 2, "sits in"),
                                       (RACE_READY_FORM[1] + 1, "above")])
def test_advice_matches_the_band(tsb, word):
    assert word in race_day_advice({"tsb": tsb})
