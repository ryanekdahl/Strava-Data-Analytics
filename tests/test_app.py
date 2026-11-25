"""Smoke tests: the whole dashboard runs on the sample data under every sidebar setting.

These drive ``app.py`` through Streamlit's ``AppTest``, so every tab executes.
The first run may generate the sample export and parse its GPS files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP = Path(__file__).resolve().parent.parent / "app.py"
TIMEOUT_S = 600


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    return at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]


def test_default_view_runs(app):
    _assert_clean(app)
    assert app.title[0].value == "Strava Run Analytics"


def test_miles_change_the_numbers_on_screen(app):
    app.radio(key="units").set_value("Miles").run()
    _assert_clean(app)
    distance = next(m for m in app.metric if m.label == "Total distance")
    assert distance.value.endswith(" mi")
    app.radio(key="units").set_value("Kilometres").run()


def test_a_period_narrows_the_runs(app):
    everything = app.markdown[0].value
    app.selectbox(key="period").set_value("Last 3 months").run()
    _assert_clean(app)
    assert "of " in app.markdown[0].value and app.markdown[0].value != everything
    app.selectbox(key="period").set_value("All time").run()


def test_a_period_compares_with_the_one_before(app):
    distance = next(m for m in app.metric if m.label == "Total distance")
    assert not distance.delta  # all time has nothing before it
    app.selectbox(key="period").set_value("Last 3 months").run()
    _assert_clean(app)
    distance = next(m for m in app.metric if m.label == "Total distance")
    assert distance.delta.endswith(" km vs previous period")
    app.selectbox(key="period").set_value("All time").run()


def test_route_map_with_and_without_street_tiles(app):
    app.radio(key="route_color").set_value("Pace").run()
    _assert_clean(app)
    assert any("Median pace" in m.value for m in app.markdown)
    app.radio(key="route_color").set_value("Heart rate").run()
    _assert_clean(app)
    app.toggle(key="route_basemap").set_value(True).run()
    _assert_clean(app)


def test_run_replay_plays_on_both_backdrops(app):
    app.toggle(key="replay_on").set_value(True).run()
    _assert_clean(app)
    for basemap in (True, False):
        app.toggle(key="route_basemap").set_value(basemap).run()
        _assert_clean(app)
        app.segmented_control(key="replay_length").set_value("10 s").run()
        _assert_clean(app)


def test_tabs(app):
    assert [tab.label for tab in app.tabs] == [
        "Overview", "Activities", "Routes", "Best efforts", "Fitness & Freshness", "Heart rate",
        "Gear", "Compare"]


def test_routes_tab_shows_the_pr_heat_map_for_each_route(app):
    picker = app.selectbox(key="route_pick")
    assert len(picker.options) >= 3  # the sample runner's regular routes
    for option in picker.options[:2]:
        picker.set_value(option).run()
        _assert_clean(app)
        assert any("Median pace" in m.value for m in app.markdown)
    app.toggle(key="routes_basemap").set_value(True).run()
    _assert_clean(app)


def test_fitness_tab_shows_load_and_workload_ratio(app):
    labels = {m.label for m in app.metric}
    assert {"Fitness (CTL)", "Fatigue (ATL)", "Form (TSB)", "ACWR"} <= labels


def test_best_efforts_and_race_calculator(app):
    assert any(h.value == "Best efforts" for h in app.subheader)
    assert any(h.value == "Race predictions" for h in app.subheader)
    good = app.text_input(key="calc_time").value
    app.text_input(key="calc_time").set_value("not a time").run()
    _assert_clean(app)
    assert app.warning
    app.text_input(key="calc_time").set_value(good).run()


def test_pr_watch_compares_recent_form_with_records(app):
    assert any(h.value == "PR watch" for h in app.subheader)
    assert any(m.label == "Within reach" for m in app.metric)


def test_run_detail_grade_adjusted_pace(app):
    gap = next(m for m in app.metric if m.label == "Grade-adjusted pace")
    assert gap.value != "--"
    app.radio(key="units").set_value("Miles").run()
    _assert_clean(app)
    assert next(m for m in app.metric if m.label == "Grade-adjusted pace").value.endswith("/mi")
    app.radio(key="units").set_value("Kilometres").run()


def test_run_detail_lines_up_hills_with_pace_and_heart_rate(app):
    assert any(h.value.startswith("Hills") for h in app.subheader)
    assert any("On the climbs you ran" in m.value for m in app.markdown)


def test_run_detail_search_narrows_the_run_list(app):
    picker = app.selectbox(key="detail_run")
    assert picker.label.endswith(" runs)")
    app.text_input(key="detail_run_query").set_value("zzz-no-such-run").run()
    _assert_clean(app)
    assert any("No runs match" in i.value for i in app.info)
    app.text_input(key="detail_run_query").set_value("").run()
    _assert_clean(app)
