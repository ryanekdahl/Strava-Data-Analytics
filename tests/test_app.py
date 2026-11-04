"""Smoke tests: the whole dashboard runs on the sample data under every sidebar setting.

These drive ``app.py`` through Streamlit's ``AppTest``, so every tab executes.
The first run may generate the sample export and parse its GPS files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.head_to_head import METRICS  # noqa: E402

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
    app.radio(key="route_color").set_value("Heart rate").run()
    _assert_clean(app)
    app.toggle(key="route_basemap").set_value(True).run()
    _assert_clean(app)


def test_other_sports_tab_and_toggle(app):
    assert any(tab.label == "All sports" for tab in app.tabs)
    with_other = next(m for m in app.metric if m.label == "Fitness (CTL)").value
    app.toggle(key="include_other").set_value(False).run()
    _assert_clean(app)
    runs_only = next(m for m in app.metric if m.label == "Fitness (CTL)").value
    assert float(runs_only) < float(with_other)
    app.toggle(key="include_other").set_value(True).run()
    app.number_input(key="ftp").set_value(240).run()
    app.radio(key="sport_stack").set_value("Training load").run()
    _assert_clean(app)
    app.number_input(key="ftp").set_value(0).run()


def test_share_tab_makes_period_and_run_stickers(app):
    assert any(tab.label == "Share" for tab in app.tabs)
    app.multiselect(key="sticker_stats_This period").set_value(["Distance", "Fitness"]).run()
    _assert_clean(app)
    app.radio(key="sticker_kind").set_value("One run").run()
    _assert_clean(app)
    app.radio(key="sticker_ink").set_value("Black").run()
    _assert_clean(app)
    app.radio(key="sticker_kind").set_value("This period").run()


def test_race_plan_tab_projects_to_race_day(app):
    assert any(tab.label == "Race plan" for tab in app.tabs)
    assert any(m.label == "Race-day Form" for m in app.metric)
    app.selectbox(key="plan_taper").set_value(0).run()
    _assert_clean(app)
    untapered = float(next(m for m in app.metric if m.label == "Race-day Form").value)
    app.selectbox(key="plan_taper").set_value(3).run()
    _assert_clean(app)
    tapered = float(next(m for m in app.metric if m.label == "Race-day Form").value)
    assert tapered > untapered
    app.radio(key="time_constants").set_value("Fitted to my data").run()
    _assert_clean(app)
    app.radio(key="time_constants").set_value(app.radio(key="time_constants").options[0]).run()
    app.selectbox(key="plan_taper").set_value(2).run()


def test_head_to_head_tab_compares_sample_friends(app):
    assert any(tab.label == "Head to head" for tab in app.tabs)
    assert any(m.label == "Runs together" for m in app.metric)
    app.multiselect(key="h2h_friends").set_value(["Sam"]).run()
    _assert_clean(app)
    assert {"You", "Sam"} <= {m.label for m in app.metric}
    app.text_input(key="h2h_me").set_value("Ryan").run()
    app.selectbox(key="period").set_value("Last 4 weeks").run()
    _assert_clean(app)
    app.selectbox(key="period").set_value("All time").run()
    app.text_input(key="h2h_me").set_value("You").run()
    app.multiselect(key="h2h_friends").set_value(["Alex", "Sam"]).run()
    app.selectbox(key="h2h_shared_friend").set_value("Sam").run()
    _assert_clean(app)
    for measure in ("hours", "elevation_m", "runs", "distance_km"):
        app.radio(key="h2h_measure").set_value(measure).run()
        _assert_clean(app)
    app.multiselect(key="h2h_categories").set_value(["distance_km", "runs"]).run()
    _assert_clean(app)
    assert any(m.value.endswith("of 2") for m in app.metric)
    app.multiselect(key="h2h_categories").set_value([]).run()
    _assert_clean(app)
    app.multiselect(key="h2h_categories").set_value([key for key, _, _ in METRICS]).run()
    app.slider(key="h2h_window").set_value(1).run()
    app.slider(key="h2h_tolerance").set_value(5).run()
    _assert_clean(app)
    app.slider(key="h2h_window").set_value(15).run()
    app.slider(key="h2h_tolerance").set_value(15).run()
