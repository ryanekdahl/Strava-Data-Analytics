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


def test_route_map_with_and_without_street_tiles(app):
    app.radio(key="route_color").set_value("Heart rate").run()
    _assert_clean(app)
    app.toggle(key="route_basemap").set_value(True).run()
    _assert_clean(app)
