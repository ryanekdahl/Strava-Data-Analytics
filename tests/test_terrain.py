"""Tests for cutting a route into climbs, descents and flat stretches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import plots, streams, terrain


def make_route(altitude, speed=4.0, hr=None, spacing=20.0) -> pd.DataFrame:
    altitude = np.asarray(altitude, dtype="float64")
    d = np.arange(len(altitude)) * spacing
    route = pd.DataFrame({
        "distance_m": d, "moving_s": d / speed, "altitude_m": altitude,
        "hr": np.full(len(d), 150.0) if hr is None else np.asarray(hr, dtype="float64"),
        "pace_min_per_km": 1000.0 / speed / 60.0,
    })
    route["grade"] = streams.route_grade(d, altitude)
    route["gap_pace_min_per_km"] = route["pace_min_per_km"] / streams.grade_cost(route["grade"])
    return route


def hill(rise=20.0, flat=50, up=50, down=50) -> np.ndarray:
    """Flat, a steady climb of ``rise`` metres, a descent back down, flat again."""
    return np.concatenate([np.zeros(flat), np.linspace(0, rise, up), np.linspace(rise, 0, down),
                           np.zeros(flat)])


def test_turning_points_ignore_wobbles_smaller_than_the_threshold():
    noisy = hill() + np.tile([0.0, 2.0], 100)  # 2 m zig-zag on every point
    turns = terrain.turning_points(terrain.smoothed_altitude(make_route(noisy)))
    assert len(turns) == 4  # start, foot, top, end (foot and top bracket the hill)


def test_a_hill_is_one_climb_and_one_descent_between_flats():
    segs = terrain.segments(make_route(hill()))
    kinds = segs["kind"].tolist()
    assert kinds[0] == kinds[-1] == "flat"  # the flat road either side is not part of the hill
    assert [k for k in kinds if k != "flat"] == ["climb", "descent"]
    lead = segs.iloc[0]
    assert lead["distance_m"] == pytest.approx(49 * 20.0, abs=60.0)
    climb = segs.set_index("kind").loc["climb"]
    assert climb["change_m"] == pytest.approx(20.0, abs=1.5)
    assert climb["grade"] > 0
    assert segs["distance_m"].sum() == pytest.approx(199 * 20.0)


def test_gap_is_faster_than_pace_on_the_climb_and_slower_on_the_descent():
    segs = terrain.segments(make_route(hill(rise=40.0))).set_index("kind")
    assert segs.loc["climb", "gap_pace_min_per_km"] < segs.loc["climb", "pace_min_per_km"]
    assert segs.loc["descent", "gap_pace_min_per_km"] > segs.loc["descent", "pace_min_per_km"]


def test_heart_rate_rises_through_a_climb():
    alt = hill(rise=30.0)
    hr = 140.0 + np.clip(alt, 0, None)  # heart rate tracks height
    segs = terrain.segments(make_route(alt, hr=hr)).set_index("kind")
    assert segs.loc["climb", "hr_end"] > segs.loc["climb", "hr_start"]
    assert segs.loc["descent", "hr_end"] < segs.loc["descent", "hr_start"]
    summary = terrain.terrain_summary(terrain.segments(make_route(alt, hr=hr))).set_index("kind")
    assert summary.loc["climb", "avg_hr"] > summary.loc["flat", "avg_hr"]
    assert summary["time_share"].sum() == pytest.approx(1.0)
    assert list(summary.index) == ["climb", "flat", "descent"]


def test_flat_and_altitude_free_routes():
    assert set(terrain.segments(make_route(np.zeros(100)))["kind"]) == {"flat"}
    assert terrain.segments(make_route(np.full(100, np.nan))).empty
    assert terrain.terrain_summary(terrain.segments(make_route(np.full(100, np.nan)))).empty


def test_terrain_figure_shades_every_hill_through_all_panels():
    route = make_route(np.concatenate([hill(), hill()]))
    segs = terrain.segments(route)
    fig = plots.terrain_figure(route, segs)
    hills = (segs["kind"] != "flat").sum()
    assert len(fig.layout.shapes) == hills * 3  # one band per panel
    no_hr = route.assign(hr=np.nan)
    assert len(plots.terrain_figure(no_hr, segs).layout.shapes) == hills * 2
