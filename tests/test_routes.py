"""Most frequent routes: matching runs by GPS track, grouping, and route PRs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import plots, routes
from src.units import IMPERIAL, METRIC

LAT0, LON0 = 51.5, -0.12
M_PER_DEG = routes._EARTH_M_PER_DEG


def _track(east_m, north_m, jitter_m=0.0, seed=0) -> pd.DataFrame:
    """A track from east/north metres offsets around a fixed start, densely sampled."""
    east, north = np.asarray(east_m, float), np.asarray(north_m, float)
    rng = np.random.default_rng(seed)
    east = east + rng.normal(0, jitter_m, len(east))
    north = north + rng.normal(0, jitter_m, len(north))
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))])
    return pd.DataFrame({
        "lat": LAT0 + north / M_PER_DEG,
        "lon": LON0 + east / (M_PER_DEG * np.cos(np.radians(LAT0))),
        "distance_m": distance,
    })


def _loop(radius_m=800.0, n=300, reverse=False):
    theta = np.linspace(0, 2 * np.pi, n)
    if reverse:
        theta = -theta
    return radius_m * (np.cos(theta) - 1), radius_m * np.sin(theta)


def _out_and_back(length_m=2500.0, n=300):
    out = np.linspace(0, length_m, n // 2)
    north = np.concatenate([out, out[::-1]])
    east = 40.0 * np.sin(north / 300.0)
    return east, north


def test_jittered_copies_of_one_loop_match_and_a_detour_does_not():
    a = routes.signature(_track(*_loop(), jitter_m=8, seed=1), LAT0)
    b = routes.signature(_track(*_loop(), jitter_m=8, seed=2), LAT0)
    assert routes.same_route(a, b)
    # Same loop, but a 500 m spur straight out from the loop and back, a quarter of the
    # way round (where the loop's top is, so "out" is north).
    east, north = _loop()
    q = len(east) // 4
    spur = np.linspace(0, 500, 40)
    detour = _track(np.concatenate([east[:q], np.full(80, east[q]), east[q:]]),
                    np.concatenate([north[:q], north[q] + spur, north[q] + spur[::-1], north[q:]]))
    assert not routes.same_route(a, routes.signature(detour, LAT0))


def test_group_routes_finds_the_regulars_and_leaves_one_offs_out():
    tracks = {}
    for i in range(5):
        tracks[100 + i] = _track(*_loop(), jitter_m=6, seed=i)
    for i in range(3):
        tracks[200 + i] = _track(*_out_and_back(), jitter_m=6, seed=10 + i)
    tracks[300] = _track(*_loop(radius_m=1600))           # same start, twice the size
    tracks[301] = _track(*_out_and_back(length_m=2500)[::-1])  # off east, not north
    tracks[400] = _track(*_loop(reverse=True), jitter_m=6, seed=20)  # the loop, other way
    groups = routes.group_routes(tracks)
    by_route = groups.groupby("route_id")["activity_id"].apply(set).to_dict()
    assert by_route[1] == {100, 101, 102, 103, 104, 400}
    assert by_route[2] == {200, 201, 202}
    assert set(groups["activity_id"]) .isdisjoint({300, 301})
    assert groups.groupby("route_id")["is_reference"].sum().eq(1).all()


def test_group_routes_never_chains_runs_that_do_not_match_directly():
    # Each loop is 70 m east of the last: neighbours match, the ends do not.
    tracks = {}
    for i in range(5):
        east, north = _loop()
        tracks[i + 1] = _track(east + 70 * i, north)
    a, e = (routes.signature(tracks[k], LAT0) for k in (1, 5))
    assert not routes.same_route(a, e)
    groups = routes.group_routes(tracks, min_runs=2)
    for _, members in groups.groupby("route_id"):
        ref = members.loc[members["is_reference"], "activity_id"].iloc[0]
        for other in members["activity_id"]:
            assert routes.same_route(routes.signature(tracks[ref], LAT0),
                                     routes.signature(tracks[other], LAT0))


def test_too_few_runs_give_no_routes():
    tracks = {1: _track(*_loop()), 2: _track(*_loop(), jitter_m=5)}
    assert routes.group_routes(tracks).empty
    assert routes.group_routes({}).empty


def test_route_shapes():
    assert routes.route_shape(_track(*_loop())) == "loop"
    assert routes.route_shape(_track(*_out_and_back())) == "out-and-back"
    east = np.linspace(0, 3000, 200)
    assert routes.route_shape(_track(east, 0 * east)) == "point-to-point"


def test_route_name_prefers_a_recurring_custom_title():
    names = pd.Series(["Canal loop", "Canal loop", "Morning Run", "Canal loop"])
    assert routes.route_name(names, 6.0, "loop", "6.0 km") == "Canal loop"
    generic = pd.Series(["Morning Run"] * 4 + ["Tempo"])
    assert routes.route_name(generic, 6.0, "loop", "6.0 km") == "6.0 km loop"


def _activities(ids, paces, distances=None):
    distances = distances or [6.0] * len(ids)
    return pd.DataFrame({
        "activity_id": ids,
        "date": pd.date_range("2026-01-01", periods=len(ids), freq="7D"),
        "name": ["Canal loop"] * len(ids),
        "distance_km": distances,
        "moving_time_s": [p * 60 * d for p, d in zip(paces, distances)],
    })


def test_route_pr_is_the_fastest_pace_and_the_progression_steps_down():
    tracks = {i: _track(*_loop(), jitter_m=5, seed=i) for i in range(1, 6)}
    groups = routes.group_routes(tracks)
    # Run 4 is the fastest by pace even though run 5 is shorter and so quicker in time.
    acts = _activities([1, 2, 3, 4, 5], [6.0, 5.8, 5.9, 5.5, 5.6],
                       distances=[6.0, 6.0, 6.0, 6.2, 5.8])
    efforts = routes.route_efforts(groups, acts, 1)
    assert efforts.loc[efforts["is_pr"], "activity_id"].tolist() == [4]
    assert efforts["best_so_far"].tolist() == pytest.approx([6.0, 5.8, 5.8, 5.5, 5.5])
    table = routes.route_table(groups, acts, tracks)
    row = table.iloc[0]
    assert row["pr_id"] == 4 and row["runs"] == 5 and row["name"] == "Canal loop"
    assert row["shape"] == "loop"
    assert row["pr_time_s"] == pytest.approx(5.5 * 60 * 6.0)  # carried over the median 6 km


def test_route_table_counts_only_the_runs_given():
    tracks = {i: _track(*_loop(), jitter_m=5, seed=i) for i in range(1, 6)}
    groups = routes.group_routes(tracks)
    acts = _activities([1, 2, 3, 4, 5], [6.0, 5.8, 5.9, 5.5, 5.6])
    assert routes.route_table(groups, acts.iloc[:3], tracks).iloc[0]["pr_id"] == 2
    assert routes.route_table(groups, acts.iloc[:2], tracks).empty  # under three runs


@pytest.mark.parametrize("basemap", [False, True])
@pytest.mark.parametrize("units", [METRIC, IMPERIAL])
def test_route_figures_build(basemap, units):
    tracks = {i: _track(*_loop(), jitter_m=5, seed=i) for i in range(1, 6)}
    groups = routes.group_routes(tracks)
    acts = _activities([1, 2, 3, 4, 5], [6.0, 5.8, 5.9, 5.5, 5.6])
    table = routes.route_table(groups, acts, tracks)
    overview = plots.routes_overview_figure(table, tracks, units, basemap, selected=1)
    assert overview.data[0].line.color == plots.route_color(1)
    history = plots.route_history_figure(routes.route_efforts(groups, acts, 1), units)
    assert any("PR" in a.text for a in history.layout.annotations)
