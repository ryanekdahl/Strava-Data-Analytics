"""Run replay: frames on the moving clock, and split milestones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import plots, replay
from src.units import IMPERIAL, METRIC


def _route(distance_m, moving_s, hr=150.0) -> pd.DataFrame:
    distance = np.asarray(distance_m, dtype="float64")
    moving = np.asarray(moving_s, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = np.gradient(moving, distance, edge_order=1) * 1000.0 / 60.0
    return pd.DataFrame({
        "lat": 51.5 + distance / 111_000.0, "lon": np.full(len(distance), -0.1),
        "distance_m": distance, "moving_s": moving, "hr": hr,
        "altitude_m": 10.0, "pace_min_per_km": pace,
    })


def test_frames_are_even_in_time_so_slow_stretches_get_more_of_them():
    # 1 km at 4:00/km, then 1 km at 8:00/km.
    route = _route(np.linspace(0, 2000, 201),
                   np.concatenate([np.linspace(0, 240, 101), np.linspace(240, 720, 101)[1:]]))
    frames = replay.replay_frames(route, frames=73)
    assert len(frames) == 73
    assert np.allclose(np.diff(frames["moving_s"]), 10.0)
    assert frames["moving_s"].iloc[0] == 0 and frames["moving_s"].iloc[-1] == pytest.approx(720)
    assert frames["distance_m"].is_monotonic_increasing
    first_km = (frames["distance_m"] < 1000).sum()
    assert first_km == pytest.approx(len(frames) / 3, abs=1)  # a third of the time, half the way
    assert frames["trail"].iloc[0] == 1 and frames["trail"].iloc[-1] == len(route)
    assert frames["lat"].iloc[-1] == pytest.approx(route["lat"].iloc[-1])


def test_frames_tolerate_missing_heart_rate_and_need_a_moving_route():
    route = _route(np.linspace(0, 1000, 11), np.linspace(0, 300, 11), hr=np.nan)
    assert replay.replay_frames(route)["hr"].isna().all()
    assert replay.replay_frames(route.iloc[:1]).empty
    assert replay.replay_frames(_route([0, 0, 0], [0, 0, 0])).empty


def test_milestones_mark_each_whole_split_with_its_time():
    route = _route(np.linspace(0, 3500, 351), np.linspace(0, 3500 * 0.3, 351))  # 5:00/km
    marks = replay.milestones(route, METRIC.split_m)
    assert list(marks["split"]) == [1, 2, 3]
    assert np.allclose(marks["moving_s"], [300, 600, 900])
    assert np.allclose(marks["split_s"], 300)
    assert np.allclose(marks["pace_min_per_km"], 5.0)
    miles = replay.milestones(route, IMPERIAL.split_m)
    assert list(miles["split"]) == [1, 2]
    assert miles["split_s"].iloc[0] == pytest.approx(IMPERIAL.split_m * 0.3)
    assert replay.milestones(route.iloc[:50], METRIC.split_m).empty


@pytest.mark.parametrize("basemap", [False, True])
def test_replay_figure_animates_the_moving_traces(basemap):
    route = _route(np.linspace(0, 5000, 300), np.linspace(0, 1500, 300))
    frames = replay.replay_frames(route, frames=40)
    marks = replay.milestones(route, IMPERIAL.split_m)
    fig = plots.route_replay_figure(route, frames, marks, IMPERIAL, basemap=basemap,
                                    duration_s=8.0)
    assert len(fig.frames) == 40
    assert list(fig.frames[0].traces) == [1, 3, 5, 6]
    assert "Mile 3 split" in fig.frames[-1].layout.title.text
    assert "Mile" not in fig.frames[0].layout.title.text
    assert fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] == 200
    assert len(fig.layout.sliders[0].steps) == 40
