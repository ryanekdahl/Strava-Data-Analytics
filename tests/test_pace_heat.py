"""Pace heat map: bands around the run's median, distance shares, split extremes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import pace_heat, plots
from src.units import IMPERIAL, METRIC


def _route(paces_min_per_km, step_m=50.0) -> pd.DataFrame:
    """A straight route of ``step_m`` stretches, each run at the given pace."""
    paces = np.asarray(paces_min_per_km, dtype="float64")
    distance = np.arange(len(paces)) * step_m
    # Each stretch runs at the pace of the point it starts from.
    moving = np.concatenate([[0.0], np.cumsum(np.nan_to_num(paces[:-1], nan=5.0)
                                              * 60.0 * step_m / 1000.0)])
    return pd.DataFrame({
        "lat": 51.5 + distance / 111_000.0, "lon": np.full(len(paces), -0.1),
        "distance_m": distance, "moving_s": moving, "hr": 150.0,
        "altitude_m": 10.0, "pace_min_per_km": paces,
    })


def test_weighted_quantile_counts_values_by_weight():
    assert pace_heat.weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == pytest.approx(2)
    assert pace_heat.weighted_quantile([4, 6], [9, 1], 0.5) == pytest.approx(4, abs=0.6)
    assert np.isnan(pace_heat.weighted_quantile([np.nan], [1], 0.5))


def test_band_edges_are_whole_five_second_steps_centred_on_the_median():
    edges, step = pace_heat.band_edges(5.0, spread=0.5, n_bands=7)
    assert len(edges) == 6
    assert (step * 60) % 5 == pytest.approx(0)
    assert np.allclose(np.diff(edges), step)
    # The middle band straddles the median.
    assert edges[2] < 5.0 < edges[3]
    # A dead-even run still gets 5-second bands rather than zero-width ones.
    assert pace_heat.band_edges(5.0, 0.0)[1] == pytest.approx(5 / 60)


def test_fast_and_slow_stretches_land_in_opposite_bands():
    # 1 km at 4:30, 2 km at 5:00, 1 km at 5:30 /km.
    route = _route([4.5] * 20 + [5.0] * 40 + [5.5] * 20)
    heat = pace_heat.pace_heat(route)
    assert heat.median == pytest.approx(5.0, abs=0.02)
    segs = heat.segments
    middle = pace_heat.N_BANDS // 2
    assert segs.loc[segs["distance_m"] < 900, "band"].max() < middle
    assert (segs.loc[(segs["distance_m"] > 1100) & (segs["distance_m"] < 2900), "band"]
            == middle).all()
    assert segs.loc[segs["distance_m"] > 3100, "band"].min() > middle
    assert heat.bands["share"].sum() == pytest.approx(1.0)
    assert pace_heat.near_median_share(heat) == pytest.approx(0.5, abs=0.03)
    assert heat.bands["label"].iloc[0].startswith("under ")
    assert heat.bands["label"].iloc[-1].startswith("over ")


def test_missing_pace_is_kept_out_of_the_shares():
    paces = [5.0] * 40
    paces[10] = paces[11] = np.nan
    heat = pace_heat.pace_heat(_route(paces))
    assert (heat.segments["band"] == pace_heat.NO_PACE).sum() == 1
    assert heat.bands["share"].sum() == pytest.approx(1.0)


def test_bands_are_in_the_display_unit():
    route = _route([4.5] * 20 + [5.0] * 40 + [5.5] * 20)
    km, mi = pace_heat.pace_heat(route, METRIC), pace_heat.pace_heat(route, IMPERIAL)
    assert mi.median == pytest.approx(km.median * 1.609344, rel=0.01)
    assert (mi.step * 60) % 5 == pytest.approx(0)


def test_empty_or_tiny_routes_give_an_empty_heat():
    assert pace_heat.pace_heat(_route([5.0])).empty
    assert pace_heat.pace_heat(_route([np.nan] * 10)).empty


def test_split_extremes_find_the_fastest_and_slowest_split():
    # km 1 at 5:00, km 2 at 4:00, km 3 at 6:00, km 4 at 5:00.
    route = _route([5.0] * 20 + [4.0] * 20 + [6.0] * 20 + [5.0] * 21)
    ext = pace_heat.split_extremes(route).set_index("kind")
    assert ext.loc["fastest", "split"] == 2
    assert ext.loc["slowest", "split"] == 3
    assert ext.loc["fastest", "pace"] == pytest.approx(4.0, abs=0.06)
    assert ext.loc["slowest", "pace"] == pytest.approx(6.0, abs=0.06)
    # Halfway through km 2 is about 1.5 km north of the start.
    assert (ext.loc["fastest", "lat"] - 51.5) * 111_000 == pytest.approx(1500, abs=60)
    # Under three whole splits there is nothing worth calling out.
    assert pace_heat.split_extremes(_route([5.0] * 50)).empty


@pytest.mark.parametrize("basemap", [False, True])
@pytest.mark.parametrize("units", [METRIC, IMPERIAL])
def test_heatmap_figure_builds(basemap, units):
    route = _route([5.0] * 20 + [4.0] * 20 + [6.0] * 20 + [np.nan] * 3 + [5.0] * 60)
    heat = pace_heat.pace_heat(route, units)
    fig = plots.pace_heatmap_figure(heat, pace_heat.split_extremes(route, units), units, basemap)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 2  # the distribution legend and the ribbon
    assert list(bars[0].marker.color) == plots.PACE_HEAT_COLORS
    assert "median" in fig.layout.title.text
