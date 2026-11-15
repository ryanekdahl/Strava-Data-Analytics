"""Tests for finding runs by title and distance."""

from __future__ import annotations

import pandas as pd

from src.run_search import filter_runs

RUNS = pd.DataFrame({
    "activity_id": [1, 2, 3, 4],
    "name": ["Sunday Long Run", "Morning Run", "Track 400s", None],
    "distance_km": [21.1, 8.0, 6.4, 5.0],
})


def ids(frame: pd.DataFrame) -> list[int]:
    return frame["activity_id"].tolist()


def test_empty_search_keeps_every_run():
    assert ids(filter_runs(RUNS)) == [1, 2, 3, 4]


def test_title_search_ignores_case_and_word_order():
    assert ids(filter_runs(RUNS, "long SUNDAY")) == [1]
    assert ids(filter_runs(RUNS, "run")) == [1, 2]
    assert ids(filter_runs(RUNS, "400")) == [3]


def test_every_word_must_match():
    assert ids(filter_runs(RUNS, "morning long")) == []


def test_distance_bounds_are_inclusive_and_optional():
    assert ids(filter_runs(RUNS, min_km=6.4, max_km=21.1)) == [1, 2, 3]
    assert ids(filter_runs(RUNS, max_km=6.4)) == [3, 4]
    assert ids(filter_runs(RUNS, min_km=10)) == [1]


def test_title_and_distance_combine():
    assert ids(filter_runs(RUNS, "run", max_km=10)) == [2]
