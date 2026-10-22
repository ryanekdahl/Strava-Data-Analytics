"""Tests for shoe mileage."""

from __future__ import annotations

import pandas as pd

from src.gear import gear_summary, unassigned_share


def make_runs() -> pd.DataFrame:
    return pd.DataFrame({
        "activity_id": [1, 2, 3, 4, 5],
        "date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-06-01", "2024-06-02",
                                "2024-06-03"]),
        "distance_km": [400.0, 350.0, 640.0, 10.0, 8.0],
        "gear": pd.array(["Old", "Old", "Worn", "Fresh", pd.NA], dtype="string"),
    })


def test_status_follows_mileage_and_recency():
    gear = gear_summary(make_runs(), retire_km=700).set_index("gear")
    assert gear.loc["Old", "status"] == "Retired"  # 750 km but unused for months
    assert gear.loc["Worn", "status"] == "Replace soon"  # 640 >= 90% of 700
    assert gear.loc["Fresh", "status"] == "Fine"
    assert gear.loc["Old", "distance_km"] == 750.0
    assert gear.loc["Old", "runs"] == 2
    assert gear_summary(make_runs(), retire_km=600).set_index("gear").loc["Worn", "status"] \
        == "Past limit"


def test_no_gear_column_or_values():
    runs = make_runs()
    assert gear_summary(runs.drop(columns="gear")).empty
    assert gear_summary(runs.assign(gear=pd.NA)).empty
    assert unassigned_share(runs) == 0.2
