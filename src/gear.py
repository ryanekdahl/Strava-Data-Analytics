"""Shoe mileage from the ``Activity Gear`` column of the export.

Running shoes lose cushioning with distance, and the common rule of thumb is
to replace them somewhere between 500 and 800 km.  Strava tracks this per
shoe; this module rebuilds it from the runs themselves.  A shoe not worn in
the last :data:`INACTIVE_DAYS` of the export is treated as retired, since the
export does not say which shoes the athlete has retired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RETIRE_KM = 700.0
WARNING_SHARE = 0.9  # flag a shoe once it has used 90% of its life
INACTIVE_DAYS = 60


def gear_summary(
    activities: pd.DataFrame, retire_km: float = DEFAULT_RETIRE_KM
) -> pd.DataFrame:
    """One row per shoe: runs, distance, first and last use, and a status.

    Returns ``DataFrame[gear, runs, distance_km, first_used, last_used, active,
    remaining_km, status]`` sorted by most recent use.  ``status`` is one of
    ``Retired``, ``Past limit``, ``Replace soon`` or ``Fine``.
    """
    columns = ["gear", "runs", "distance_km", "first_used", "last_used", "active",
               "remaining_km", "status"]
    if "gear" not in activities or activities["gear"].isna().all():
        return pd.DataFrame(columns=columns)
    worn = activities.dropna(subset=["gear"])
    summary = worn.groupby("gear").agg(
        runs=("activity_id", "size"), distance_km=("distance_km", "sum"),
        first_used=("date", "min"), last_used=("date", "max"),
    ).reset_index()
    summary["gear"] = summary["gear"].astype(str)
    cutoff = activities["date"].max() - pd.Timedelta(days=INACTIVE_DAYS)
    summary["active"] = summary["last_used"] >= cutoff
    summary["remaining_km"] = retire_km - summary["distance_km"]
    summary["status"] = np.select(
        [~summary["active"],
         summary["distance_km"] >= retire_km,
         summary["distance_km"] >= WARNING_SHARE * retire_km],
        ["Retired", "Past limit", "Replace soon"], default="Fine",
    )
    return summary.sort_values("last_used", ascending=False).reset_index(drop=True)[columns]


def unassigned_share(activities: pd.DataFrame) -> float:
    """Fraction of runs logged without a shoe."""
    if activities.empty or "gear" not in activities:
        return 0.0
    return float(activities["gear"].isna().mean())
