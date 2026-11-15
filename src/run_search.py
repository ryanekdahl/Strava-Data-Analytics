"""Find runs by title and distance.

Backs the run pickers in the dashboard: with hundreds of runs a plain dropdown
is unusable, so the list is narrowed by words in the run's title and by a
distance range before the user picks one.
"""

from __future__ import annotations

import pandas as pd


def filter_runs(
    runs: pd.DataFrame,
    query: str = "",
    min_km: float | None = None,
    max_km: float | None = None,
) -> pd.DataFrame:
    """Runs whose title contains every word of ``query`` and whose distance is in range.

    Matching ignores case and word order, so ``"long sunday"`` finds
    "Sunday Long Run".  Bounds are inclusive and in kilometres; ``None`` leaves
    that side open.  Runs without a title only match an empty query.
    """
    keep = pd.Series(True, index=runs.index)
    names = runs["name"].fillna("").astype(str).str.casefold()
    for word in query.casefold().split():
        keep &= names.str.contains(word, regex=False)
    if min_km is not None:
        keep &= runs["distance_km"] >= min_km
    if max_km is not None:
        keep &= runs["distance_km"] <= max_km
    return runs[keep]
