"""Display units: kilometres or miles.

Every computation in ``src/`` works in kilometres, metres and minutes per
kilometre.  Units are applied only at the edge -- where a number is formatted
or plotted -- so switching them changes how a result reads, never the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.performance import format_pace

KM_PER_MILE = 1.609344
FEET_PER_METRE = 1.0 / 0.3048


@dataclass(frozen=True)
class Units:
    """One display system.  ``per_km`` turns kilometres into the display distance."""

    name: str
    distance_label: str   # "km" or "mi"
    split_label: str      # "Kilometre" or "Mile"
    per_km: float
    split_m: float        # length of one split, in metres
    elevation_label: str  # "m" or "ft"
    per_metre: float      # metres -> display elevation

    @property
    def pace_label(self) -> str:
        return f"min/{self.distance_label}"

    def distance(self, km):
        """Kilometres (a number, array or Series) in display distance."""
        return km * self.per_km

    def pace(self, min_per_km):
        """Minutes per kilometre in minutes per display distance."""
        return min_per_km / self.per_km

    def format_distance(self, km: float | None, decimals: int = 1) -> str:
        if km is None or not np.isfinite(km):
            return "--"
        return f"{self.distance(km):,.{decimals}f} {self.distance_label}"

    def format_pace(self, min_per_km: float | None) -> str:
        if min_per_km is None or not np.isfinite(min_per_km):
            return "--"
        return format_pace(self.pace(min_per_km), suffix=f" /{self.distance_label}")

    def format_elevation(self, metres: float | None) -> str:
        if metres is None or not np.isfinite(metres):
            return "--"
        return f"{metres * self.per_metre:+.0f} {self.elevation_label}"


METRIC = Units("Kilometres", "km", "Kilometre", 1.0, 1000.0, "m", 1.0)
IMPERIAL = Units("Miles", "mi", "Mile", 1.0 / KM_PER_MILE, KM_PER_MILE * 1000.0, "ft",
                 FEET_PER_METRE)
UNIT_SYSTEMS: dict[str, Units] = {units.name: units for units in (METRIC, IMPERIAL)}
