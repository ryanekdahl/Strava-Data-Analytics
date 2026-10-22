"""Models fitted to *this* athlete instead of population constants.

Two of the dashboard's numbers are textbook defaults: Riegel's fatigue exponent
of 1.06, and the 42- and 7-day time constants behind Fitness and Fatigue.  Both
are averages over other people.  This module estimates each from the athlete's
own history and reports how sure that estimate is, so the dashboard can show
when the data support a personal value and when they cannot tell it apart from
the default.

Personal Riegel exponent -- :func:`fit_riegel`
    Riegel's model ``T = a * D ** b`` is a straight line on log-log axes,
    ``log T = log a + b log D``, so ``b`` is the slope of an ordinary
    least-squares fit of log best-effort time on log distance.  A 95% interval
    comes from the slope's standard error and a t distribution with ``n - 2``
    degrees of freedom.

    Two caveats are stated rather than hidden.  Best efforts at several
    distances often come from the same run, so the points are not independent
    and the interval is optimistic.  And a best "half marathon" from an easy
    long run is not a race effort, which tilts the line steeper than a race
    series would; restricting the fit to the distances actually raced fixes
    that.

Personal Fitness and Fatigue time constants -- :func:`fit_banister`
    Banister's impulse-response model says performance on day ``t`` is

        ``p(t) = p0 + k1 * F_tau1(t-1) - k2 * F_tau2(t-1)``

    where ``F_tau`` is the same zero-seeded exponentially weighted average of
    daily load that drives the Fitness chart, with time constant ``tau``.

    An export starts mid-career: the athlete was usually already fit on its
    first day, and a zero-seeded average would read that as no fitness at all
    and spend months "building" it.  Left alone, that ramp drags the fit toward
    short time constants, which warm up fastest.  So the model carries one more
    term, ``c * (1 - 1/tau1) ** t``: whatever fitness existed on day zero,
    decaying at the fitness rate.  Its size ``c`` is estimated, not assumed.

    For any fixed ``(tau1, tau2)`` the model is linear in ``(p0, k1, k2, c)``,
    so the fit is a grid search over the two time constants with a
    least-squares solve at each point, keeping only fits where both gains are
    positive (fitness must help and fatigue must hurt, or the model has no
    meaning).  A bootstrap over runs gives an interval for each time constant.

    *Performance* needs a number measured on ordinary runs, not only races.
    :func:`performance_index` uses speed relative to heart rate: a pooled
    regression of log speed on average heart rate and log distance, whose
    residual is "how much faster this run was than your heart rate and the
    distance would predict", in percent.  Getting fitter shows up as running
    faster at the same heart rate, which is what this index measures.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.performance import RIEGEL_EXPONENT, best_efforts
from src.streams import best_effort_table
from src.training_load import ATL_DAYS, CTL_DAYS, _ewma, effort_pace

# Two-sided 95% critical values of Student's t, by degrees of freedom.  Written
# out rather than imported so the project keeps its no-scipy dependency list.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}

MIN_RIEGEL_POINTS = 3
MIN_PERFORMANCE_RUNS = 30

TAU1_GRID = tuple(range(10, 91, 2))  # fitness: 10-90 days
TAU2_GRID = tuple(range(2, 26))  # fatigue: 2-25 days
N_BOOTSTRAP = 200
IMPULSE_HORIZON_DAYS = 150


def t_critical(df: int) -> float:
    """95% two-sided t critical value, conservative between tabulated points."""
    if df < 1:
        return float("nan")
    eligible = [k for k in _T_975 if k <= df]
    return _T_975[max(eligible)] if df <= 120 else 1.960


# --------------------------------------------------------------------------
# Riegel
# --------------------------------------------------------------------------

class RiegelFit(NamedTuple):
    exponent: float
    ci_low: float
    ci_high: float
    coefficient: float  # predicted seconds for 1 km
    r2: float
    n: int
    points: pd.DataFrame  # the efforts fitted: distance, distance_km, time_s

    def predict(self, distance_km: float) -> float:
        return float(self.coefficient * distance_km ** self.exponent)

    @property
    def differs_from_default(self) -> bool:
        """Whether the 95% interval excludes Riegel's population value of 1.06."""
        return not (self.ci_low <= RIEGEL_EXPONENT <= self.ci_high)


def riegel_efforts(
    activities: pd.DataFrame,
    stream_best: pd.DataFrame | None = None,
    since: pd.Timestamp | None = None,
    min_km: float = 1.0,
    max_km: float = float("inf"),
) -> pd.DataFrame:
    """Best efforts to fit Riegel's curve to: ``DataFrame[distance, distance_km, time_s]``.

    True in-run bests from the GPS recordings when there are any, otherwise the
    activity-level bests from ``activities.csv``; either way restricted to runs
    since ``since`` (current shape, not an old peak) and distances in
    ``[min_km, max_km]``.
    """
    if stream_best is not None and not stream_best.empty:
        efforts = best_effort_table(stream_best, activities, since)
    else:
        recent = activities if since is None else activities[activities["date"] >= since]
        efforts = best_efforts(recent).rename(columns={"implied_time_s": "time_s"})
    efforts = efforts[efforts["distance_km"].between(min_km, max_km)]
    return efforts[["distance", "distance_km", "time_s"]].reset_index(drop=True)


def fit_riegel(efforts: pd.DataFrame) -> RiegelFit | None:
    """Fit ``T = a * D ** b`` to best efforts by least squares on log-log axes.

    ``efforts`` needs ``distance_km`` and ``time_s`` (and keeps any other
    columns for display).  Returns ``None`` with fewer than three distinct
    distances, where a slope and its uncertainty cannot both be estimated.
    """
    points = efforts.dropna(subset=["distance_km", "time_s"])
    points = points[(points["distance_km"] > 0) & (points["time_s"] > 0)]
    if points["distance_km"].nunique() < MIN_RIEGEL_POINTS:
        return None
    x = np.log(points["distance_km"].to_numpy(dtype="float64"))
    y = np.log(points["time_s"].to_numpy(dtype="float64"))
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    sxx = float(((x - x.mean()) ** 2).sum())
    se = np.sqrt(sse / (n - 2) / sxx) if n > 2 else float("nan")
    margin = t_critical(n - 2) * se
    return RiegelFit(
        exponent=float(slope), ci_low=float(slope - margin), ci_high=float(slope + margin),
        coefficient=float(np.exp(intercept)), r2=1.0 - sse / sst if sst > 0 else float("nan"),
        n=n, points=points.reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Banister
# --------------------------------------------------------------------------

def performance_index(activities: pd.DataFrame) -> pd.DataFrame:
    """Per-run speed relative to heart rate and distance, in percent.

    Fits ``log(speed) = b0 + b1 * avg_hr + b2 * log(distance)`` across every run
    with heart rate (speed from grade-adjusted pace where available) and
    returns each run's residual times 100 -- roughly the percent by which it
    beat what its heart rate and distance predicted.  Returns
    ``DataFrame[activity_id, date, performance]``; empty with fewer than
    :data:`MIN_PERFORMANCE_RUNS` runs.
    """
    columns = ["activity_id", "date", "performance"]
    runs = activities.assign(effort_pace=effort_pace(activities))
    runs = runs.dropna(subset=["avg_hr", "effort_pace"])
    runs = runs[(runs["distance_km"] > 0) & (runs["effort_pace"] > 0)]
    if len(runs) < MIN_PERFORMANCE_RUNS:
        return pd.DataFrame(columns=columns)
    speed = np.log(1000.0 / 60.0 / runs["effort_pace"].to_numpy(dtype="float64"))
    design = np.column_stack([
        np.ones(len(runs)), runs["avg_hr"].to_numpy(dtype="float64"),
        np.log(runs["distance_km"].to_numpy(dtype="float64")),
    ])
    beta, *_ = np.linalg.lstsq(design, speed, rcond=None)
    return pd.DataFrame({
        "activity_id": runs["activity_id"].to_numpy(),
        "date": runs["date"].to_numpy(),
        "performance": (speed - design @ beta) * 100.0,
    })[columns].reset_index(drop=True)


class BanisterFit(NamedTuple):
    tau1: int  # fitness time constant, days
    tau2: int  # fatigue time constant, days
    k1: float
    k2: float
    p0: float
    carry: float  # performance from fitness already held on the export's first day
    r2: float
    r2_default: float  # the same model held at the dashboard's 42 / 7 days
    n: int
    tau1_interval: tuple[float, float]  # bootstrap 5th-95th percentiles
    tau2_interval: tuple[float, float]
    surface: pd.DataFrame  # tau1, tau2, r2 for every admissible pair
    impulse: pd.DataFrame  # day, effect: the modelled response to one session
    positive_day: int | None  # first day a session's net effect turns positive
    peak_day: int | None  # day a session's net effect peaks


def _filtered(daily_load: pd.Series, taus: tuple[int, ...]) -> np.ndarray:
    """``F_tau`` for each tau as columns, shifted so row t holds day t-1's value."""
    values = daily_load.astype("float64").to_numpy()
    columns = [np.concatenate([[0.0], _ewma(values, tau)[:-1]]) for tau in taus]
    return np.column_stack(columns)


def _carryover(n_days: int, taus: tuple[int, ...]) -> np.ndarray:
    """Day-zero fitness decaying at each tau: ``(1 - 1/tau) ** t`` as columns."""
    days = np.arange(n_days)[:, None]
    return (1.0 - 1.0 / np.asarray(taus, dtype="float64")[None, :]) ** days


def _solve_grid(a: np.ndarray, b: np.ndarray, c: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares of ``y ~ 1 + A_i - B_j + C_i`` for every pair ``(i, j)``.

    ``A`` holds the fitness averages, ``B`` the fatigue averages and ``C`` the
    day-zero carryover, one column per time constant.  The weighted Gram matrix
    of every column is built once, and each pair's 4x4 normal equations are
    gathered from it by index and solved in one batched call.  Returns
    ``(coefficients[p, q, 4], sse[p, q])`` with coefficients ``(p0, k1, k2, c)``.
    """
    p, q = a.shape[1], b.shape[1]
    # The fatigue term enters with a minus sign, so k2 comes out positive.
    columns = np.column_stack([np.ones(len(y)), a, -b, c])
    weighted = columns * w[:, None]
    gram = columns.T @ weighted
    moment = weighted.T @ y
    i = np.arange(p)[:, None]
    j = np.arange(q)[None, :]
    index = np.stack(np.broadcast_arrays(np.zeros_like(i + j), 1 + i + 0 * j,
                                         1 + p + j + 0 * i, 1 + p + q + i + 0 * j), axis=-1)
    xtx = gram[index[..., :, None], index[..., None, :]]
    xty = moment[index]
    coef = np.linalg.solve(xtx + np.eye(4) * 1e-9, xty[..., None])[..., 0]
    sse = w @ (y * y) - (coef * xty).sum(-1)
    return coef, sse


def _best_pair(coef, sse, admissible) -> tuple[int, int] | None:
    valid = admissible & (coef[..., 1] > 0) & (coef[..., 2] > 0)
    if not valid.any():
        return None
    masked = np.where(valid, sse, np.inf)
    i, j = np.unravel_index(np.argmin(masked), masked.shape)
    return int(i), int(j)


def _impulse(tau1: int, tau2: int, k1: float, k2: float) -> pd.DataFrame:
    """Modelled performance effect, day by day, of one unit of load on day 0."""
    days = IMPULSE_HORIZON_DAYS
    pulse = np.zeros(days + 1)
    pulse[0] = 1.0
    effect = k1 * np.concatenate([[0.0], _ewma(pulse, tau1)[:-1]]) \
        - k2 * np.concatenate([[0.0], _ewma(pulse, tau2)[:-1]])
    return pd.DataFrame({"day": np.arange(days + 1), "effect": effect})


def fit_banister(
    daily_load: pd.Series,
    performance: pd.DataFrame,
    tau1_grid: tuple[int, ...] = TAU1_GRID,
    tau2_grid: tuple[int, ...] = TAU2_GRID,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> BanisterFit | None:
    """Fit the impulse-response model to a performance series.

    ``daily_load`` is the gap-free daily series from
    :func:`src.training_load.daily_load_series`; ``performance`` has ``date``
    and ``performance`` columns.  Returns ``None`` if there are too few
    performances or no time constants give positive fitness and fatigue gains.
    """
    perf = performance.dropna(subset=["performance"])
    if len(perf) < MIN_PERFORMANCE_RUNS or daily_load.empty:
        return None
    day = daily_load.index.get_indexer(pd.DatetimeIndex(perf["date"]).normalize())
    keep = day > 0  # day 0 has no yesterday to read fitness from
    day = day[keep]
    y = perf["performance"].to_numpy(dtype="float64")[keep]
    if len(y) < MIN_PERFORMANCE_RUNS:
        return None

    a = _filtered(daily_load, tau1_grid)[day]
    b = _filtered(daily_load, tau2_grid)[day]
    c = _carryover(len(daily_load), tau1_grid)[day]
    admissible = np.array(tau1_grid)[:, None] > np.array(tau2_grid)[None, :]
    weights = np.ones(len(y))
    coef, sse = _solve_grid(a, b, c, y, weights)
    best = _best_pair(coef, sse, admissible)
    if best is None:
        return None
    i, j = best
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst

    # The same model at the dashboard's default constants, for comparison.
    defaults = _filtered(daily_load, (CTL_DAYS, ATL_DAYS))[day]
    default_carry = _carryover(len(daily_load), (CTL_DAYS,))[day]
    _, d_sse = _solve_grid(defaults[:, :1], defaults[:, 1:], default_carry, y, weights)
    r2_default = float(1.0 - d_sse[0, 0] / sst)

    # Bootstrap: resample runs with replacement and refit the whole grid.
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
        boot = _best_pair(*_solve_grid(a, b, c, y, counts), admissible)
        if boot is not None:
            picks.append((tau1_grid[boot[0]], tau2_grid[boot[1]]))
    picks_arr = np.array(picks, dtype="float64") if picks else np.full((1, 2), np.nan)

    tau1, tau2 = tau1_grid[i], tau2_grid[j]
    p0, k1, k2, carry = (float(v) for v in coef[i, j])
    surface = pd.DataFrame(
        [(t1, t2, float(r2[ii, jj]) if admissible[ii, jj] and coef[ii, jj, 1] > 0
          and coef[ii, jj, 2] > 0 else np.nan)
         for ii, t1 in enumerate(tau1_grid) for jj, t2 in enumerate(tau2_grid)],
        columns=["tau1", "tau2", "r2"],
    )
    impulse = _impulse(tau1, tau2, k1, k2)
    after = impulse[impulse["day"] >= 1]
    positive = after[after["effect"] > 0]
    return BanisterFit(
        tau1=tau1, tau2=tau2, k1=k1, k2=k2, p0=p0, carry=carry, r2=float(r2[i, j]),
        r2_default=r2_default, n=len(y),
        tau1_interval=tuple(float(v) for v in np.nanpercentile(picks_arr[:, 0], [5, 95])),
        tau2_interval=tuple(float(v) for v in np.nanpercentile(picks_arr[:, 1], [5, 95])),
        surface=surface, impulse=impulse,
        positive_day=int(positive["day"].iloc[0]) if not positive.empty else None,
        peak_day=int(after.loc[after["effect"].idxmax(), "day"])
        if (after["effect"] > 0).any() else None,
    )


def modelled_performance(daily_load: pd.Series, fit: BanisterFit) -> pd.DataFrame:
    """The fitted model's performance prediction for every day: ``DataFrame[date, predicted]``."""
    filtered = _filtered(daily_load, (fit.tau1, fit.tau2))
    carry = _carryover(len(daily_load), (fit.tau1,))[:, 0]
    return pd.DataFrame({
        "date": pd.DatetimeIndex(daily_load.index),
        "predicted": (fit.p0 + fit.k1 * filtered[:, 0] - fit.k2 * filtered[:, 1]
                      + fit.carry * carry),
    })
