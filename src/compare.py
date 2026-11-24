"""Side-by-side comparison of two uploaded Strava exports.

The heavy lifting (metric table, best efforts, predictions, weekly totals) lives
in :mod:`src.head_to_head`; this module decides *which window* each export is
measured over and lines the two up on one axis.

Two exports rarely cover the same dates -- a friend's may start years earlier,
or you may be comparing this year's export with an old one of your own -- so
the window is a choice:

* **Dates both cover** -- the calendar overlap of the two exports, so both are
  measured over exactly the same days (offered only when they overlap).
* **Last N months / All time** -- counted back from *each export's own* latest
  run, so two exports from different years can still be compared fairly on an
  equally long stretch.

When both windows are the same dates, charts use the calendar; otherwise they
use "day N" / "week N" of each window so the lines start together.
"""

from __future__ import annotations

import pandas as pd

from src.head_to_head import cumulative_totals, cut_period, fitness_comparison, period_weeks

SHARED_DATES = "Dates both cover"
RELATIVE_WINDOWS: dict[str, int | None] = {
    "Last 12 months": 365, "Last 6 months": 182, "Last 3 months": 91,
    "Last 4 weeks": 28, "All time": None,
}

Window = tuple[pd.Timestamp, pd.Timestamp]  # start, exclusive end


def _span(runs: pd.DataFrame) -> Window:
    """First day of ``runs`` and the day after the last."""
    return (runs["date"].min().normalize(),
            runs["date"].max().normalize() + pd.Timedelta(days=1))


def shared_window(runs_by_name: dict[str, pd.DataFrame]) -> Window | None:
    """The dates every export covers, or ``None`` when they do not overlap by a full day."""
    spans = [_span(runs) for runs in runs_by_name.values()]
    start = max(span[0] for span in spans)
    end = min(span[1] for span in spans)
    return (start, end) if end > start else None


def window_choices(runs_by_name: dict[str, pd.DataFrame]) -> list[str]:
    """The window options for these exports: shared dates first, when they overlap."""
    shared = [SHARED_DATES] if shared_window(runs_by_name) is not None else []
    return shared + list(RELATIVE_WINDOWS)


def comparison_windows(runs_by_name: dict[str, pd.DataFrame], choice: str) -> dict[str, Window]:
    """Each export's ``(start, end)`` for one of :func:`window_choices`."""
    if choice == SHARED_DATES:
        shared = shared_window(runs_by_name)
        if shared is None:
            raise ValueError("These exports do not cover any of the same dates.")
        return {name: shared for name in runs_by_name}
    days = RELATIVE_WINDOWS[choice]
    windows = {}
    for name, runs in runs_by_name.items():
        first, end = _span(runs)
        start = first if days is None else max(first, end - pd.Timedelta(days=days))
        windows[name] = (start, end)
    return windows


def is_aligned(windows: dict[str, Window]) -> bool:
    """Whether every export is measured over the same dates (so charts can use the calendar)."""
    return len(set(windows.values())) <= 1


def runs_in_windows(runs_by_name: dict[str, pd.DataFrame],
                    windows: dict[str, Window]) -> dict[str, pd.DataFrame]:
    """Each export's runs cut to its own window."""
    return {name: cut_period(runs, *windows[name]) for name, runs in runs_by_name.items()}


def _relabel(frame: pd.DataFrame, column: str, start: pd.Timestamp, step_days: int,
             aligned: bool) -> pd.DataFrame:
    """Swap calendar dates for "day N" / "week N" of the window unless the windows are aligned."""
    if aligned:
        return frame
    return frame.assign(**{column: (frame[column] - start).dt.days // step_days + 1})


def cumulative(runs_by_name: dict[str, pd.DataFrame], windows: dict[str, Window],
               measure: str = "distance_km") -> pd.DataFrame:
    """Running total of ``measure`` through each window: ``DataFrame[date, athlete, <measure>]``.

    ``date`` holds calendar days when the windows are aligned, else day numbers.
    """
    aligned = is_aligned(windows)
    frames = [_relabel(cumulative_totals({name: runs}, *windows[name], measure),
                       "date", windows[name][0], 1, aligned)
              for name, runs in runs_by_name.items()]
    return pd.concat(frames, ignore_index=True)


def weekly(runs_by_name: dict[str, pd.DataFrame], windows: dict[str, Window],
           measure: str = "distance_km") -> pd.DataFrame:
    """Each export's weekly ``measure`` side by side: ``DataFrame[week_start, <athlete>...]``.

    ``week_start`` holds Monday dates when the windows are aligned, else week
    numbers (week 1 is the week each window starts in).  Weeks one export has
    and the other does not are left empty.
    """
    aligned = is_aligned(windows)
    columns = {}
    for name, runs in runs_by_name.items():
        start, end = windows[name]
        weeks = period_weeks(runs, start, end)
        monday = start.normalize() - pd.Timedelta(days=start.weekday())
        weeks = _relabel(weeks, "week_start", monday, 7, aligned)
        columns[name] = weeks.set_index("week_start")[measure]
    return pd.DataFrame(columns).rename_axis("week_start").reset_index()


def fitness_lines(fitness_by_name: dict[str, pd.DataFrame],
                  windows: dict[str, Window]) -> pd.DataFrame:
    """Each export's Fitness (CTL) through its window: ``DataFrame[date, athlete, ctl]``."""
    aligned = is_aligned(windows)
    frames = [_relabel(fitness_comparison({name: fitness}, *windows[name]),
                       "date", windows[name][0], 1, aligned)
              for name, fitness in fitness_by_name.items()]
    return pd.concat(frames, ignore_index=True)


def distinct_names(names: list[str], defaults: list[str]) -> list[str]:
    """Display names: blanks fall back to ``defaults``, and a repeated name gets a ``(2)``."""
    out: list[str] = []
    for name, default in zip(names, defaults):
        name = name.strip() or default
        candidate, n = name, 2
        while candidate in out:
            candidate, n = f"{name} ({n})", n + 1
        out.append(candidate)
    return out
