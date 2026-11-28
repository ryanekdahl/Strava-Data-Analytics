"""Sleep score and readiness from Apple Health, lined up against each run.

Strava knows how fast you ran and how hard your heart worked; Apple Health
knows how you slept and how recovered you looked that morning.  This module
imports the second and asks: **do I run faster, or at a lower heart rate, after
a better night?**

Input
-----
* **Apple Health's export** -- the ``export.zip`` from Health > profile >
  *Export All Health Data*, unchanged, or the ``export.xml`` inside it.
* **A CSV** with a date column and a sleep-score and/or readiness column, for
  apps that compute their own scores (Athlytic, Bevel, Oura, Whoop, Health Auto
  Export...).  Names are matched loosely: ``date,sleep_score,readiness`` works,
  and so do ``Sleep Score`` or ``Recovery %``.

What is read from Apple Health
------------------------------
The export is several GB of XML for a long-time Watch wearer, so it is scanned
line by line (Apple writes one ``<Record>`` per line) and only four kinds of
record are parsed; nothing is written to disk.

====================================  ===========================================
``...SleepAnalysis``                  sleep stages (Core / Deep / REM / Awake)
``...HeartRateVariabilitySDNN``       overnight HRV readings
``...RestingHeartRate``               one resting heart rate per day
any type naming *SleepScore* or       a score the Watch or another app wrote --
*Readiness*                           used as-is wherever one exists
====================================  ===========================================

**Every date is the morning the number describes**, so a run is paired with the
night before it.  A sleep segment ending at or after 15:00 belongs to the next
morning (``date = (end + 9 h).normalize()``).

Sleep score (0-100), when Apple Health has none of its own
----------------------------------------------------------
Segments from the source with the most sleep that night (Watch and iPhone both
write sleep, and summing them would double count) are grouped into blocks split
by gaps over 2 h; the block with the most sleep is the night, so a nap is
ignored.  The score adds three parts, weighted like Apple's own Sleep Score:

* **Duration, 50 pts** -- 0 at 4 h asleep, rising linearly to 50 at 8 h or more.
* **Bedtime, 30 pts** -- full marks within 30 min of your median bedtime over
  the previous 14 nights (at least 5), falling to 0 at 2 h off.
* **Interruptions, 20 pts** -- time awake between falling asleep and the final
  wake-up: 20 at none, 0 at 60 min or more.

A part that cannot be computed (no bedtime history yet) is left out and the
others are rescaled to 100.

Readiness (0-100), when Apple Health has none of its own
--------------------------------------------------------
How the morning's overnight HRV and resting heart rate compare with *your own*
previous 60 days (at least 14 readings), as z-scores capped at +/-3:

    readiness = 70 + 15 * (0.6 * z(ln HRV) - 0.4 * z(resting HR))

HRV is log-transformed because it is right-skewed.  Readings from 20:00 the
evening before to 10:00 count as that morning's.  70 is a typical day; each
standard deviation better than usual adds about 10.  Sleep is deliberately
left out so the two scores measure different things.

The run metrics
---------------
``pace_min_per_km`` (lower is faster), ``avg_hr``, and ``efficiency`` -- metres
covered per heartbeat, ``speed / avg_hr`` -- which rises when you go faster for
the same heart rate, so an easy and a hard run are on a fairer footing.

Correlation is Pearson's ``r`` with a 95% interval from Fisher's z
transformation, ``tanh(atanh(r) +/- 1.96 / sqrt(n - 3))``.  It shows
association, not cause: a hard week, a late night and a slow run can share a
third cause (travel, heat, illness), and getting fitter over months moves pace
regardless of sleep.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

from src.recovery import _normalize, _to_number, strength_label

WELLNESS_COLUMNS = ["date", "sleep_score", "readiness"]
WELLNESS_METRICS = {"sleep_score": "Sleep score", "readiness": "Readiness"}
RUN_METRICS = {
    "pace_min_per_km": "Pace",
    "avg_hr": "Average heart rate",
    "efficiency": "Metres per heartbeat",
}
# Runs grouped by distance, so easy runs and long runs can be compared on their own.
RUN_GROUPS: dict[str, tuple[float, float]] = {
    "All runs": (0.0, np.inf),
    "Short": (0.0, 8.0),
    "Medium": (8.0, 15.0),
    "Long": (15.0, np.inf),
}

MIN_PAIRS = 10  # below this, r is noise and is reported as missing
EVENING_HOUR = 15  # a night ending at/after this hour belongs to the next morning
_NIGHT_SHIFT = pd.Timedelta(hours=24 - EVENING_HOUR)
NIGHT_GAP = pd.Timedelta(hours=2)  # a gap longer than this starts a new sleep block

TARGET_SLEEP_H, MIN_SLEEP_H = 8.0, 4.0
BEDTIME_FULL_MIN, BEDTIME_ZERO_MIN = 30.0, 120.0
BEDTIME_NIGHTS, BEDTIME_MIN_NIGHTS = 14, 5
AWAKE_ZERO_MIN = 60.0
POINTS = {"duration": 50.0, "bedtime": 30.0, "interruptions": 20.0}

BASELINE_DAYS, BASELINE_MIN = 60, 14
READINESS_CENTER, READINESS_SCALE = 70.0, 15.0
HRV_WEIGHT, RHR_WEIGHT = 0.6, 0.4
Z_CAP = 3.0

TEMPLATE_CSV = "date,sleep_score,readiness\n2026-03-01,82,74\n2026-03-02,75,61\n"

_SLEEP = b"HKCategoryTypeIdentifierSleepAnalysis"
_HRV = b"HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
_RHR = b"HKQuantityTypeIdentifierRestingHeartRate"
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
# The opening tag of every record this module reads; Apple writes ``type`` first.
_RECORD = re.compile(rb'<Record type="(HK(?:CategoryTypeIdentifierSleepAnalysis'
                     rb'|QuantityTypeIdentifier(?:HeartRateVariabilitySDNN|RestingHeartRate)))'
                     rb'"[^>]*>')
_ANY_RECORD = re.compile(rb'<Record type="([^"]*)"[^>]*>')
# Score records are rare and their type names vary, so they are found by these words.
_SCORE_WORDS = (b"leepScore", b"leepscore", b"eadiness")
CHUNK_BYTES = 32 * 1024**2
_ASLEEP_VALUES = ("Asleep", "AsleepUnspecified", "AsleepCore", "AsleepDeep", "AsleepREM")


class WellnessError(ValueError):
    """The upload has no usable sleep or readiness data; the message says why."""


# --------------------------------------------------------------------------- reading


def _open_export(source: bytes | BinaryIO, name: str) -> tuple[str, BinaryIO]:
    """Return ``(kind, stream)`` with kind ``"xml"`` or ``"csv"``.

    A zip is opened in place, and its ``export.xml`` (Apple's) or first CSV is
    streamed from it without being extracted.
    """
    if isinstance(source, bytes):
        stream = io.BytesIO(source)
    else:
        stream = source
    lower = name.lower()
    if lower.endswith(".zip"):
        try:
            archive = zipfile.ZipFile(stream)
        except zipfile.BadZipFile as err:
            raise WellnessError("That file is not a valid zip archive.") from err
        members = [m for m in archive.namelist() if not m.startswith("__MACOSX")]
        xml = [m for m in members if m.lower().rsplit("/", 1)[-1] == "export.xml"]
        if xml:
            return "xml", archive.open(xml[0])
        csv = [m for m in members if m.lower().endswith(".csv")]
        if csv:
            return "csv", archive.open(csv[0])
        raise WellnessError("No export.xml (Apple Health) or CSV found inside the zip.")
    if lower.endswith(".xml"):
        return "xml", stream
    if lower.endswith(".csv"):
        return "csv", stream
    raise WellnessError("Upload Apple Health's export.zip, its export.xml, or a CSV.")


def load_wellness(source: str | Path | bytes | BinaryIO, name: str = "") -> pd.DataFrame:
    """Read any supported upload into ``DataFrame[date, sleep_score, readiness, ...]``.

    One row per morning.  Apple Health exports also carry ``asleep_h``,
    ``hrv_ms``, ``resting_hr`` and the ``*_native`` flags saying which scores
    came from Apple Health itself rather than this module's formulas.
    Raises :class:`WellnessError` when nothing usable is found.  A file object
    passed in (e.g. a Streamlit upload) is read but left open for its owner.
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            return load_wellness(handle, name or str(source))
    kind, stream = _open_export(source, name)
    try:
        frame = (parse_apple_health(stream) if kind == "xml"
                 else load_wellness_csv(stream.read()))
    finally:
        if stream is not source:
            stream.close()
    if frame.empty or frame[list(WELLNESS_METRICS)].isna().all().all():
        raise WellnessError("No sleep score, sleep stages, HRV or resting heart rate "
                            "readings were found in that file.")
    return frame


def _chunks(stream: BinaryIO | Iterable[bytes], size: int = CHUNK_BYTES) -> Iterable[bytes]:
    """The stream in large pieces that always end on a line break."""
    if not hasattr(stream, "read"):
        yield b"".join(stream)
        return
    carry = b""
    while True:
        block = stream.read(size)
        if not block:
            if carry:
                yield carry
            return
        block = carry + block
        cut = block.rfind(b"\n") + 1
        if cut == 0:
            carry = block
            continue
        carry = block[cut:]
        yield block[:cut]


def _lines_with(chunk: bytes, words: tuple[bytes, ...]) -> list[bytes]:
    """Each distinct line of ``chunk`` containing any of ``words`` (fast C-level search)."""
    starts = set()
    for word in words:
        position = chunk.find(word)
        while position != -1:
            start = chunk.rfind(b"\n", 0, position) + 1
            starts.add(start)
            end = chunk.find(b"\n", position)
            position = chunk.find(word, end if end != -1 else len(chunk))
    lines = []
    for start in sorted(starts):
        end = chunk.find(b"\n", start)
        lines.append(chunk[start:end if end != -1 else len(chunk)])
    return lines


def scan_records(stream: BinaryIO | Iterable[bytes]) -> dict[str, pd.DataFrame]:
    """Pull the records this module uses out of Apple Health's XML.

    Returns frames keyed ``sleep`` (source, value, start, end), ``hrv`` and
    ``rhr`` (start, value) and ``scores`` (kind, end, value).  Compiled
    patterns and byte searches run over large chunks of the file rather than
    an XML parser or a Python loop per line: it is many times faster on a multi-GB export, and
    some iOS versions wrote export headers that strict parsers reject.
    """
    sleep, hrv, rhr, scores = [], [], [], []
    for chunk in _chunks(stream):
        for match in _RECORD.finditer(chunk):
            record_type = match.group(1)
            attrs = dict(_ATTR.findall(match.group(0)))
            try:
                if record_type == _SLEEP:
                    value = attrs[b"value"].decode().replace("HKCategoryValueSleepAnalysis", "")
                    sleep.append((attrs.get(b"sourceName", b"").decode(errors="replace"), value,
                                  attrs[b"startDate"][:19], attrs[b"endDate"][:19]))
                elif record_type == _HRV:
                    hrv.append((attrs[b"startDate"][:19], float(attrs[b"value"])))
                elif record_type == _RHR:
                    rhr.append((attrs[b"startDate"][:19], float(attrs[b"value"])))
            except (KeyError, ValueError):
                continue  # one malformed record never sinks the import
        for line in _lines_with(chunk, _SCORE_WORDS):
            match = _ANY_RECORD.search(line)
            if match is None:
                continue
            record_type = match.group(1).lower()  # the words may also sit in metadata
            kind = ("sleep_score" if b"sleepscore" in record_type
                    else "readiness" if b"readiness" in record_type else None)
            attrs = dict(_ATTR.findall(match.group(0)))
            if kind is None or b"value" not in attrs or b"endDate" not in attrs:
                continue
            try:
                scores.append((kind, attrs[b"endDate"][:19], float(attrs[b"value"])))
            except ValueError:
                continue

    def frame(rows: list, columns: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=columns)
        for column in ("start", "end"):
            if column in out:  # local wall-clock time; Apple's UTC offset is dropped
                out[column] = pd.to_datetime(out[column].str.decode("ascii"),
                                             format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if "value" in out and columns[-1] == "value":
            out["value"] = out["value"].astype("float64")
        return out

    return {
        "sleep": frame(sleep, ["source", "value", "start", "end"]),
        "hrv": frame(hrv, ["start", "value"]),
        "rhr": frame(rhr, ["start", "value"]),
        "scores": frame(scores, ["kind", "end", "value"]),
    }


def parse_apple_health(stream: BinaryIO | Iterable[bytes]) -> pd.DataFrame:
    """Daily sleep score and readiness from Apple Health's ``export.xml``.

    Apple's own scores are used wherever they exist; other mornings get the
    formulas in the module docstring.
    """
    records = scan_records(stream)
    nights = nightly_sleep(records["sleep"])
    nights["sleep_score"] = sleep_score(nights)
    daily = morning_hrv_rhr(records["hrv"], records["rhr"])
    daily["readiness"] = readiness_score(daily)

    frame = nights[["date", "asleep_h", "sleep_score"]].merge(
        daily[["date", "hrv_ms", "resting_hr", "readiness"]], on="date", how="outer")
    frame = frame.rename_axis(None).reset_index(drop=True)
    for kind in WELLNESS_METRICS:
        native = native_scores(records["scores"], kind).rename(f"{kind}_apple")
        frame = frame.merge(native.rename_axis("date").reset_index(), on="date", how="outer")
        frame[f"{kind}_native"] = frame[f"{kind}_apple"].notna()
        frame[kind] = frame[f"{kind}_apple"].combine_first(frame[kind])
        frame = frame.drop(columns=f"{kind}_apple")
    frame = frame.dropna(subset=list(WELLNESS_METRICS), how="all")
    order = WELLNESS_COLUMNS + ["asleep_h", "hrv_ms", "resting_hr",
                                "sleep_score_native", "readiness_native"]
    return frame.sort_values("date").reset_index(drop=True)[order]


def native_scores(scores: pd.DataFrame, kind: str) -> pd.Series:
    """Apple Health's own daily ``sleep_score`` or ``readiness``, indexed by morning."""
    rows = scores[scores["kind"] == kind]
    if rows.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"))
    morning = (rows["end"] + _NIGHT_SHIFT).dt.normalize()
    values = rows["value"].where(rows["value"].between(0, 100))
    if values.max() <= 1.0:  # stored as a fraction
        values = values * 100.0
    return values.groupby(morning.rename("date")).mean().dropna()


# --------------------------------------------------------------------------- sleep


def nightly_sleep(sleep: pd.DataFrame) -> pd.DataFrame:
    """One row per morning: ``date, asleep_h, awake_min, onset, wake`` of the main sleep."""
    columns = ["date", "asleep_h", "awake_min", "onset", "wake"]
    asleep = sleep[sleep["value"].isin(_ASLEEP_VALUES)].copy()
    asleep = asleep[asleep["end"] > asleep["start"]]
    if asleep.empty:
        return pd.DataFrame(columns=columns).astype({"date": "datetime64[ns]"})
    asleep["date"] = (asleep["end"] + _NIGHT_SHIFT).dt.normalize()
    asleep["seconds"] = (asleep["end"] - asleep["start"]).dt.total_seconds()

    # Keep one source per night: the one that recorded the most sleep.
    totals = asleep.groupby(["date", "source"])["seconds"].sum().reset_index()
    best = totals.sort_values("seconds").drop_duplicates("date", keep="last")
    asleep = asleep.merge(best[["date", "source"]], on=["date", "source"])

    rows = []
    for day, night in asleep.sort_values("start").groupby("date"):
        reach = night["end"].cummax().shift()
        block = (night["start"] - reach > NIGHT_GAP).cumsum()
        main = max((part for _, part in night.groupby(block)),
                   key=lambda part: part["seconds"].sum())
        onset, wake = main["start"].min(), main["end"].max()
        asleep_s = _covered_seconds(main)
        awake_min = max((wake - onset).total_seconds() - asleep_s, 0.0) / 60.0
        rows.append((day, asleep_s / 3600.0, awake_min, onset, wake))
    return pd.DataFrame(rows, columns=columns)


def _covered_seconds(segments: pd.DataFrame) -> float:
    """Total time covered by possibly overlapping segments."""
    total, current_end = 0.0, None
    for start, end in zip(segments["start"], segments["end"]):
        if current_end is None or start > current_end:
            total += (end - start).total_seconds()
            current_end = end
        elif end > current_end:
            total += (end - current_end).total_seconds()
            current_end = end
    return total


def sleep_score(nights: pd.DataFrame) -> pd.Series:
    """The 0-100 sleep score in the module docstring, one per night."""
    if nights.empty:
        return pd.Series(dtype="float64")
    duration = np.clip((nights["asleep_h"] - MIN_SLEEP_H) / (TARGET_SLEEP_H - MIN_SLEEP_H),
                       0.0, 1.0)
    interruptions = 1.0 - np.clip(nights["awake_min"] / AWAKE_ZERO_MIN, 0.0, 1.0)

    # Bedtime as minutes after 18:00 the evening before, so 23:30 and 00:30 are an hour apart.
    evening = nights["date"] - pd.Timedelta(hours=6)
    bedtime = (nights["onset"] - evening).dt.total_seconds() / 60.0
    usual = (bedtime.rolling(BEDTIME_NIGHTS, min_periods=BEDTIME_MIN_NIGHTS).median().shift())
    off = (bedtime - usual).abs()
    regularity = 1.0 - np.clip((off - BEDTIME_FULL_MIN) / (BEDTIME_ZERO_MIN - BEDTIME_FULL_MIN),
                               0.0, 1.0)

    earned = (POINTS["duration"] * duration + POINTS["interruptions"] * interruptions
              + (POINTS["bedtime"] * regularity).fillna(0.0))
    available = (POINTS["duration"] + POINTS["interruptions"]
                 + np.where(regularity.notna(), POINTS["bedtime"], 0.0))
    return (100.0 * earned / available).round(1)


# --------------------------------------------------------------------------- readiness


def morning_hrv_rhr(hrv: pd.DataFrame, rhr: pd.DataFrame) -> pd.DataFrame:
    """``date, hrv_ms, resting_hr``: overnight HRV mean and that day's resting HR."""
    hour = hrv["start"].dt.hour
    overnight = hrv[(hour >= 20) | (hour < 10)]
    hrv_day = (overnight["start"] + pd.to_timedelta((overnight["start"].dt.hour >= 20)
                                                    .astype(int), unit="D")).dt.normalize()
    hrv_daily = overnight["value"].where(overnight["value"].between(5, 300)).groupby(
        hrv_day.rename("date")).mean().rename("hrv_ms")
    rhr_daily = rhr["value"].where(rhr["value"].between(25, 140)).groupby(
        rhr["start"].dt.normalize().rename("date")).mean().rename("resting_hr")
    daily = pd.concat([hrv_daily, rhr_daily], axis=1).sort_index()
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily.reset_index()


def _baseline_z(series: pd.Series) -> pd.Series:
    """Each day against the previous ``BASELINE_DAYS`` calendar days, as a capped z-score."""
    trailing = series.rolling(f"{BASELINE_DAYS}D", min_periods=BASELINE_MIN, closed="left")
    z = (series - trailing.mean()) / trailing.std()
    return z.replace([np.inf, -np.inf], np.nan).clip(-Z_CAP, Z_CAP)


def readiness_score(daily: pd.DataFrame) -> pd.Series:
    """The 0-100 readiness in the module docstring, aligned to ``daily``'s rows."""
    if daily.empty:
        return pd.Series(dtype="float64")
    indexed = daily.set_index("date")
    z_hrv = _baseline_z(np.log(indexed["hrv_ms"].dropna())).reindex(indexed.index)
    z_rhr = _baseline_z(indexed["resting_hr"].dropna()).reindex(indexed.index)
    both = HRV_WEIGHT * z_hrv - RHR_WEIGHT * z_rhr
    # With only one signal on a day, it carries the whole score at the same total weight.
    composite = both.fillna(z_hrv).fillna(-z_rhr)
    score = READINESS_CENTER + READINESS_SCALE * composite
    return pd.Series(score.clip(0.0, 100.0).round(1).to_numpy(), index=daily.index)


# --------------------------------------------------------------------------- CSV


def _find_column(columns: list[str], wanted: str) -> str | None:
    normalized = {column: _normalize(column) for column in columns}
    if wanted == "date":
        for target in ("date", "day", "calendardate", "cyclestarttime", "summarydate"):
            for column, norm in normalized.items():
                if norm == target:
                    return column
        return next((c for c, n in normalized.items() if "date" in n), None)
    smoothed = ("baseline", "avg7", "7d", "7day", "weekly", "average", "contributor")
    candidates = [c for c, n in normalized.items() if not any(s in n for s in smoothed)]
    if wanted == "sleep_score":
        exact = [c for c in candidates if normalized[c] == "sleepscore"]
        loose = [c for c in candidates if "sleep" in normalized[c]
                 and ("score" in normalized[c] or "performance" in normalized[c])]
    else:
        exact = [c for c in candidates if normalized[c] in ("readiness", "readinessscore")]
        loose = [c for c in candidates if "sleep" not in normalized[c]
                 and ("readiness" in normalized[c] or "recovery" in normalized[c])]
    return next(iter(exact + loose), None)


def load_wellness_csv(content: bytes) -> pd.DataFrame:
    """A ``date, sleep_score, readiness`` CSV (or a vendor export), one row per morning.

    Timestamps at or after 15:00 describe the next morning; several rows on one
    day are averaged; values outside 0-100 are dropped.
    """
    raw = pd.read_csv(io.BytesIO(content))
    raw.columns = [str(column).strip() for column in raw.columns]
    columns = list(raw.columns)
    date_column = _find_column(columns, "date")
    if date_column is None:
        raise WellnessError(f"No date column found in {columns}.")
    metric_columns = {key: _find_column(columns, key) for key in WELLNESS_METRICS}
    if all(column is None for column in metric_columns.values()):
        raise WellnessError(f"No sleep-score or readiness column found in {columns}.")

    stamp = pd.to_datetime(raw[date_column], format="mixed", errors="coerce")
    if isinstance(stamp.dtype, pd.DatetimeTZDtype):
        stamp = stamp.dt.tz_localize(None)
    day = (stamp + _NIGHT_SHIFT).dt.normalize()
    tidy = pd.DataFrame({"date": day})
    for key, column in metric_columns.items():
        values = (_to_number(raw[column]) if column is not None
                  else pd.Series(np.nan, index=raw.index, dtype="float64"))
        tidy[key] = values.where(values.between(0.0, 100.0))
    tidy = tidy.dropna(subset=["date"]).dropna(subset=list(WELLNESS_METRICS), how="all")
    daily = tidy.groupby("date", as_index=False)[list(WELLNESS_METRICS)].mean()
    return daily.sort_values("date").reset_index(drop=True)[WELLNESS_COLUMNS]


# --------------------------------------------------------------------------- pairing


def join_runs(wellness: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Each run with that morning's sleep score and readiness, plus its efficiency.

    Runs with neither score are dropped.  Returns ``activity_id, date, name,
    distance_km, pace_min_per_km, avg_hr, efficiency, sleep_score, readiness``.
    """
    columns = ["activity_id", "date", "name", "distance_km", "pace_min_per_km", "avg_hr",
               "efficiency", *WELLNESS_METRICS]
    if runs.empty or wellness.empty:
        return pd.DataFrame(columns=columns)
    paired = runs[["activity_id", "date", "name", "distance_km", "pace_min_per_km",
                   "avg_hr"]].copy()
    paired["day"] = paired["date"].dt.normalize()
    scores = wellness[WELLNESS_COLUMNS].rename(columns={"date": "day"})
    paired = paired.merge(scores, on="day", how="inner").drop(columns="day")
    speed_m_per_min = 1000.0 / paired["pace_min_per_km"]
    paired["efficiency"] = speed_m_per_min / paired["avg_hr"].where(paired["avg_hr"] > 0)
    paired = paired.dropna(subset=list(WELLNESS_METRICS), how="all")
    return paired.sort_values("date").reset_index(drop=True)[columns]


def in_group(joined: pd.DataFrame, group: str) -> pd.DataFrame:
    """The runs whose distance falls in one of :data:`RUN_GROUPS`."""
    low, high = RUN_GROUPS[group]
    distance = joined["distance_km"]
    return joined[(distance >= low) & (distance < high)].reset_index(drop=True)


# --------------------------------------------------------------------------- statistics


def correlate(x: pd.Series, y: pd.Series, min_pairs: int = MIN_PAIRS) -> dict:
    """Pearson ``r`` with its 95% Fisher interval: ``{r, n, low, high}`` (NaN when too few)."""
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    result = {"r": np.nan, "n": n, "low": np.nan, "high": np.nan}
    if n < min_pairs or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
        return result
    r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    result["r"] = r
    if n > 3 and abs(r) < 1.0:
        z, half = np.arctanh(r), 1.96 / np.sqrt(n - 3)
        result["low"], result["high"] = float(np.tanh(z - half)), float(np.tanh(z + half))
    return result


def correlation_table(joined: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """Every wellness x run-metric pair: ``wellness, metric, *_label, r, n, low, high, strength``."""
    rows = []
    for wellness, wellness_label in WELLNESS_METRICS.items():
        for metric, metric_label in RUN_METRICS.items():
            stats = correlate(joined[wellness], joined[metric], min_pairs)
            rows.append({"wellness": wellness, "metric": metric,
                         "wellness_label": wellness_label, "metric_label": metric_label,
                         **stats, "strength": strength_label(stats["r"])})
    return pd.DataFrame(rows)


def reading(row: pd.Series | dict) -> str:
    """One sentence on what a correlation says, in the metric's own direction."""
    r = row["r"]
    if not np.isfinite(r):
        return f"Not enough runs with a {row['wellness_label'].lower()} to say."
    if abs(r) < 0.1:
        return f"Your {row['metric_label'].lower()} barely moves with your " \
               f"{row['wellness_label'].lower()}."
    better = {"pace_min_per_km": ("faster", "slower"),
              "avg_hr": ("at a higher heart rate", "at a lower heart rate"),
              "efficiency": ("more efficiently (more metres per heartbeat)",
                             "less efficiently (fewer metres per heartbeat)")}[row["metric"]]
    rising = r > 0
    if row["metric"] == "pace_min_per_km":
        rising = not rising  # a larger pace number is slower
    sentence = (f"After a higher {row['wellness_label'].lower()} you tended to run "
                f"{better[0] if rising else better[1]}")
    if np.isfinite(row["low"]) and row["low"] <= 0 <= row["high"]:
        sentence += ", but the interval includes zero, so this could be chance"
    return sentence + "."


def thirds(joined: pd.DataFrame, wellness: str, metric: str) -> pd.DataFrame:
    """Runs split by the morning's score into thirds, with each third's mean and 95% CI.

    Returns ``band, score_low, score_high, mean, low, high, n``; empty when
    there are too few runs to make three groups of at least 3.
    """
    pair = joined[[wellness, metric]].dropna()
    columns = ["band", "score_low", "score_high", "mean", "low", "high", "n"]
    if len(pair) < 9 or pair[wellness].nunique() < 3:
        return pd.DataFrame(columns=columns)
    labels = ["Lowest third", "Middle third", "Highest third"]
    pair["band"] = pd.qcut(pair[wellness].rank(method="first"), 3, labels=labels)
    rows = []
    for band, group in pair.groupby("band", observed=True):
        values = group[metric]
        mean, n = float(values.mean()), len(values)
        half = 1.96 * values.std() / np.sqrt(n) if n > 1 else np.nan
        rows.append((band, group[wellness].min(), group[wellness].max(), mean,
                     mean - half, mean + half, n))
    return pd.DataFrame(rows, columns=columns)


def baseline(wellness: pd.DataFrame, column: str, days: int = 7) -> pd.Series:
    """Trailing ``days``-day mean of a score over calendar time."""
    series = wellness.set_index("date")[column]
    return series.rolling(f"{days}D", min_periods=3).mean().reset_index(drop=True)


# --------------------------------------------------------------------------- demo data


def synthetic_wellness(runs: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Plausible nightly scores for the synthetic runner, so the tab has data on first launch.

    On run days the scores lean on how good the run was relative to similar
    runs (efficiency against the run's distance group), so a modest positive
    link exists to be found; noise dominates, as in real wearable data, and
    ~8% of mornings are missing.  Only ever paired with the synthetic runs.
    """
    rng = np.random.default_rng(seed)
    if runs.empty:
        return pd.DataFrame(columns=WELLNESS_COLUMNS)
    days = pd.date_range(runs["date"].min().normalize(), runs["date"].max().normalize(),
                         freq="D", name="date")
    n = len(days)

    scored = runs[["date", "distance_km", "pace_min_per_km", "avg_hr"]].copy()
    scored["efficiency"] = (1000.0 / scored["pace_min_per_km"]) / scored["avg_hr"]
    group = pd.cut(scored["distance_km"], [0, 8, 15, np.inf], right=False)
    # Efficiency against similar runs, detrended so fitness gains don't drive the link.
    trend = scored["efficiency"].rolling(30, min_periods=5, center=True).median()
    relative = scored["efficiency"] / trend
    z = relative.groupby(group, observed=True).transform(lambda s: (s - s.mean()) / s.std())
    by_day = z.groupby(scored["date"].dt.normalize()).mean().reindex(days)
    form = by_day.fillna(0.0).to_numpy()

    readiness = 70.0 + 4.5 * form + rng.normal(0.0, 9.0, n)
    sleep = 76.0 + 2.5 * form + 0.35 * (readiness - 70.0) + rng.normal(0.0, 8.0, n)
    frame = pd.DataFrame({
        "date": days,
        "sleep_score": np.clip(np.round(sleep), 25, 100),
        "readiness": np.clip(np.round(readiness), 15, 100),
    })
    missing = rng.random(n) < 0.08
    frame.loc[missing, list(WELLNESS_METRICS)] = np.nan
    return frame.dropna(how="all", subset=list(WELLNESS_METRICS)).reset_index(drop=True)
