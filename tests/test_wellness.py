"""Tests for the Apple Health sleep/readiness import and its pairing with runs."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import wellness
from src.wellness import (
    WellnessError, correlate, correlation_table, in_group, join_runs, load_wellness,
    nightly_sleep, readiness_score, reading, sleep_score, synthetic_wellness, thirds,
)

HEADER = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE HealthData [\n'
          '<!ELEMENT HealthData (ExportDate,Me,(Record|Workout)*)>\n]>\n'
          '<HealthData locale="en_US">\n <ExportDate value="2026-03-20 08:00:00 -0700"/>\n')


def sleep_record(value: str, start: str, end: str, source: str = "Watch") -> str:
    return (f' <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="{source}" '
            f'creationDate="{end} -0700" startDate="{start} -0700" endDate="{end} -0700" '
            f'value="HKCategoryValueSleepAnalysis{value}"/>\n')


def quantity(kind: str, stamp: str, value: float) -> str:
    return (f' <Record type="HKQuantityTypeIdentifier{kind}" sourceName="Watch" unit="ms" '
            f'creationDate="{stamp} -0700" startDate="{stamp} -0700" endDate="{stamp} -0700" '
            f'value="{value}">\n  <MetadataEntry key="HKAlgorithmVersion" value="2"/>\n'
            f' </Record>\n')


def export_xml(records: list[str]) -> bytes:
    return (HEADER + "".join(records) + "</HealthData>\n").encode()


def night(day: pd.Timestamp, onset_h: float = 23.0, hours: float = 8.0,
          source: str = "Watch") -> list[str]:
    """An unbroken night of Core sleep ending on the morning of ``day``."""
    start = day - pd.Timedelta(days=1) + pd.Timedelta(hours=onset_h)
    end = start + pd.Timedelta(hours=hours)
    return [sleep_record("AsleepCore", f"{start:%Y-%m-%d %H:%M:%S}",
                         f"{end:%Y-%m-%d %H:%M:%S}", source)]


def test_a_full_night_on_a_regular_bedtime_scores_100():
    days = pd.date_range("2026-03-01", periods=10)
    records = [r for day in days for r in night(day)]
    scores = load_wellness(export_xml(records), "export.xml")
    assert len(scores) == 10
    assert scores["sleep_score"].iloc[-1] == pytest.approx(100.0)
    assert scores["asleep_h"].iloc[-1] == pytest.approx(8.0)
    assert not scores["sleep_score_native"].any()


def test_short_broken_late_night_scores_lower():
    days = pd.date_range("2026-03-01", periods=8)
    records = [r for day in days[:-1] for r in night(day)]
    # 5 h asleep in two halves with 40 min awake between, starting 2 h late (01:00).
    records += [sleep_record("AsleepCore", "2026-03-08 01:00:00", "2026-03-08 03:30:00"),
                sleep_record("Awake", "2026-03-08 03:30:00", "2026-03-08 04:10:00"),
                sleep_record("AsleepDeep", "2026-03-08 04:10:00", "2026-03-08 06:40:00")]
    scores = load_wellness(export_xml(records), "export.xml")
    final = scores.iloc[-1]
    assert final["asleep_h"] == pytest.approx(5.0)
    # duration 50*(1/4)=12.5, bedtime 2 h late = 0, interruptions 20*(1-40/60)=6.67
    assert final["sleep_score"] == pytest.approx(19.2, abs=0.1)


def test_naps_and_a_second_source_are_not_double_counted():
    day = pd.Timestamp("2026-03-05")
    records = night(day) + night(day, hours=7.5, source="iPhone")
    records.append(sleep_record("AsleepUnspecified", "2026-03-05 13:00:00",
                                "2026-03-05 13:40:00"))
    nights = nightly_sleep(wellness.scan_records(io.BytesIO(export_xml(records)))["sleep"])
    assert len(nights) == 1
    assert nights.loc[0, "asleep_h"] == pytest.approx(8.0)


def test_evening_segments_belong_to_the_next_morning():
    records = [sleep_record("AsleepCore", "2026-03-01 22:00:00", "2026-03-01 23:59:00"),
               sleep_record("AsleepCore", "2026-03-02 00:00:00", "2026-03-02 06:00:00")]
    nights = nightly_sleep(wellness.scan_records(io.BytesIO(export_xml(records)))["sleep"])
    assert list(nights["date"]) == [pd.Timestamp("2026-03-02")]
    assert nights.loc[0, "awake_min"] == pytest.approx(1.0)


def test_readiness_is_70_on_a_typical_morning_and_higher_on_a_good_one():
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-01-01", periods=40)
    hrv = np.exp(np.log(50) + rng.normal(0, 0.1, 40))
    rhr = 50 + rng.normal(0, 2, 40)
    hrv[-1], rhr[-1] = 50 * np.exp(0.2), 46  # +2 sd HRV, -2 sd resting HR
    daily = pd.DataFrame({"date": days, "hrv_ms": hrv, "resting_hr": rhr})
    score = readiness_score(daily)
    assert score.iloc[:wellness.BASELINE_MIN].isna().all()  # no baseline yet
    assert score.iloc[-1] > 85
    assert 40 < score.iloc[wellness.BASELINE_MIN:-1].mean() < 100


def test_overnight_hrv_and_resting_hr_produce_readiness():
    days = pd.date_range("2026-01-01", periods=30)
    records = []
    for i, day in enumerate(days):
        records.append(quantity("HeartRateVariabilitySDNN", f"{day:%Y-%m-%d} 03:00:00",
                                50 + (i % 5)))
        records.append(quantity("HeartRateVariabilitySDNN", f"{day:%Y-%m-%d} 14:00:00", 999))
        records.append(quantity("RestingHeartRate", f"{day:%Y-%m-%d} 09:00:00", 52 - (i % 3)))
    scores = load_wellness(export_xml(records), "export.xml")
    assert scores["readiness"].notna().sum() == 30 - wellness.BASELINE_MIN
    assert scores["hrv_ms"].max() <= 54  # the 14:00 reading is daytime, not overnight


def test_apple_sleep_score_is_used_where_present():
    days = pd.date_range("2026-03-01", periods=3)
    records = [r for day in days for r in night(day)]
    records.append(' <Record type="HKQuantityTypeIdentifierAppleSleepScore" sourceName="Watch" '
                   'startDate="2026-03-02 06:59:00 -0700" endDate="2026-03-02 07:00:00 -0700" '
                   'value="64"/>\n')
    scores = load_wellness(export_xml(records), "export.xml").set_index("date")
    assert scores.loc["2026-03-02", "sleep_score"] == 64
    assert scores.loc["2026-03-02", "sleep_score_native"]
    assert not scores.loc["2026-03-03", "sleep_score_native"]


def test_apple_export_zip_is_read_in_place():
    records = [r for day in pd.date_range("2026-03-01", periods=3) for r in night(day)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("apple_health_export/export_cda.xml", "<x/>")
        zf.writestr("apple_health_export/export.xml", export_xml(records))
    buffer.seek(0)
    scores = load_wellness(buffer, "export.zip")
    assert len(scores) == 3
    assert not buffer.closed  # the caller's upload is left open


def test_csv_with_vendor_column_names():
    csv = (b"Cycle start time,Sleep Score,Recovery %\n"
           b"2026-03-01 22:40:00,81,66%\n2026-03-03,77,140\n")
    scores = load_wellness(csv, "scores.csv")
    assert list(scores["date"]) == [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")]
    assert scores.loc[0, "readiness"] == 66
    assert np.isnan(scores.loc[1, "readiness"])  # out of range


@pytest.mark.parametrize("content,name", [(b"a,b\n1,2\n", "x.csv"), (b"hello", "x.txt"),
                                          (export_xml([]), "export.xml")])
def test_unusable_uploads_say_why(content, name):
    with pytest.raises(WellnessError):
        load_wellness(content, name)


def runs_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "activity_id": [1, 2, 3],
        "date": pd.to_datetime(["2026-03-01 07:00", "2026-03-02 18:30", "2026-03-04 07:00"]),
        "name": ["a", "b", "c"], "distance_km": [5.0, 10.0, 20.0],
        "pace_min_per_km": [5.0, 5.5, 6.0], "avg_hr": [150.0, np.nan, 140.0],
    })


def test_runs_take_that_mornings_scores():
    scores = pd.DataFrame({"date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
                           "sleep_score": [80.0, 70.0, 60.0], "readiness": [75.0, np.nan, 50.0]})
    joined = join_runs(scores, runs_frame())
    assert list(joined["activity_id"]) == [1, 2]
    assert joined.loc[0, "efficiency"] == pytest.approx(200.0 / 150.0)
    assert np.isnan(joined.loc[1, "efficiency"])
    assert list(in_group(joined, "Medium")["activity_id"]) == [2]


def test_correlation_has_a_fisher_interval_and_needs_enough_runs():
    x = pd.Series(np.arange(30, dtype=float))
    y = x * 0.5 + np.random.default_rng(1).normal(0, 3, 30)
    stats = correlate(x, y)
    assert stats["low"] < stats["r"] < stats["high"]
    assert stats["n"] == 30
    assert np.isnan(correlate(x.head(5), y.head(5))["r"])


def test_reading_flips_direction_for_pace():
    row = {"r": -0.4, "low": -0.6, "high": -0.2, "wellness_label": "Sleep score",
           "metric_label": "Pace", "metric": "pace_min_per_km"}
    assert "faster" in reading(row)
    assert "chance" in reading({**row, "low": -0.7, "high": 0.05})


def test_synthetic_scores_show_a_link_on_the_sample_runs():
    from src.loader import SAMPLE_CSV, load_activities
    runs = load_activities(SAMPLE_CSV)
    joined = join_runs(synthetic_wellness(runs), runs)
    assert len(joined) > 0.8 * len(runs)
    table = correlation_table(joined).set_index(["wellness", "metric"])
    assert table.loc[("readiness", "efficiency"), "r"] > 0.2
    summary = thirds(joined, "readiness", "efficiency")
    both = joined[["readiness", "efficiency"]].dropna()
    assert summary["n"].sum() == len(both)
    assert summary["n"].max() - summary["n"].min() <= 1  # equal thirds
    assert summary["score_low"].is_monotonic_increasing
