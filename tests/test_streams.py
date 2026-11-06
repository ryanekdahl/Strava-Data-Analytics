"""Tests for GPS-file parsing, in-run best efforts, splits and the stream cache."""

from __future__ import annotations

import gzip
import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import streams
from src.generate_sample import generate_sample
from src.loader import load_activities

START = datetime(2024, 1, 1, 7, 0, tzinfo=timezone.utc)


def make_stream(t, d, hr=150.0, altitude=0.0) -> pd.DataFrame:
    t = np.asarray(t, dtype="float64")
    return pd.DataFrame({
        "t_s": t, "distance_m": np.asarray(d, dtype="float64"),
        "hr": np.broadcast_to(np.asarray(hr, dtype="float64"), t.shape).copy(),
        "cadence_spm": 170.0, "altitude_m": altitude,
    })


def steady(speed=4.0, seconds=2500, step=5.0) -> pd.DataFrame:
    t = np.arange(0, seconds + step, step)
    return make_stream(t, t * speed)


# --------------------------------------------------------------------------
# best efforts
# --------------------------------------------------------------------------

def test_best_times_on_a_steady_run_are_exact():
    times = streams.best_times(steady(), [1000, 5000, 10000, 10001])
    np.testing.assert_allclose(times[:3], [250.0, 1250.0, 2500.0])
    assert np.isnan(times[3])  # longer than the run


def test_best_effort_finds_a_fast_stretch_inside_a_slower_run():
    # 3 km at 3 m/s, then 1 km at 5 m/s, then 3 km at 3 m/s.
    t = [0, 1000, 1200, 2200]
    d = [0, 3000, 4000, 7000]
    times = streams.best_times(make_stream(t, d), [1000, 2000])
    assert times[0] == pytest.approx(200.0)
    # Best 2 km: the fast km plus 1 km at 3 m/s on either side.
    assert times[1] == pytest.approx(200.0 + 1000.0 / 3.0)


def test_window_starting_between_samples_is_found():
    # Sparse samples: the fastest 1 km starts half-way between two of them.
    t = [0, 100, 200, 300, 400]
    d = [0, 300, 900, 1500, 1800]  # 3, 6, 6, 3 m/s
    times = streams.best_times(make_stream(t, d), [1000])
    # 6 m/s from 300 m to 1500 m covers 1200 m, so 1 km fits inside at 6 m/s.
    assert times[0] == pytest.approx(1000.0 / 6.0)


def test_a_stop_is_not_counted_as_running_time():
    run = steady()
    stopped = run.copy()
    stopped.loc[stopped["t_s"] >= 1250, "t_s"] += 120  # two minutes at a crossing
    np.testing.assert_allclose(streams.best_times(stopped, [5000, 6000]),
                               streams.best_times(run, [5000, 6000]), atol=1e-6)


def test_gps_glitch_cannot_create_a_world_record():
    run = steady()
    glitch = run.copy()
    glitch.loc[glitch.index >= 100, "distance_m"] += 2000  # a 2 km teleport in 5 s
    best_1k = streams.best_times(glitch, [1000])[0]
    assert best_1k >= 1000.0 / streams.MAX_SPEED_M_S - 1e-6
    assert best_1k > 150.0  # nowhere near the 0.5 s the raw jump implies


def test_moving_clock_credits_a_logged_pause_with_its_running_only():
    # Auto-pause: one 65 s gap during which only 15 m were covered.
    t = np.array([0, 5, 10, 75, 80])
    d = np.array([0, 15, 30, 45, 60])
    clock = streams.moving_clock(t, d)
    assert clock[-1] == pytest.approx(20.0)  # four 5-second steps of 15 m


# --------------------------------------------------------------------------
# histogram, splits, decoupling
# --------------------------------------------------------------------------

def test_hr_histogram_sums_to_moving_time():
    run = steady()
    run["hr"] = np.where(run["t_s"] < 1250, 140.0, 160.0)
    hist = streams.hr_histogram(run)
    assert set(hist) == {140, 160}
    assert sum(hist.values()) == pytest.approx(2500.0)
    assert hist[140] == pytest.approx(1250.0)


def test_km_splits_cover_the_run_and_keep_the_partial_last_split():
    t = np.arange(0, 1126, 5.0)
    run = make_stream(t, t * 4.0, altitude=t * 0.01)  # 4.5 km, climbing
    splits = streams.km_splits(run)
    assert list(splits["split"]) == [1, 2, 3, 4, 5]
    assert splits["distance_m"].sum() == pytest.approx(4500.0)
    assert splits["distance_m"].iloc[-1] == pytest.approx(500.0)
    np.testing.assert_allclose(splits["pace_min_per_km"], 1000 / 4.0 / 60.0)
    assert (splits["elevation_change_m"] > 0).all()


def test_grade_cost_shape():
    cost = streams.grade_cost([-0.5, -0.2, -0.1, 0.0, 0.1])
    np.testing.assert_allclose(cost, [streams.grade_cost(-0.3), 1.0, 0.88, 1.0, 1.36])


def test_gap_equals_pace_on_the_flat_and_beats_it_uphill():
    t = np.arange(0, 1001, 1.0)
    flat = streams.km_splits(make_stream(t, t * 4.0, altitude=0.0))
    np.testing.assert_allclose(flat["gap_pace_min_per_km"], flat["pace_min_per_km"])
    climb = streams.km_splits(make_stream(t, t * 4.0, altitude=t * 4.0 * 0.05))
    # Steady 5% climb: cost 1 + 0.12 + 0.03 = 1.15, so GAP is pace / 1.15.
    np.testing.assert_allclose(climb["gap_pace_min_per_km"].iloc[1:-1],
                               climb["pace_min_per_km"].iloc[1:-1] / 1.15, rtol=1e-3)


def test_gap_is_missing_without_altitude():
    t = np.arange(0, 1001, 1.0)
    run = make_stream(t, t * 4.0, altitude=np.nan)
    assert streams.km_splits(run)["gap_pace_min_per_km"].isna().all()
    assert streams.equivalent_distance(run) is None


def test_gap_speed_histogram_sums_to_moving_time_and_reads_climbs_as_faster():
    t = np.arange(0, 2001, 1.0)
    flat = streams.gap_speed_histogram(make_stream(t, t * 4.0))
    assert flat == {40: pytest.approx(2000.0)}
    climb = streams.gap_speed_histogram(make_stream(t, t * 4.0, altitude=t * 4.0 * 0.05))
    assert sum(climb.values()) == pytest.approx(2000.0)
    modal = max(climb, key=climb.get)
    assert modal * streams.SPEED_BIN_M_S == pytest.approx(4.6, abs=0.05)


def test_aerobic_decoupling_measures_heart_rate_drift():
    run = steady(seconds=3600)
    assert streams.aerobic_decoupling(run) == pytest.approx(0.0)
    run["hr"] = np.where(run["t_s"] <= 1800, 150.0, 165.0)  # same speed, 10% more HR
    assert streams.aerobic_decoupling(run) == pytest.approx(100.0 * (1 - 150 / 165), rel=0.02)
    assert np.isnan(streams.aerobic_decoupling(steady(seconds=600)))  # too short


# --------------------------------------------------------------------------
# file formats
# --------------------------------------------------------------------------

def _stamp(seconds: float) -> str:
    return (START + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_gpx_with_heart_rate_extension_is_read(tmp_path):
    # Due north at 0.0001 degrees of latitude (~11.1 m) every 5 s.
    points = "".join(
        f'<trkpt lat="{51.5 + i * 1e-4:.7f}" lon="-0.1"><ele>{10 + i}</ele>'
        f"<time>{_stamp(5 * i)}</time><extensions><gpxtpx:TrackPointExtension>"
        f"<gpxtpx:hr>{140 + i}</gpxtpx:hr><gpxtpx:cad>85</gpxtpx:cad>"
        f"</gpxtpx:TrackPointExtension></extensions></trkpt>"
        for i in range(11)
    )
    text = ('<gpx xmlns="http://www.topografix.com/GPX/1/1" '
            'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
            f"<trk><trkseg>{points}</trkseg></trk></gpx>")
    path = tmp_path / "run.gpx.gz"
    path.write_bytes(gzip.compress(text.encode()))
    stream = streams.read_stream(path)

    assert list(stream.columns) == streams.STREAM_COLUMNS
    assert stream["t_s"].iloc[-1] == 50.0
    assert stream["distance_m"].iloc[-1] == pytest.approx(10 * 11.12, rel=0.01)
    assert stream["hr"].iloc[0] == 140 and stream["hr"].iloc[-1] == 150
    assert stream["cadence_spm"].iloc[0] == 170  # per-leg 85 doubled
    assert stream["altitude_m"].iloc[-1] == 20
    assert stream["lat"].iloc[-1] == pytest.approx(51.501)
    assert (stream["lon"] == -0.1).all()


def test_tcx_is_read(tmp_path):
    points = "".join(
        f"<Trackpoint><Time>{_stamp(10 * i)}</Time><DistanceMeters>{30 * i}</DistanceMeters>"
        f"<HeartRateBpm><Value>{150}</Value></HeartRateBpm></Trackpoint>"
        for i in range(6)
    )
    path = tmp_path / "run.tcx"
    path.write_text('<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/'
                    'TrainingCenterDatabase/v2"><Activities><Activity><Lap><Track>'
                    f"{points}</Track></Lap></Activity></Activities></TrainingCenterDatabase>")
    stream = streams.read_stream(path)
    assert stream["distance_m"].iloc[-1] == 150.0
    assert stream["t_s"].iloc[-1] == 50.0
    assert (stream["hr"] == 150).all()
    assert stream["lat"].isna().all()  # no <Position>: a treadmill-style recording


def test_tcx_positions_are_read(tmp_path):
    points = "".join(
        f"<Trackpoint><Time>{_stamp(10 * i)}</Time><Position><LatitudeDegrees>{40 + i / 1e4}"
        f"</LatitudeDegrees><LongitudeDegrees>-74</LongitudeDegrees></Position>"
        f"<DistanceMeters>{30 * i}</DistanceMeters></Trackpoint>"
        for i in range(6)
    )
    path = tmp_path / "run.tcx"
    path.write_text('<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/'
                    'TrainingCenterDatabase/v2"><Activities><Activity><Lap><Track>'
                    f"{points}</Track></Lap></Activity></Activities></TrainingCenterDatabase>")
    stream = streams.read_stream(path)
    assert stream["lat"].iloc[-1] == pytest.approx(40.0005)
    assert (stream["lon"] == -74).all()


_FIT_CRC_TABLE = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
                  0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]
_FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)


def _fit_crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        for nibble in (byte & 0xF, byte >> 4):
            tmp = _FIT_CRC_TABLE[crc & 0xF]
            crc = ((crc >> 4) & 0x0FFF) ^ tmp ^ _FIT_CRC_TABLE[nibble]
    return crc


def make_fit(samples: list[tuple[float, float, int]]) -> bytes:
    """A minimal FIT file of ``record`` messages: (seconds, meters, bpm) per sample."""
    # Definition for local type 0 -> global message 20 (record): timestamp,
    # distance (uint32, centimetres) and heart_rate (uint8).
    body = bytes([0x40, 0, 0]) + struct.pack("<HB", 20, 3) + bytes([253, 4, 0x86, 5, 4, 0x86,
                                                                     3, 1, 0x02])
    start = int((START - _FIT_EPOCH).total_seconds())
    for seconds, meters, bpm in samples:
        body += bytes([0x00]) + struct.pack("<IIB", start + int(seconds), int(meters * 100), bpm)
    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", _fit_crc(header))
    return header + body + struct.pack("<H", _fit_crc(header + body))


def test_fit_file_is_read(tmp_path):
    pytest.importorskip("fitdecode")
    path = tmp_path / "run.fit.gz"
    path.write_bytes(gzip.compress(make_fit([(5 * i, 15.0 * i, 140 + i) for i in range(21)])))
    stream = streams.read_stream(path)
    assert len(stream) == 21
    assert stream["t_s"].iloc[-1] == 100.0
    assert stream["distance_m"].iloc[-1] == pytest.approx(300.0)
    assert stream["hr"].iloc[-1] == 160
    assert streams.best_times(stream, [150.0])[0] == pytest.approx(50.0)


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "run.kml"
    path.write_text("<kml/>")
    with pytest.raises(ValueError, match="unsupported"):
        streams.read_stream(path)


# --------------------------------------------------------------------------
# the cache, end to end on the synthetic export
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "activities.csv"
    generate_sample(path, seed=99)
    return path, load_activities(path)


def test_sample_recordings_parse_and_agree_with_the_csv(sample):
    path, runs = sample
    assert streams.build_cache(runs, path) == len(runs)
    data = streams.load_stream_data(runs, path)
    assert data.status == {"parsed": len(runs), "failed": 0, "missing": 0}

    merged = data.runs.merge(runs, on="activity_id")
    assert (merged["distance_m"] - merged["distance_km"] * 1000).abs().max() < 20.0
    assert (merged["moving_s"] - merged["moving_time_s"]).abs().max() < 60.0


def test_sample_recordings_carry_grade_adjusted_data(sample):
    path, runs = sample
    data = streams.load_stream_data(runs, path)
    assert data.splits["gap_pace_min_per_km"].notna().all()
    assert data.runs["gap_distance_m"].notna().all()
    per_run = data.gap_seconds.groupby("activity_id")["seconds"].sum()
    moving = data.runs.set_index("activity_id")["moving_s"]
    assert ((per_run - moving.loc[per_run.index]).abs() < 30.0).all()


def test_in_run_bests_are_never_slower_than_whole_run_bests(sample):
    path, runs = sample
    data = streams.load_stream_data(runs, path)
    best = streams.best_effort_table(data.best, runs).set_index("distance")
    for label, km in (("5K", 5.0), ("10K", 10.0)):
        whole_run = (runs.loc[runs["distance_km"] >= km, "pace_min_per_km"].min() * km * 60.0)
        assert best.loc[label, "time_s"] <= whole_run + 1.0


def test_cache_is_reused_then_refreshed_when_a_file_changes(sample):
    path, runs = sample
    streams.build_cache(runs, path)
    assert streams.pending_files(runs, path) == []
    first = path.parent / runs["filename"].iloc[0]
    first.write_bytes(first.read_bytes())  # same content, new modification time
    pending = streams.pending_files(runs, path)
    assert [file for _, file in pending] == [first]


def test_a_disk_error_is_retried_not_cached_as_a_bad_file(tmp_path, monkeypatch):
    source = tmp_path / "1.gpx"
    source.write_text("<gpx/>")

    def timeout(path):
        raise TimeoutError(60, "Operation timed out")

    monkeypatch.setattr(streams, "read_stream", timeout)
    assert streams.cache_summary(1, source, tmp_path / "cache") is False
    assert not list((tmp_path / "cache").glob("*.json")) if (tmp_path / "cache").exists() else True


def test_unreadable_and_missing_files_are_counted_not_fatal(tmp_path):
    csv = tmp_path / "activities.csv"
    csv.write_text("x\n")
    (tmp_path / "activities").mkdir()
    (tmp_path / "activities" / "1.gpx").write_text("this is not xml")
    runs = pd.DataFrame({"activity_id": [1, 2], "date": pd.to_datetime(["2024-01-01"] * 2),
                         "name": ["a", "b"], "filename": ["activities/1.gpx", "activities/2.gpx"]})
    streams.build_cache(runs, csv)
    data = streams.load_stream_data(runs, csv)
    assert data.status == {"parsed": 0, "failed": 1, "missing": 1}
    assert data.best.empty and data.splits.empty


def test_route_is_thinned_by_distance_so_a_stop_adds_no_points():
    # 1 km due north at 4 m/s, a 10-minute stop logged every second, then 1 km more.
    t = np.concatenate([np.arange(0, 251), 250 + np.arange(1, 601), 850 + np.arange(1, 251)])
    d = np.concatenate([np.arange(0, 251) * 4.0, np.full(600, 1000.0),
                        1000.0 + np.arange(1, 251) * 4.0])
    stream = make_stream(t, d).assign(lat=51.5 + d / 111_195.0, lon=-0.1)
    route = streams.route_track(stream, points=50)

    assert 45 <= len(route) <= 50
    assert route["distance_m"].iloc[-1] == pytest.approx(2000.0)
    # Evenly spaced in distance: no cluster of points where the runner stood still.
    assert np.diff(route["distance_m"]).max() < 2.5 * 2000.0 / 50
    assert route["moving_s"].iloc[-1] < 600.0  # the stop is not running time


def test_a_stream_without_positions_has_no_route():
    assert streams.route_track(steady()).empty
    assert streams.route_track(steady().assign(lat=np.nan, lon=np.nan)).empty


def test_mile_splits_cover_the_same_distance_as_km_splits(sample):
    path, runs = sample
    streams.build_cache(runs, path)  # an earlier test touches a file
    data = streams.load_stream_data(runs, path)
    km = data.splits.groupby("activity_id")["distance_m"].sum()
    miles = data.mile_splits.groupby("activity_id")["distance_m"].sum()
    assert (km - miles).abs().max() < 1.0
    # Every split but a run's last is exactly a mile (the last may absorb a sliver).
    last = data.mile_splits.groupby("activity_id")["split"].transform("max")
    inner = data.mile_splits[data.mile_splits["split"] < last]
    assert inner["distance_m"].to_numpy() == pytest.approx(streams.MILE_M)


def test_restrict_keeps_only_the_named_runs(sample):
    path, runs = sample
    streams.build_cache(runs, path)  # an earlier test touches a file
    data = streams.load_stream_data(runs, path)
    keep = runs["activity_id"].iloc[:3]
    restricted = streams.restrict(data, keep)
    for frame in (restricted.best, restricted.splits, restricted.mile_splits, restricted.runs):
        assert set(frame["activity_id"]) == set(keep)
    assert restricted.status == data.status


def test_route_for_reads_the_cached_track_with_a_plausible_pace(sample):
    path, runs = sample
    streams.build_cache(runs, path)
    run = runs.iloc[0]
    route = streams.route_for(int(run["activity_id"]), runs, path)
    assert len(route) > 100
    assert route[["lat", "lon"]].notna().all().all()
    # The median point pace sits near the run's average pace.
    assert route["pace_min_per_km"].median() == pytest.approx(run["pace_min_per_km"], rel=0.15)
    assert route["altitude_m"].notna().all() and route["grade"].abs().max() < 0.1
    assert route["gap_pace_min_per_km"].notna().all()
    assert streams.route_for(-1, runs, path).empty


def test_reparsing_a_run_removes_its_stale_cache_files(sample, tmp_path):
    path, runs = sample
    activity_id, source = streams.stream_files(runs, path)[0]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale = cache_dir / f"v1-{activity_id}-123-456.json"
    other = cache_dir / f"v1-{activity_id + 100000}-123-456.json"
    stale.write_text("{}")
    other.write_text("{}")
    assert streams.cache_summary(activity_id, source, cache_dir)
    assert not stale.exists() and other.exists()


def test_pace_curve_is_monotone_in_time(sample):
    path, runs = sample
    curve = streams.pace_curve(streams.load_stream_data(runs, path).best)
    assert curve["distance_km"].is_monotonic_increasing
    assert curve["time_s"].is_monotonic_increasing  # longer never takes less time
