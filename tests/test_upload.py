"""Uploaded exports: what gets unpacked, what is refused, and workspace cleanup."""

from __future__ import annotations

import io
import os
import time
import zipfile

import pandas as pd
import pytest

from src import streams, upload
from src.upload import ExportError

CSV = b"Activity ID,Activity Date,Activity Type,Distance\n1,\"Jan 1, 2025, 7:00:00 AM\",Run,5\n"


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_zip_keeps_only_the_csv_and_gps_folder(tmp_path):
    archive = _zip({
        "activities.csv": CSV,
        "activities/1.gpx.gz": b"gps",
        "media/photo.jpg": b"photo",
        "messages.csv": b"private",
    })
    csv_path = upload.extract_export(archive, "export_123.zip", tmp_path)
    assert csv_path == tmp_path / "export" / "activities.csv"
    assert csv_path.read_bytes() == CSV
    assert (tmp_path / "export" / "activities" / "1.gpx.gz").read_bytes() == b"gps"
    kept = {p.relative_to(tmp_path / "export").as_posix()
            for p in (tmp_path / "export").rglob("*") if p.is_file()}
    assert kept == {"activities.csv", "activities/1.gpx.gz"}


def test_zip_with_a_top_level_folder_is_flattened(tmp_path):
    archive = _zip({"export_123/activities.csv": CSV,
                    "export_123/activities/9.fit.gz": b"fit",
                    "__MACOSX/export_123/activities.csv": b"junk"})
    csv_path = upload.extract_export(archive, "export.ZIP", tmp_path)
    assert csv_path.read_bytes() == CSV
    assert (csv_path.parent / "activities" / "9.fit.gz").is_file()


def test_a_bare_csv_is_saved_as_is(tmp_path):
    csv_path = upload.extract_export(CSV, "activities.csv", tmp_path)
    assert csv_path.read_bytes() == CSV


def test_a_zip_without_activities_csv_is_refused(tmp_path):
    with pytest.raises(ExportError, match="No `activities.csv`"):
        upload.extract_export(_zip({"profile.csv": b"x"}), "export.zip", tmp_path)


def test_a_broken_zip_is_refused(tmp_path):
    with pytest.raises(ExportError, match="not a valid zip"):
        upload.extract_export(b"not a zip", "export.zip", tmp_path)


def test_paths_that_climb_out_are_never_written(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    archive = _zip({"activities.csv": CSV, "activities/../../escape.txt": b"x",
                    "/abs/activities.csv": b"y"})
    upload.extract_export(archive, "export.zip", workspace)
    assert not (tmp_path / "escape.txt").exists()
    assert not any(tmp_path.rglob("escape.txt"))


def test_oversized_activity_data_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(upload, "MAX_EXTRACTED_BYTES", 100)
    archive = _zip({"activities.csv": CSV, "activities/1.gpx": b"0" * 1000})
    with pytest.raises(ExportError, match="too large"):
        upload.extract_export(archive, "export.zip", tmp_path)


def test_read_export_csv_from_zip_or_csv():
    assert upload.read_export_csv(_zip({"x/activities.csv": CSV}), "a.zip") == CSV
    assert upload.read_export_csv(CSV, "activities.csv") == CSV


def test_workspaces_are_private_and_purged_when_stale(tmp_path):
    fresh = upload.new_workspace(tmp_path)
    stale = upload.new_workspace(tmp_path)
    assert fresh != stale
    assert (fresh.stat().st_mode & 0o777) == 0o700
    old = time.time() - upload.MAX_WORKSPACE_AGE_S - 60
    os.utime(stale, (old, old))
    assert upload.purge_old_workspaces(tmp_path) == 1
    assert fresh.is_dir() and not stale.exists()


def test_remove_workspace_refuses_outside_paths(tmp_path):
    workspace = upload.new_workspace(tmp_path)
    with pytest.raises(ValueError):
        upload.remove_workspace(tmp_path, tmp_path)
    upload.remove_workspace(workspace, tmp_path)
    assert not workspace.exists()


def test_stream_files_ignore_names_outside_the_export(tmp_path):
    activities = pd.DataFrame({"activity_id": [1, 2, 3],
                               "filename": ["activities/1.gpx", "../../etc/passwd", "/etc/hosts"]})
    found = streams.stream_files(activities, tmp_path / "activities.csv")
    assert found == [(1, tmp_path / "activities" / "1.gpx")]
