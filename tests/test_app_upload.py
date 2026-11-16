"""The hosted flow: a visitor uploads their archive, then removes it.

Drives ``app.py`` through Streamlit's ``AppTest`` with a zip built from the
sample export (its CSV plus a few GPS files, to keep the run quick).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.generate_sample import generate_sample, sample_is_current  # noqa: E402
from src.loader import SAMPLE_CSV  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"
TIMEOUT_S = 600
GPS_FILES = 5


@pytest.fixture(scope="module")
def archive() -> bytes:
    if not sample_is_current() or not (SAMPLE_CSV.parent / "activities").is_dir():
        generate_sample(SAMPLE_CSV)
    runs = pd.read_csv(SAMPLE_CSV)
    with_gps = runs["Filename"].dropna().head(GPS_FILES)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("export_42/activities.csv", SAMPLE_CSV.read_bytes())
        for name in with_gps:
            zf.write(SAMPLE_CSV.parent / name, f"export_42/{name}")
        zf.writestr("export_42/media/photo.jpg", b"not needed")
    return buffer.getvalue()


def _uploader(at: AppTest, prefix: str):
    return next(u for u in at.file_uploader if (u.key or "").startswith(prefix))


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]


def test_upload_and_remove(archive):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    _assert_clean(at)
    assert any("demo data" in info.value for info in at.info)

    _uploader(at, "export_upload").set_value(("export_42.zip", archive, "application/zip"))
    at.run()
    _assert_clean(at)
    assert at.radio(key="data_source").value == "My uploaded export"
    assert not any("demo data" in info.value for info in at.info)
    workspace = Path(at.session_state["workspace"])
    export = workspace / "export"
    assert (export / "activities.csv").is_file()
    assert len(list((export / "activities").iterdir())) == GPS_FILES
    assert not (export / "media").exists()

    at.button(key="remove_upload").click()
    at.run()
    _assert_clean(at)
    assert not workspace.exists()
    assert at.radio(key="data_source").value == "Sample data (synthetic)"


def test_a_bad_upload_explains_itself():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    _uploader(at, "export_upload").set_value(("export.zip", b"not a zip", "application/zip"))
    at.run()
    _assert_clean(at)
    assert any("not a valid zip" in error.value for error in at.error)
    assert at.radio(key="data_source").value == "Sample data (synthetic)"
