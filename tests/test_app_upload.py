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

from src.generate_sample import (  # noqa: E402
    friends_are_current, generate_friends, generate_sample, sample_is_current,
)
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


def test_compare_two_uploaded_exports():
    if not friends_are_current(SAMPLE_CSV):
        generate_friends(SAMPLE_CSV)
    friends = SAMPLE_CSV.parent / "friends"
    alex = io.BytesIO()
    with zipfile.ZipFile(alex, "w") as zf:
        zf.writestr("export_7/activities.csv", (friends / "alex.csv").read_bytes())
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert any("Add an export on each side" in info.value for info in at.info)

    _uploader(at, "compare_upload_a").set_value(("export_7.zip", alex.getvalue(),
                                                 "application/zip"))
    at.text_input(key="compare_name_a").set_value("Alex")
    _uploader(at, "compare_upload_b").set_value(("sam.csv", (friends / "sam.csv").read_bytes(),
                                                 "text/csv"))
    at.text_input(key="compare_name_b").set_value("Sam")
    at.run()
    _assert_clean(at)
    assert at.selectbox(key="compare_window").value == "Dates both cover"
    assert {m.label for m in at.metric} >= {"Alex leads", "Sam leads"}
    assert any("Runs together" in h.value for h in at.subheader)

    at.selectbox(key="compare_window").set_value("Last 3 months").run()
    _assert_clean(at)
    at.segmented_control(key="compare_measure").set_value("elevation_m").run()
    _assert_clean(at)
    at.radio(key="units").set_value("Miles").run()
    _assert_clean(at)

    stored = Path(at.session_state["workspace"]) / "compare"
    assert (stored / "a.csv").is_file() and (stored / "b.csv").is_file()
    at.button(key="compare_clear").click()
    at.run()
    _assert_clean(at)
    assert not (stored / "a.csv").exists() and not (stored / "b.csv").exists()
    assert any("Add an export on each side" in info.value for info in at.info)


def test_a_bad_comparison_upload_explains_itself():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    _uploader(at, "compare_upload_a").set_value(("x.zip", b"not a zip", "application/zip"))
    at.run()
    _assert_clean(at)
    assert any("not a valid zip" in error.value for error in at.error)


def test_sleep_scores_upload_and_remove():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    runs = pd.read_csv(SAMPLE_CSV)
    days = pd.to_datetime(runs["Activity Date"], format="mixed").dt.normalize().unique()
    rows = "".join(f"{day:%Y-%m-%d},{70 + i % 20},{60 + i % 30}\n"
                   for i, day in enumerate(days))
    csv = ("date,Sleep Score,Readiness\n" + rows).encode()
    _uploader(at, "wellness_file").set_value(("scores.csv", csv, "text/csv"))
    at.run()
    _assert_clean(at)
    assert not any("Demo scores" in info.value for info in at.info)
    assert any("scores.csv" in caption.value for caption in at.caption)
    assert at.session_state["wellness"][1] is not None

    at.button(key="wellness_remove").click()
    at.run()
    _assert_clean(at)
    assert any("Demo scores" in info.value for info in at.info)


def test_a_bad_sleep_upload_explains_itself():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    _uploader(at, "wellness_file").set_value(("scores.csv", b"a,b\n1,2\n", "text/csv"))
    at.run()
    _assert_clean(at)
    assert any("could not be read" in error.value for error in at.error)
