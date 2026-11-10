"""Uploaded Strava exports: unpack them into a private, temporary per-visitor folder.

The analytics pipeline reads an export from a path (``activities.csv`` plus the
``activities/`` folder of GPS files beside it), so an upload is written to a
fresh folder under the system temp directory -- one per browser session, never
shared between visitors, and deleted when the visitor removes their data or
after :data:`MAX_WORKSPACE_AGE_S` at the latest.

A Strava archive also carries photos, routes, messages and more; only
``activities.csv`` and the ``activities/`` folder are ever extracted, so the
rest of the archive is never written to disk.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

WORKSPACES_ROOT = Path(tempfile.gettempdir()) / "strava-run-analytics"
MAX_WORKSPACE_AGE_S = 6 * 3600
EXPORT_CSV = "activities.csv"
GPS_FOLDER = "activities"
# Guards against zip bombs: an honest export's GPS files are small and already
# compressed, so these limits are far above any real archive's activity data.
MAX_EXTRACTED_BYTES = 3 * 1024**3
MAX_GPS_FILES = 50_000


class ExportError(ValueError):
    """The upload is not a usable Strava export; the message says why, for the visitor."""


def new_workspace(root: Path = WORKSPACES_ROOT) -> Path:
    """Create an empty, private folder for one visitor's data."""
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / uuid.uuid4().hex
    workspace.mkdir(mode=0o700)
    return workspace


def remove_workspace(workspace: Path, root: Path = WORKSPACES_ROOT) -> None:
    """Delete a workspace, refusing anything that is not directly inside ``root``."""
    workspace = Path(workspace)
    if workspace.resolve().parent != root.resolve():
        raise ValueError(f"{workspace} is not a workspace in {root}.")
    shutil.rmtree(workspace, ignore_errors=True)


def purge_old_workspaces(root: Path = WORKSPACES_ROOT, max_age_s: float = MAX_WORKSPACE_AGE_S,
                         now: float | None = None) -> int:
    """Delete workspaces untouched for ``max_age_s``; returns how many were removed.

    Covers visitors who closed the tab without removing their data.
    """
    if not root.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for workspace in root.iterdir():
        try:
            stale = workspace.is_dir() and now - workspace.stat().st_mtime > max_age_s
        except OSError:
            continue
        if stale:
            shutil.rmtree(workspace, ignore_errors=True)
            removed += 1
    return removed


def _member_path(name: str) -> PurePosixPath | None:
    """A zip member's path, or ``None`` if it is absolute or climbs out with ``..``."""
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return path


def _find_export_csv(names: list[str]) -> PurePosixPath:
    """The archive's ``activities.csv``: the shallowest one, to skip any nested copies."""
    candidates = [path for name in names
                  if (path := _member_path(name)) is not None and path.name == EXPORT_CSV
                  and not any(part.startswith("__MACOSX") for part in path.parts)]
    if not candidates:
        raise ExportError(
            "No `activities.csv` was found in that zip. Upload the archive Strava emailed "
            "you, unchanged, or just its `activities.csv`.")
    return min(candidates, key=lambda path: len(path.parts))


def extract_export(upload: BinaryIO | bytes, filename: str, workspace: Path) -> Path:
    """Write an uploaded export into ``workspace/export/`` and return its ``activities.csv``.

    ``upload`` is either a Strava archive (``.zip``) -- from which only
    ``activities.csv`` and the ``activities/`` GPS folder beside it are
    extracted -- or a bare ``activities.csv``.  Raises :class:`ExportError`
    with a visitor-facing message when the upload cannot be used.
    """
    data = io.BytesIO(upload) if isinstance(upload, bytes) else upload
    target = Path(workspace) / "export"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    csv_path = target / EXPORT_CSV

    if not filename.lower().endswith(".zip"):
        csv_path.write_bytes(data.read())
        return csv_path

    try:
        archive = zipfile.ZipFile(data)
    except zipfile.BadZipFile as err:
        raise ExportError("That file is not a valid zip archive.") from err
    with archive:
        members = archive.infolist()
        csv_member = _find_export_csv([member.filename for member in members])
        base = csv_member.parent
        gps_prefix = base / GPS_FOLDER
        wanted = []
        for member in members:
            path = _member_path(member.filename)
            if path is None or member.is_dir():
                continue
            if path == csv_member or (gps_prefix in path.parents):
                wanted.append((member, path.relative_to(base)))
        gps_count = len(wanted) - 1
        if gps_count > MAX_GPS_FILES:
            raise ExportError(f"That archive holds {gps_count:,} activity files, more than "
                              f"the {MAX_GPS_FILES:,} this app accepts.")
        if sum(member.file_size for member, _ in wanted) > MAX_EXTRACTED_BYTES:
            raise ExportError("The activity data in that archive is too large to unpack.")
        written = 0
        for member, relative in wanted:
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as sink:
                # Count real bytes rather than trusting the sizes the zip declares.
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_EXTRACTED_BYTES:
                        raise ExportError(
                            "The activity data in that archive is too large to unpack.")
                    sink.write(chunk)
    return csv_path


def read_export_csv(upload: BinaryIO | bytes, filename: str) -> bytes:
    """Just the ``activities.csv`` bytes from an uploaded zip or CSV (for a friend's export)."""
    data = io.BytesIO(upload) if isinstance(upload, bytes) else upload
    if not filename.lower().endswith(".zip"):
        return data.read()
    try:
        archive = zipfile.ZipFile(data)
    except zipfile.BadZipFile as err:
        raise ExportError("That file is not a valid zip archive.") from err
    with archive:
        csv_member = _find_export_csv(archive.namelist())
        name = next(n for n in archive.namelist() if _member_path(n) == csv_member)
        info = archive.getinfo(name)
        if info.file_size > MAX_EXTRACTED_BYTES:
            raise ExportError("The `activities.csv` in that archive is too large.")
        return archive.read(name)
