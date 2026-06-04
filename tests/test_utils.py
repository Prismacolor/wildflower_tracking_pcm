"""
test_utils.py
Tests for scripts/utils.py helper functions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.utils import (
    collect_images,
    datestamp,
    ensure_dir,
    latest_subdirectory,
    load_species_tags,
    lookup_status,
    timestamp,
    two_most_recent_files,
)


# ---------------------------------------------------------------------------
# datestamp / timestamp
# ---------------------------------------------------------------------------

def test_datestamp_format():
    ds = datestamp()
    assert len(ds) == 8
    assert ds.isdigit()


def test_timestamp_format():
    ts = timestamp()
    assert len(ts) == 15         # YYYYMMDD_HHMMSS
    assert "_" in ts


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

def test_ensure_dir_creates(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.is_dir()
    assert result == target


def test_ensure_dir_idempotent(tmp_path):
    ensure_dir(tmp_path)   # already exists — should not raise
    assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# latest_subdirectory
# ---------------------------------------------------------------------------

def test_latest_subdirectory(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    # Touch new to make it newer
    import time; time.sleep(0.01)
    new.touch()
    assert latest_subdirectory(tmp_path) == new


def test_latest_subdirectory_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_subdirectory(tmp_path)


# ---------------------------------------------------------------------------
# two_most_recent_files
# ---------------------------------------------------------------------------

def test_two_most_recent_files(tmp_path):
    import time
    f1 = tmp_path / "results_a.csv"; f1.write_text("x")
    time.sleep(0.01)
    f2 = tmp_path / "results_b.csv"; f2.write_text("x")
    time.sleep(0.01)
    f3 = tmp_path / "results_c.csv"; f3.write_text("x")

    newer, older = two_most_recent_files(tmp_path, "*.csv")
    assert newer == f3
    assert older == f2


def test_two_most_recent_files_not_enough(tmp_path):
    (tmp_path / "only.csv").write_text("x")
    with pytest.raises(ValueError):
        two_most_recent_files(tmp_path, "*.csv")


# ---------------------------------------------------------------------------
# load_species_tags / lookup_status
# ---------------------------------------------------------------------------

def _write_tags(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["species_name", "status"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_species_tags(tmp_path):
    csv_path = tmp_path / "tags.csv"
    _write_tags(csv_path, [
        {"species_name": "Echinacea purpurea", "status": "native"},
        {"species_name": "Lythrum salicaria", "status": "invasive"},
    ])
    tags = load_species_tags(csv_path)
    assert tags["echinacea purpurea"] == "native"
    assert tags["lythrum salicaria"] == "invasive"


def test_load_species_tags_missing_status(tmp_path):
    csv_path = tmp_path / "tags.csv"
    _write_tags(csv_path, [{"species_name": "Mystery plant", "status": ""}])
    tags = load_species_tags(csv_path)
    assert tags["mystery plant"] == "unknown"


def test_load_species_tags_missing_file(tmp_path):
    tags = load_species_tags(tmp_path / "nonexistent.csv")
    assert tags == {}


def test_lookup_status_known(tmp_path):
    tags = {"echinacea purpurea": "native"}
    assert lookup_status("Echinacea purpurea", tags) == "native"


def test_lookup_status_unknown():
    assert lookup_status("Alien Plant", {}) == "unknown"


# ---------------------------------------------------------------------------
# collect_images
# ---------------------------------------------------------------------------

def test_collect_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.PNG").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "d.jpeg").write_bytes(b"")

    images = collect_images(tmp_path)
    names = {p.name for p in images}
    assert "a.jpg" in names
    assert "b.PNG" in names
    assert "d.jpeg" in names
    assert "c.txt" not in names
