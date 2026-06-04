"""
test_calculator.py
Tests for scripts/results_evaluator.py — ResultsComparator.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from scripts.results_evaluator import ResultsComparator


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["species", "count", "percentage", "status"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


OLDER_ROWS = [
    {"species": "echinacea_purpurea", "count": 10, "percentage": 40.0, "status": "native"},
    {"species": "lythrum_salicaria",  "count":  8, "percentage": 32.0, "status": "invasive"},
    {"species": "solidago_canadensis","count":  7, "percentage": 28.0, "status": "native"},
]

NEWER_ROWS = [
    {"species": "echinacea_purpurea", "count": 12, "percentage": 50.0, "status": "native"},
    {"species": "lythrum_salicaria",  "count":  5, "percentage": 20.0, "status": "invasive"},
    {"species": "asclepias_tuberosa", "count":  8, "percentage": 30.0, "status": "native"},
]


@pytest.fixture()
def results_dir(tmp_path) -> Path:
    older = tmp_path / "results_20260501_120000.csv"
    _write_csv(older, OLDER_ROWS)
    time.sleep(0.02)
    newer = tmp_path / "results_20260601_120000.csv"
    _write_csv(newer, NEWER_ROWS)
    return tmp_path


class TestResultsComparator:

    def test_compare_detects_added_species(self, results_dir):
        comp = ResultsComparator(results_dir=results_dir)
        diff = comp.compare_latest()
        assert "asclepias_tuberosa" in diff["added"]

    def test_compare_detects_removed_species(self, results_dir):
        comp = ResultsComparator(results_dir=results_dir)
        diff = comp.compare_latest()
        assert "solidago_canadensis" in diff["removed"]

    def test_compare_calculates_delta(self, results_dir):
        comp = ResultsComparator(results_dir=results_dir)
        diff = comp.compare_latest()
        echinacea = next(r for r in diff["changed"] if r["species"] == "echinacea_purpurea")
        assert echinacea["delta"] == pytest.approx(10.0)

    def test_compare_file_references(self, results_dir):
        comp = ResultsComparator(results_dir=results_dir)
        diff = comp.compare_latest()
        assert "20260601" in diff["newer_file"]
        assert "20260501" in diff["older_file"]

    def test_compare_raises_on_insufficient_files(self, tmp_path):
        only = tmp_path / "results_only.csv"
        _write_csv(only, OLDER_ROWS)
        comp = ResultsComparator(results_dir=tmp_path)
        with pytest.raises(ValueError):
            comp.compare_latest()

    def test_all_species_contains_union(self, results_dir):
        comp = ResultsComparator(results_dir=results_dir)
        diff = comp.compare_latest()
        all_names = {r["species"] for r in diff["all_species"]}
        assert "echinacea_purpurea" in all_names
        assert "solidago_canadensis" in all_names
        assert "asclepias_tuberosa" in all_names
