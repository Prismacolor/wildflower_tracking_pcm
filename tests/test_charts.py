"""
test_charts.py
Tests for visualizations/charts.py — chart generation functions.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["species", "count", "percentage", "status"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_ROWS = [
    {"species": "echinacea_purpurea", "count": 20, "percentage": 40.0, "status": "native"},
    {"species": "lythrum_salicaria",  "count": 15, "percentage": 30.0, "status": "invasive"},
    {"species": "solidago_canadensis","count": 10, "percentage": 20.0, "status": "native"},
    {"species": "unknown",            "count":  5, "percentage": 10.0, "status": "unknown"},
]


@pytest.fixture()
def results_csv(tmp_path) -> Path:
    p = tmp_path / "results_20260601_120000.csv"
    _write_results_csv(p, SAMPLE_ROWS)
    return p


@pytest.fixture()
def two_result_csvs(tmp_path) -> Path:
    older = tmp_path / "results_20260501_120000.csv"
    _write_results_csv(older, SAMPLE_ROWS)
    time.sleep(0.02)
    newer = tmp_path / "results_20260601_120000.csv"
    _write_results_csv(newer, [
        {"species": "echinacea_purpurea", "count": 25, "percentage": 50.0, "status": "native"},
        {"species": "lythrum_salicaria",  "count": 10, "percentage": 20.0, "status": "invasive"},
        {"species": "unknown",            "count": 15, "percentage": 30.0, "status": "unknown"},
    ])
    return tmp_path


class TestPlotNativeVsInvasive:
    def test_creates_png(self, results_csv, tmp_path):
        from visualizations.charts import plot_native_vs_invasive
        out = plot_native_vs_invasive(results_path=results_csv, save_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".png"


class TestPlotTopSpecies:
    def test_creates_png(self, results_csv, tmp_path):
        from visualizations.charts import plot_top_species
        out = plot_top_species(results_path=results_csv, top_n=3, save_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".png"

    def test_respects_top_n(self, results_csv, tmp_path):
        """File should be created regardless of whether top_n > available species."""
        from visualizations.charts import plot_top_species
        out = plot_top_species(results_path=results_csv, top_n=100, save_dir=tmp_path)
        assert out.exists()


class TestPlotRunComparison:
    def test_creates_png(self, two_result_csvs, tmp_path, monkeypatch):
        from scripts import config
        monkeypatch.setattr(config, "RESULTS_DIR", two_result_csvs)
        from visualizations.charts import plot_run_comparison
        out = plot_run_comparison(top_n=5, save_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".png"


class TestGenerateComparisonReport:
    def test_creates_text_file(self, two_result_csvs, tmp_path, monkeypatch):
        from scripts import config
        monkeypatch.setattr(config, "RESULTS_DIR", two_result_csvs)
        from visualizations.charts import generate_comparison_report
        out = generate_comparison_report(save_dir=tmp_path)
        assert out.exists()
        content = out.read_text()
        assert "COMPARISON REPORT" in content
