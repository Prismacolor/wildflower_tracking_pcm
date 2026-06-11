"""
test_segmenter.py
Tests for scripts/processor.py — SlidingWindowSegmenter and PredictionPipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from scripts.processor import PredictionPipeline, SlidingWindowSegmenter


# ---------------------------------------------------------------------------
# SlidingWindowSegmenter
# ---------------------------------------------------------------------------

class TestSlidingWindowSegmenter:

    def _segmenter(self) -> SlidingWindowSegmenter:
        return SlidingWindowSegmenter(
            window_configs=[(64, 32)],   # small for test speed
            output_format="jpg",
        )

    def test_segment_image_produces_patches(self, tmp_path):
        """A 128×128 image with window=64 step=32 should yield multiple patches."""
        import cv2
        segmenter = self._segmenter()
        img_path = tmp_path / "test.jpg"
        # Write a solid-colour 128×128 image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        out_dir = tmp_path / "patches"
        out_dir.mkdir()
        patches = segmenter.segment_image(img_path, out_dir)
        assert len(patches) > 0
        for p in patches:
            assert p.exists()

    def test_segment_image_unreadable(self, tmp_path, caplog):
        """Unreadable file should return empty list without raising."""
        import logging
        segmenter = self._segmenter()
        bad_path = tmp_path / "bad.jpg"
        bad_path.write_bytes(b"not an image")
        out_dir = tmp_path / "patches"
        out_dir.mkdir()
        with caplog.at_level(logging.WARNING, logger="scripts.processor"):
            result = segmenter.segment_image(bad_path, out_dir)
        assert result == []

    def test_segment_directory(self, tmp_path):
        """segment_directory should process all images in a folder."""
        import cv2
        segmenter = self._segmenter()
        stills_dir = tmp_path / "stills"
        stills_dir.mkdir()
        out_dir = tmp_path / "patches"

        for i in range(3):
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(stills_dir / f"frame_{i}.jpg"), img)

        patches = segmenter.segment_directory(stills_dir, out_dir)
        assert out_dir.is_dir()
        assert len(patches) > 0


# ---------------------------------------------------------------------------
# PredictionPipeline
# ---------------------------------------------------------------------------

class TestPredictionPipeline:

    def _write_dummy_results(self, path: Path) -> None:
        """Write a minimal results CSV for testing."""
        rows = [
            {"species": "echinacea_purpurea", "count": 10, "percentage": 50.0, "status": "native"},
            {"species": "lythrum_salicaria",  "count":  6, "percentage": 30.0, "status": "invasive"},
            {"species": "unknown",             "count":  4, "percentage": 20.0, "status": "unknown"},
        ]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["species", "count", "percentage", "status"])
            writer.writeheader()
            writer.writerows(rows)

    def test_save_report_writes_csv(self, tmp_path):
        """_save_report should write a valid CSV with the right columns."""
        pipeline = PredictionPipeline(
            segmented_base=tmp_path / "seg",
            results_dir=tmp_path / "results",
            species_tags_path=tmp_path / "tags.csv",
        )
        predictions = [
            {"patch": "p1.jpg", "species": "echinacea", "confidence": 0.9, "status": "native"},
            {"patch": "p2.jpg", "species": "echinacea", "confidence": 0.8, "status": "native"},
            {"patch": "p3.jpg", "species": "unknown",   "confidence": 0.3, "status": "unknown"},
        ]
        report = pipeline._save_report(predictions)
        assert report.exists()
        with report.open() as fh:
            rows = list(csv.DictReader(fh))
        species_names = [r["species"] for r in rows]
        assert "echinacea" in species_names

    def test_classify_patches_handles_bad_image(self, tmp_path):
        """Bad image files should be skipped gracefully."""
        pipeline = PredictionPipeline(
            segmented_base=tmp_path / "seg",
            results_dir=tmp_path / "results",
            species_tags_path=tmp_path / "tags.csv",
        )
        pipeline.classifier = MagicMock()
        pipeline.classifier.predict_image.return_value = ("echinacea", 0.9)

        bad_img = tmp_path / "bad.jpg"
        bad_img.write_bytes(b"not an image")

        results = pipeline._classify_patches([bad_img])
        # Bad image → skipped → empty list
        assert results == []
