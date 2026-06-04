"""
test_pipeline.py
Integration-level tests for the segment → predict → report data flow.
Uses mocked classifier so no real model is needed.
"""

from __future__ import annotations

import csv
from unittest.mock import patch

import numpy as np
import pytest

from models.processor import PredictionPipeline, SlidingWindowSegmenter


@pytest.fixture()
def prairie_stills(tmp_path):
    """Create a few fake still images for testing."""
    import cv2
    stills = tmp_path / "stills"
    stills.mkdir()
    for i in range(2):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(stills / f"frame_{i:03d}.jpg"), img)
    return stills


def test_segment_then_predict_produces_report(tmp_path, prairie_stills):
    """
    Full data-flow test: segment stills → classify patches → write report CSV.
    The classifier is mocked so this tests the pipeline logic, not the model.
    """
    seg_dir = tmp_path / "segmented"
    results_dir = tmp_path / "results"

    # Segment with a very small window for speed
    segmenter = SlidingWindowSegmenter(window_configs=[(64, 64)], output_format="jpg")
    patches = segmenter.segment_directory(prairie_stills, seg_dir)
    assert len(patches) > 0, "Segmenter produced no patches"

    # Build pipeline with mocked classifier
    pipeline = PredictionPipeline(
        segmented_base=tmp_path,
        results_dir=results_dir,
        species_tags_path=tmp_path / "tags.csv",
    )

    # Patch latest_subdirectory to point at our seg_dir
    with patch("scripts.processor.latest_subdirectory", return_value=seg_dir), \
         patch.object(pipeline.classifier, "load"), \
         patch.object(pipeline.classifier, "predict_image", return_value=("echinacea_purpurea", 0.85)):
        report = pipeline.run()

    assert report.exists()
    with report.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > 0
    assert rows[0]["species"] == "echinacea_purpurea"
    assert float(rows[0]["percentage"]) == pytest.approx(100.0)
