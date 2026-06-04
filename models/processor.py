"""
processor.py
Sliding-window segmentation of prairie stills into individual plant patches,
and pipeline to run patches through the PlantClassifier.

Usage (from project root):
    python -m scripts.processor
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2

from scripts import config
from models.plant_classifier import PlantClassifier
from scripts.utils import (
    collect_images,
    ensure_dir,
    get_logger,
    latest_subdirectory,
    load_species_tags,
    lookup_status,
    timestamp,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

class SlidingWindowSegmenter:
    """
    Cuts each still image into overlapping patches using multiple window sizes.

    Parameters
    ----------
    window_configs:
        List of (window_size, step_size) tuples in pixels.
    output_format:
        Image format for saved patches (e.g. 'jpg').
    """

    def __init__(
        self,
        window_configs: list[tuple[int, int]] = config.WINDOW_CONFIGS,
        output_format: str = config.SEGMENT_FORMAT,
    ) -> None:
        self.window_configs = window_configs
        self.output_format = output_format

    def segment_image(self, image_path: Path, output_dir: Path) -> list[Path]:
        """
        Slice a single image into patches across all window configs.
        Returns the list of saved patch paths.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Cannot read image: %s", image_path)
            return []

        h, w = image.shape[:2]
        saved: list[Path] = []
        stem = image_path.stem

        for win_size, step in self.window_configs:
            for y in range(0, h - win_size + 1, step):
                for x in range(0, w - win_size + 1, step):
                    patch = image[y: y + win_size, x: x + win_size]
                    fname = (
                        output_dir
                        / f"{stem}_w{win_size}_y{y:04d}_x{x:04d}.{self.output_format}"
                    )
                    cv2.imwrite(str(fname), patch)
                    saved.append(fname)

        return saved

    def segment_directory(self, stills_dir: Path, output_dir: Path) -> list[Path]:
        """Segment every image in *stills_dir* and save patches to *output_dir*."""
        ensure_dir(output_dir)
        images = collect_images(stills_dir)
        logger.info("Segmenting %d images from %s", len(images), stills_dir)

        all_patches: list[Path] = []
        for img_path in images:
            patches = self.segment_image(img_path, output_dir)
            all_patches.extend(patches)

        logger.info("Total patches saved: %d → %s", len(all_patches), output_dir)
        return all_patches


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

class PredictionPipeline:
    """
    Loads the most recent segmented patch directory, runs each patch through
    the PlantClassifier, and writes a CSV results report.
    """

    def __init__(
        self,
        segmented_base: Path = config.SEGMENTED_DIR,
        results_dir: Path = config.RESULTS_DIR,
        species_tags_path: Path = config.SPECIES_TAGS_CSV,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.segmented_base = Path(segmented_base)
        self.results_dir = Path(results_dir)
        self.species_tags = load_species_tags(species_tags_path)
        self.confidence_threshold = confidence_threshold
        self.classifier = PlantClassifier(confidence_threshold=confidence_threshold)

    def run(self) -> Path:
        """
        Run the full prediction pipeline.  Returns path to the saved CSV report.
        """
        self.classifier.load()
        patch_dir = latest_subdirectory(self.segmented_base)
        logger.info("Predicting on patches in: %s", patch_dir)

        patches = collect_images(patch_dir)
        logger.info("Patches to classify: %d", len(patches))

        predictions = self._classify_patches(patches)
        report_path = self._save_report(predictions)
        return report_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_patches(self, patches: list[Path]) -> list[dict]:
        """Classify each patch and return a list of result dicts."""
        results = []
        for patch_path in patches:
            image = cv2.imread(str(patch_path))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            species, confidence = self.classifier.predict_image(image_rgb)
            status = lookup_status(species, self.species_tags) if species != "unknown" else "unknown"
            results.append({
                "patch": patch_path.name,
                "species": species,
                "confidence": round(confidence, 4),
                "status": status,
            })
        return results

    def _save_report(self, predictions: list[dict]) -> Path:
        """Aggregate predictions into percentages and write CSV."""
        ensure_dir(self.results_dir)
        ts = timestamp()
        report_path = self.results_dir / f"results_{ts}.csv"

        total = len(predictions)
        counts: Counter = Counter(p["species"] for p in predictions)

        rows = []
        for species, count in counts.most_common():
            sample = next(p for p in predictions if p["species"] == species)
            rows.append({
                "species": species,
                "count": count,
                "percentage": round(count / total * 100, 2) if total else 0.0,
                "status": sample["status"],
            })

        fieldnames = ["species", "count", "percentage", "status"]
        with report_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Results saved: %s  (%d species, %d patches)", report_path, len(rows), total)
        return report_path


# ---------------------------------------------------------------------------
# Convenience runner — segment then predict
# ---------------------------------------------------------------------------

class SegmentAndPredict:
    """Orchestrates segmentation → prediction in one call."""

    def __init__(self) -> None:
        self.segmenter = SlidingWindowSegmenter()
        self.pipeline = PredictionPipeline()

    def run(self) -> Path:
        stills_dir = latest_subdirectory(config.STILLS_DIR)
        output_dir = config.SEGMENTED_DIR / f"segments_{timestamp()}"
        self.segmenter.segment_directory(stills_dir, output_dir)
        return self.pipeline.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    runner = SegmentAndPredict()
    report = runner.run()
    print(f"Report saved to: {report}")


if __name__ == "__main__":
    main()
