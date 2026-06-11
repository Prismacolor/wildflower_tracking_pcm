"""
segmenter.py — SAM Segmentation Wrapper
-------------------------------------------
Takes a frame and draws borders around every distinct plant region.
Returns cropped images of each region for the classifier.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import config

logger = logging.getLogger(__name__)


@dataclass
class MaskResult:
    """One segmented plant region from a frame."""

    mask: np.ndarray         # Boolean (H, W) — True = this region
    cropped_image: np.ndarray  # RGB crop with non-mask pixels blacked out
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area_pixels: int
    area_fraction: float     # Fraction of total image area
    stability: float         # SAM's confidence for this mask


class Segmenter:
    """
    Wraps Meta's Segment Anything Model.

    SAM is pretrained — it already knows how to find objects.
    We load it, point it at a frame, and it returns cropped regions.

    Usage:
        seg = Segmenter()
        seg.load()
        masks = seg.predict(frame)
    """

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or config.SAM_CHECKPOINT
        self.device = config.get_device()
        self._model = None
        self._generator = None
        self.is_loaded = False

    def load(self) -> None:
        """Load SAM weights. Call once before predict()."""
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint}\n"
                "Run: python setup_sam.py"
            )

        logger.info("Loading SAM (%s) on %s", config.SAM_MODEL_TYPE, self.device)

        self._model = sam_model_registry[config.SAM_MODEL_TYPE](
            checkpoint=str(self.checkpoint)
        )
        self._model.to(self.device)

        self._generator = SamAutomaticMaskGenerator(
            model=self._model,
            points_per_side=config.SAM_POINTS_PER_SIDE,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )
        self.is_loaded = True
        logger.info("SAM ready.")

    def predict(self, image: np.ndarray) -> list[MaskResult]:
        """
        Segment a frame into plant regions.

        Returns list of MaskResult, filtered by size, sorted largest first.
        """
        if not self.is_loaded:
            raise RuntimeError("Call load() before predict().")

        total_pixels = image.shape[0] * image.shape[1]
        min_area = config.SAM_MIN_MASK_AREA_PCT / 100.0
        max_area = config.SAM_MAX_MASK_AREA_PCT / 100.0

        raw_masks = self._generator.generate(image)

        results: list[MaskResult] = []
        for m in raw_masks:
            area = int(m["area"])
            frac = area / total_pixels

            if frac < min_area or frac > max_area:
                continue

            bool_mask = m["segmentation"]
            bbox = tuple(m["bbox"])
            cropped = self._crop(image, bool_mask, bbox)

            results.append(MaskResult(
                mask=bool_mask,
                cropped_image=cropped,
                bbox=bbox,
                area_pixels=area,
                area_fraction=frac,
                stability=float(m["stability_score"]),
            ))

        results.sort(key=lambda r: r.area_pixels, reverse=True)
        return results

    @staticmethod
    def _crop(
        image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop to bounding box and black out non-mask pixels."""
        x, y, w, h = bbox
        crop = image[y:y + h, x:x + w].copy()
        crop_mask = mask[y:y + h, x:x + w]
        crop[~crop_mask] = 0
        return crop

    @staticmethod
    def get_vegetation_fraction(masks: list[MaskResult]) -> float:
        """Union all masks and return total vegetation fraction."""
        if not masks:
            return 0.0
        combined = np.zeros_like(masks[0].mask, dtype=bool)
        for m in masks:
            combined |= m.mask
        return float(combined.sum() / combined.size)
