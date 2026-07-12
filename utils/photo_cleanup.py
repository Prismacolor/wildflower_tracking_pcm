"""
cleanup_images.py
Scans data/iNat_data/<place_id>/ for corrupt or unreadable images and
removes them. Run this before training if you hit JPEG decode errors.

Uses Pillow with strict JPEG validation (same decoder TensorFlow uses)
rather than OpenCV, which is too lenient and misses truncated files.

Usage (from project root):
    python -m scripts.cleanup_images
"""

from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm

from scripts import config
from utils.utils import collect_images, get_logger

logger = get_logger(__name__)

# Ensure Pillow raises on truncated files rather than silently returning
# partial image data the way OpenCV does
ImageFile.LOAD_TRUNCATED_IMAGES = False


def remove_corrupt_images(data_dir: Path) -> tuple[int, int]:
    """
    Scan all images under data_dir, attempt to fully decode each one with
    Pillow (which uses the same strict JPEG validation as TensorFlow), and
    delete any that fail. Returns (total_checked, total_removed).

    Two-pass decode strategy:
      1. verify() — catches structural corruption and bad headers
      2. load()   — catches premature end-of-data that verify() misses
    """
    images = collect_images(data_dir)
    logger.info(f"Scanning {len(images)} images in {data_dir}")

    removed = 0
    for img_path in tqdm(images, desc="Checking images", unit="file"):
        try:
            with Image.open(img_path) as img:
                img.verify()   # catches truncation and corruption
            # verify() seeks to end but doesn't fully decode pixel data —
            # re-open and load() to catch premature-end-of-data errors
            with Image.open(img_path) as img:
                img.load()
        except Exception:
            logger.warning(f"Removing corrupt image: {img_path}")
            img_path.unlink()
            removed += 1

    return len(images), removed


def main() -> None:
    data_dir = config.INAT_DIR / config.INAT_PLACE_ID
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    total, removed = remove_corrupt_images(data_dir)
    logger.info(f"Done. Checked {total} images, removed {removed} corrupt files.")


if __name__ == "__main__":
    main()