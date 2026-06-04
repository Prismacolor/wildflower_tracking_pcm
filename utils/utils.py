"""
utils.py
Shared utility functions used across the wildflower tracking pipeline.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the project root (parent of the scripts directory)."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it does not exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def datestamp() -> str:
    """Return today's date as YYYYMMDD."""
    return datetime.now().strftime("%Y%m%d")


def timestamp() -> str:
    """Return current datetime as YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def latest_subdirectory(parent: Path | str) -> Path:
    """
    Return the most recently modified subdirectory inside *parent*.

    Raises
    ------
    FileNotFoundError
        If *parent* does not exist or contains no subdirectories.
    """
    parent = Path(parent)
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {parent}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def two_most_recent_files(directory: Path | str, pattern: str = "*.csv") -> tuple[Path, Path]:
    """
    Return the two most recently modified files matching *pattern* in *directory*.

    Raises
    ------
    ValueError
        If fewer than two matching files exist.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(files) < 2:
        raise ValueError(
            f"Need at least 2 files matching '{pattern}' in {directory}, found {len(files)}."
        )
    return files[0], files[1]


# ---------------------------------------------------------------------------
# Species tag lookup
# ---------------------------------------------------------------------------

def load_species_tags(csv_path: Path | str) -> dict[str, str]:
    """
    Load species_tags.csv and return a mapping of species_name → status.

    Expected CSV columns: species_name, status
    Status values should be: 'native', 'invasive', or left blank (→ 'unknown').
    """
    csv_path = Path(csv_path)
    mapping: dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("species_name", "").strip().lower()
            status = row.get("status", "").strip().lower() or "unknown"
            if name:
                mapping[name] = status
    return mapping


def lookup_status(species_name: str, species_tags: dict[str, str]) -> str:
    """Return native/invasive/unknown for a given species name."""
    return species_tags.get(species_name.strip().lower(), "unknown")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)


def collect_images(directory: Path | str) -> list[Path]:
    """Recursively collect all supported image files under *directory*."""
    directory = Path(directory)
    return [
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
