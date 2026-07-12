"""
Shared utility functions used across the wildflower tracking pipeline.
"""

import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path


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


SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)


def collect_images(directory: Path | str) -> list[Path]:
    """Recursively collect all supported image files under *directory*."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def filter_sparse_classes(data_dir: Path | str, min_samples: int = 3) -> list[str]:
    """
    Identify species subdirectories with fewer than min_samples images.

    Does NOT delete anything — just returns the list of sparse species
    names so the caller can decide how to handle them (e.g. exclude from
    a training run while leaving the photos on disk for later).
    """
    sparse = []
    for species_dir in Path(data_dir).iterdir():
        if not species_dir.is_dir():
            continue
        count = len(collect_images(species_dir))
        if count < min_samples:
            sparse.append(species_dir.name)
    return sparse


def build_filtered_training_dir(
    inat_base_dir: Path | str,
    place_ids: list[str],
    working_dir: Path | str,
    min_samples: int = 15,
    ) -> Path:
    """
    Merge photos from all place_id subfolders under inat_base_dir into a
    single flat working directory structured as:
        working_dir/<species_name>/<photos from all places>

    Only species that meet min_samples across the combined pool are included.
    Safe to call repeatedly — clears and rebuilds working_dir each time.

    Parameters
    ----------
    inat_base_dir:
        Root iNat data directory (config.INAT_DIR), containing one
        subdirectory per place_id.
    place_ids:
        List of place_id strings to merge (config.INAT_PLACE_IDS).
    working_dir:
        Temporary directory to write merged species folders into.
    min_samples:
        Minimum combined photo count for a species to be included.
    """
    inat_base_dir = Path(inat_base_dir)
    working_dir = Path(working_dir)

    if working_dir.exists():
        shutil.rmtree(working_dir)
    ensure_dir(working_dir)

    # Collect all species folders across every place
    # species_name -> list of source image paths
    species_sources: dict[str, list[Path]] = {}

    for place_id in place_ids:
        place_dir = inat_base_dir / place_id
        if not place_dir.exists():
            get_logger(__name__).warning(
                f"Place directory not found, skipping: {place_dir}"
            )
            continue
        for species_dir in place_dir.iterdir():
            if not species_dir.is_dir():
                continue
            images = collect_images(species_dir)
            if images:
                species_sources.setdefault(species_dir.name, []).extend(images)

    # Copy only species that meet the minimum sample threshold
    included = 0
    excluded = 0
    for species_name, image_paths in species_sources.items():
        if len(image_paths) < min_samples:
            excluded += 1
            continue
        dest_dir = ensure_dir(working_dir / species_name)
        for src in image_paths:
            # Prefix filename with place_id to avoid collisions across places
            place_id = src.parts[src.parts.index(src.parent.parent.name)]
            dest = dest_dir / f"{place_id}_{src.name}"
            if not dest.exists():
                shutil.copy2(src, dest)
        included += 1

    get_logger(__name__).info(
        f"Training directory built: {included} species included, "
        f"{excluded} excluded (< {min_samples} samples)"
    )

    return working_dir