"""
config.py
Central configuration for the wildflower tracking pipeline.
All tuneable constants live here so nothing is hard-coded elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from scripts.utils import project_root

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT: Path = project_root()

DATA_DIR: Path = ROOT / "data"
VIDEOS_DIR: Path = DATA_DIR / "videos"
STILLS_DIR: Path = DATA_DIR / "stills"
INAT_DIR: Path = DATA_DIR / "iNat_data"
SEGMENTED_DIR: Path = DATA_DIR / "segmented_data"
RESULTS_DIR: Path = ROOT / "results"
MODELS_DIR: Path = ROOT / "models"
VISUALIZATIONS_DIR: Path = ROOT / "visualizations"
ARTIFACTS_DIR: Path = VISUALIZATIONS_DIR / "artifacts"
TREND_CHARTS_DIR: Path = VISUALIZATIONS_DIR / "trend_charts"

SPECIES_TAGS_CSV: Path = DATA_DIR / "species_tags.csv"

# ---------------------------------------------------------------------------
# Video → stills
# ---------------------------------------------------------------------------

FRAMES_PER_SECOND: int = 1          # extracted frames per second of video
STILLS_FORMAT: str = "jpg"          # output image format

# ---------------------------------------------------------------------------
# Sliding-window segmentation
# ---------------------------------------------------------------------------

# Each tuple is (window_size, step_size) in pixels.
# 224 px  → catches small/medium plants
# 448 px  → catches larger plants / clumps
WINDOW_CONFIGS: list[tuple[int, int]] = [
    (224, 112),   # 50 % overlap
    (448, 224),   # 50 % overlap
]

SEGMENT_FORMAT: str = "jpg"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_INPUT_SIZE: tuple[int, int] = (224, 224)   # (height, width) fed to CNN
MODEL_FILE: Path = MODELS_DIR / "plant_classifier.keras"
CONFIDENCE_THRESHOLD: float = 0.60   # below this → "unknown"
BATCH_SIZE: int = 32
EPOCHS: int = 30
VALIDATION_SPLIT: float = 0.2
LEARNING_RATE: float = 1e-4
DROPOUT_RATE: float = 0.4

# ---------------------------------------------------------------------------
# iNat downloader
# ---------------------------------------------------------------------------

INAT_API_BASE: str = "https://api.inaturalist.org/v1"

# Rough bounding box for North Texas (swlat, swlng, nelat, nelng)
INAT_BBOX: dict[str, float] = {
    "swlat": 32.5,
    "swlng": -98.0,
    "nelat": 33.9,
    "nelng": -96.0,
}

INAT_TAXON_ID: int = 47125          # iNat taxon ID for Plantae (plants)
INAT_MAX_PHOTOS_PER_SPECIES: int = 100
INAT_QUALITY_GRADE: str = "research"  # only research-grade observations

# ---------------------------------------------------------------------------
# Reporting / visualisation
# ---------------------------------------------------------------------------

TOP_N_SPECIES: int = 5              # top-N chart in distribution report
COMPARE_TOP_N: int = 10             # top-N for run-to-run comparison chart
