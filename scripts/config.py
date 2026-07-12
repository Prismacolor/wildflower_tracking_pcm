"""
Central configuration for the wildflower tracking pipeline.
"""

from pathlib import Path
from utils.utils import project_root


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

# Converting videos to stills
FRAMES_PER_SECOND = 1          # extracted frames per second of video
STILLS_FORMAT = "jpg"          # output image format

# Each tuple is (window_size, step_size) in pixels.
# 224 px  → catches small/medium plants
# 448 px  → catches larger plants / clumps
WINDOW_CONFIGS = [
    (224, 112),   # 50 % overlap
    (448, 224),   # 50 % overlap
]

SEGMENT_FORMAT = "jpg"

# Building the convolution model
MODEL_INPUT_SIZE = (224, 224)   # (height, width) to feed into CNN
MODEL_FILE: Path = MODELS_DIR / "plant_classifier.keras"
CONFIDENCE_THRESHOLD = 0.70   # if model confidence is below this, classify as "unknown"
BATCH_SIZE = 32
EPOCHS = 150
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4

# Download data from iNaturalist
INAT_API_BASE: str = "https://api.inaturalist.org/v1"

# Primary place — used for species_tags CSV naming and results scoping
INAT_PRIMARY_PLACE_ID: str = "213020"

# All places to download from — primary first, then supplemental.
INAT_PLACE_IDS: list[str] = [
    "213020",  # Prairie Creek Marsh and Wildscape
    # "89246",  # Add supplemental place IDs here once identified
]

INAT_TAXON_ID = 47125          # iNat taxon ID for angiosperms (flowering plants)
INAT_MAX_PHOTOS_PER_SPECIES = 150
INAT_QUALITY_GRADE = "research"  # only research-grade observations
INAT_MIN_PHOTOS_PER_SPECIES = 15

SPECIES_TAGS_CSV: Path = DATA_DIR / f"species_tags_{INAT_PRIMARY_PLACE_ID}.csv"

# Reporting / visualisation
TOP_N_SPECIES = 5              # top-N chart in distribution report
COMPARE_TOP_N = 10             # top-N for run-to-run comparison chart
