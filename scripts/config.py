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

SPECIES_TAGS_CSV: Path = DATA_DIR / "species_tags.csv"

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
CONFIDENCE_THRESHOLD = 0.60   # if model confidence is below this, classify as "unknown"
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4

# Download data from iNaturalist
INAT_API_BASE: str = "https://api.inaturalist.org/v1"

# Rough bounding box for North Texas (swlat, swlng, nelat, nelng)
INAT_BBOX = {
    "swlat": 32.5,
    "swlng": -98.0,
    "nelat": 33.9,
    "nelng": -96.0,
}

INAT_TAXON_ID = 47125          # iNat taxon ID for angiosperms (flowering plants)
INAT_MAX_PHOTOS_PER_SPECIES = 100
INAT_QUALITY_GRADE = "research"  # only research-grade observations

# Reporting / visualisation
TOP_N_SPECIES = 5              # top-N chart in distribution report
COMPARE_TOP_N = 10             # top-N for run-to-run comparison chart
