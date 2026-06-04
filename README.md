# Wildflower Tracking PCM

Prairie restoration plant tracking system for Prairie Creek.  
Processes field videos → still frames → segmented patches → CNN species identification → native/invasive distribution reports and trend visualisations.

---

## Project Overview

<!-- Describe your prairie restoration project here -->

---

## Directory Structure

```
wildflower_tracking_pcm/
├── data/
│   ├── videos/                    # Raw video batches (one subfolder per session)
│   │   └── <batch_name>/
│   ├── stills/                    # Extracted frames
│   │   └── prairiecreek_YYYYMMDD/
│   ├── segmented_data/            # Sliding-window patches
│   │   └── segments_YYYYMMDD_HHMMSS/
│   ├── iNat_data/                 # Training photos, organised by species
│   │   └── <species_name>/
│   └── species_tags.csv           # Species → native/invasive lookup table
├── models/
│   ├── plant_classifier.keras     # Trained model
│   └── class_index.json           # Class name index
├── results/
│   ├── results_YYYYMMDD_HHMMSS.csv
│   └── comparison_report_YYYYMMDD_HHMMSS.txt
├── scripts/
│   ├── __init__.py
│   ├── calculator.py              # Run-to-run comparison logic
│   ├── config.py                  # All tunable constants
│   ├── detector.py                # CNN PlantClassifier class
│   ├── extractor.py               # Video → stills
│   ├── main.py                    # CLI entry point
│   ├── processor.py               # Segmentation + prediction pipeline
│   ├── setup_inat.py              # iNaturalist photo downloader
│   └── utils.py                   # Shared helpers
├── tests/
│   ├── test_calculator.py
│   ├── test_charts.py
│   ├── test_extractor.py
│   ├── test_inat.py
│   ├── test_pipeline.py
│   ├── test_segmenter.py
│   └── test_utils.py
├── visualizations/
│   ├── __init__.py
│   ├── artifacts/                 # Per-run distribution charts
│   ├── trend_charts/              # Cross-run comparison and spread charts
│   ├── charts.py                  # Distribution + comparison chart generators
│   └── spread_tracker.py         # Time-series spread visualisations
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Populate species_tags.csv

`data/species_tags.csv` contains a `species_name,status` mapping.  
Add rows as you identify new species.  Status values: `native`, `invasive`.  
Leave `status` blank for newly discovered species — the system will tag them `unknown`.

### 3. Download iNat training photos

```bash
python -m scripts.main download
```

This pulls research-grade plant observations from iNaturalist for the North Texas bounding box and organises photos into `data/iNat_data/<species_name>/`.  
See `scripts/config.py` → `INAT_BBOX` to adjust the region.

### 4. Train the model

```bash
python -m scripts.main train
```

Re-run this command whenever you add new iNat photos.  
Training parameters (epochs, learning rate, etc.) are in `scripts/config.py`.

---

## Running the Pipeline

### Full automated run (extract → segment → predict → charts)

```bash
python -m scripts.main full
```

### Individual steps

| Command | What it does |
|---|---|
| `python -m scripts.main extract` | Convert latest video batch to stills |
| `python -m scripts.main segment` | Segment latest stills into patches |
| `python -m scripts.main predict` | Classify patches, write results CSV |
| `python -m scripts.main charts`  | Generate distribution charts + comparison report |
| `python -m scripts.main spread`  | Generate time-series spread charts |
| `python -m scripts.main evaluate`| Evaluate model accuracy on iNat validation data |

---

## Configuration

All tunable constants are in `scripts/config.py`:

| Constant | Default | Description |
|---|---|---|
| `FRAMES_PER_SECOND` | `1` | Frames extracted per second of video |
| `WINDOW_CONFIGS` | `[(224,112),(448,224)]` | Sliding window (size, step) pairs |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this → "unknown" |
| `INAT_BBOX` | North Texas | Bounding box for iNat downloads |
| `INAT_MAX_PHOTOS_PER_SPECIES` | `100` | Cap photos downloaded per species |
| `TOP_N_SPECIES` | `5` | Species shown in distribution chart |
| `COMPARE_TOP_N` | `10` | Species shown in comparison chart |

---

## Running Tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ -v --cov=scripts --cov=visualizations
```

---

## Output Files

| File | Location | Description |
|---|---|---|
| `results_<ts>.csv` | `results/` | Species counts and percentages per run |
| `comparison_report_<ts>.txt` | `results/` | Text diff between two most recent runs |
| `native_vs_invasive_<ts>.png` | `visualizations/artifacts/` | Status distribution bar chart |
| `top_5_species_<ts>.png` | `visualizations/artifacts/` | Top-N species chart |
| `run_comparison_<ts>.png` | `visualizations/trend_charts/` | Side-by-side run comparison |
| `spread_over_time_<ts>.png` | `visualizations/trend_charts/` | Native/invasive trend line chart |
| `species_heatmap_<ts>.png` | `visualizations/trend_charts/` | Species × run heatmap |

---

## Notes on iNat Downloads

<!-- Add notes about your iNat account, any filters you applied, data quality observations, etc. -->

---

## Field Protocol

<!-- Describe your video capture protocol: camera type, height, walk pattern, lighting conditions, etc. -->

---

## Species Notes

<!-- Record observations about specific species: first sighting, unusual behaviour, etc. -->
<!-- Example: Eryngium leavenworthii (Leavenworth's Eryngo) — first observed 2025 season, NE quadrant -->
