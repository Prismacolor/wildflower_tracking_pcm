"""
Top-level orchestration script for the wildflower tracking pipeline.

Subcommands
-----------
extract     — Convert most-recent video batch → still frames
segment     — Sliding-window segment stills → patches
predict     — Run patches through plant classifier → results CSV
charts      — Generate distribution charts and comparison report
spread      — Generate spread-over-time charts
train       — (Re)train the plant classifier on iNat data
evaluate    — Evaluate model on iNat validation data
download    — Download iNat photos for North Texas
full        — Run extract → segment → predict → charts in sequence

Usage (from project root): python -m scripts.main <subcommand>
"""

import argparse
import sys

from scripts import config
from utils.utils import get_logger, latest_subdirectory, timestamp
from scripts.extractor import VideoExtractor
from scripts.processor import SlidingWindowSegmenter, PredictionPipeline
from scripts.plant_classifier import PlantClassifier
from visualizations.charts import main as charts_main
from visualizations.spread_tracker import main as spread_main
from utils.setup_inat import INatDownloader


logger = get_logger(__name__)

def cmd_extract(_args: argparse.Namespace) -> None:
    """Build the video extractor"""
    VideoExtractor().run()


def cmd_segment(_args: argparse.Namespace) -> None:
    """build the segmentation class"""
    stills_dir = latest_subdirectory(config.STILLS_DIR)
    out_dir = config.SEGMENTED_DIR / f"segments_{timestamp()}"
    SlidingWindowSegmenter().segment_directory(stills_dir, out_dir)


def cmd_predict(_args: argparse.Namespace) -> None:
    """Inference"""
    report = PredictionPipeline().run()
    print(f"Results: {report}")


def cmd_charts(_args: argparse.Namespace) -> None:
    """Build the visualization charts"""
    charts_main()


def cmd_spread(_args: argparse.Namespace) -> None:
    """Track the spread"""
    spread_main()


def cmd_train(_args: argparse.Namespace) -> None:
    """Train the plant classifier"""
    PlantClassifier().train(data_dir=config.INAT_DIR)


def cmd_evaluate(_args: argparse.Namespace) -> None:
    """Evaluate the Classifier model"""
    clf = PlantClassifier()
    clf.load()
    metrics = clf.evaluate()
    print(f"Loss: {metrics['loss']:.4f}  Accuracy: {metrics['accuracy']:.4f}")


def cmd_download(_args: argparse.Namespace) -> None:
    """Download iNat data"""
    INatDownloader().run()


def cmd_full(_args: argparse.Namespace) -> None:
    """End-to-end: extract → segment → predict → charts."""
    logger.info("=== Step 1/4: Extract frames ===")
    cmd_extract(_args)

    logger.info("=== Step 2/4: Segment stills ===")
    cmd_segment(_args)

    logger.info("=== Step 3/4: Predict species ===")
    cmd_predict(_args)

    logger.info("=== Step 4/4: Generate charts ===")
    cmd_charts(_args)

    logger.info("Pipeline complete.")


_COMMANDS = {
    "extract": cmd_extract,
    "segment": cmd_segment,
    "predict": cmd_predict,
    "charts": cmd_charts,
    "spread": cmd_spread,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "download": cmd_download,
    "full": cmd_full,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.main",
        description="Wildflower tracking pipeline for prairie restoration.",
    )
    parser.add_argument(
        "command",
        choices=list(_COMMANDS.keys()),
        help="Pipeline step to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
