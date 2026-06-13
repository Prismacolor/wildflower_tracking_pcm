"""
spread_tracker.py  (visualizations/spread_tracker.py)
Tracks and visualises the spread of native vs invasive plants over time
by reading all results CSVs from the results directory.

Usage (from project root):
    python -m visualizations.spread_tracker
"""

import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts import config
from utils.utils import ensure_dir, get_logger, timestamp

logger = get_logger(__name__)

_STATUS_PALETTE = {
    "native": "#4CAF50",
    "invasive": "#F44336",
    "unknown": "#9E9E9E",
}

_TIMESTAMP_RE = re.compile(r"results_(\d{8}_\d{6})\.csv")


def _load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load every results_*.csv from *results_dir* and return a combined
    DataFrame with columns: run_date, species, percentage, status.
    """
    frames = []
    for csv_path in sorted(results_dir.glob("results_*.csv")):
        match = _TIMESTAMP_RE.search(csv_path.name)
        run_label = match.group(1) if match else csv_path.stem
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                frames.append({
                    "run_date": run_label,
                    "species": row["species"],
                    "percentage": float(row.get("percentage", 0)),
                    "status": row.get("status", "unknown"),
                })
    if not frames:
        raise FileNotFoundError("No results CSVs found.")
    return pd.DataFrame(frames)


def plot_status_over_time(
    results_dir: Path = config.RESULTS_DIR,
    save_dir: Path = config.TREND_CHARTS_DIR,
) -> Path:
    """
    Line chart showing native / invasive / unknown percentage totals across
    all historical runs.  Returns saved file path.
    """
    ensure_dir(save_dir)
    df = _load_all_results(results_dir)

    # Aggregate by run and status
    agg = (
        df.groupby(["run_date", "status"])["percentage"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for status, color in _STATUS_PALETTE.items():
        subset = agg[agg["status"] == status]
        if subset.empty:
            continue
        ax.plot(
            subset["run_date"],
            subset["percentage"],
            marker="o",
            label=status.title(),
            color=color,
        )

    ax.set_xlabel("Run")
    ax.set_ylabel("Total coverage (%)")
    ax.set_title("Native vs Invasive Plant Coverage Over Time")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    sns.despine(ax=ax)

    out = save_dir / f"spread_over_time_{timestamp()}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_species_heatmap(
    results_dir: Path = config.RESULTS_DIR,
    save_dir: Path = config.TREND_CHARTS_DIR,
    top_n: int = 20,
) -> Path:
    """
    Heatmap of top-N species percentage across all runs.
    Rows = species, columns = run date.
    Returns saved file path.
    """
    ensure_dir(save_dir)
    df = _load_all_results(results_dir)

    # Pivot: species × run_date
    pivot = df.pivot_table(index="species", columns="run_date", values="percentage", fill_value=0.0)

    # Keep only the top_n species by max percentage across any run
    top_species = pivot.max(axis=1).nlargest(top_n).index
    pivot = pivot.loc[top_species]

    fig_height = max(6, len(pivot) * 0.4)
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), fig_height))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGn", ax=ax, linewidths=0.5)
    ax.set_title(f"Top {top_n} Species Coverage (%) Across Runs")
    ax.set_xlabel("Run Date")
    ax.set_ylabel("Species")
    plt.yticks(rotation=0)

    out = save_dir / f"species_heatmap_{timestamp()}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def main() -> None:
    try:
        plot_status_over_time()
        plot_species_heatmap()
    except FileNotFoundError as exc:
        logger.error("Cannot generate spread charts: %s", exc)


if __name__ == "__main__":
    main()
