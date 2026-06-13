"""
charts.py  (visualizations/charts.py)
Generates distribution charts and run-to-run comparison bar graphs.

Usage (from project root):
    python -m visualizations.charts
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scripts import config
from scripts.results_evaluator import ResultsEvaluator
from utils.utils import ensure_dir, get_logger, timestamp, two_most_recent_files

logger = get_logger(__name__)

# Consistent palette for native / invasive / unknown
_STATUS_PALETTE = {
    "native": "#4CAF50",
    "invasive": "#F44336",
    "unknown": "#9E9E9E",
}


def _latest_results_csv() -> Path:
    """Return the most recent results CSV."""
    files = sorted(config.RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No results CSV files found in results/")
    return files[0]


def _load_results(csv_path: Path) -> list[dict]:
    """Load a results CSV into a list of dicts."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "species": row["species"],
                "percentage": float(row["percentage"]),
                "status": row.get("status", "unknown"),
            })
    return rows


# Chart 1 — Native vs invasive distribution (stacked / grouped)
def plot_native_vs_invasive(results_path: Path | None = None, save_dir: Path = config.ARTIFACTS_DIR) -> Path:
    """
    Horizontal bar chart showing native / invasive / unknown proportions.
    Returns the saved file path.
    """
    results_path = results_path or _latest_results_csv()
    rows = _load_results(results_path)
    ensure_dir(save_dir)

    # Aggregate by status
    totals: dict[str, float] = {"native": 0.0, "invasive": 0.0, "unknown": 0.0}
    for r in rows:
        status = r["status"] if r["status"] in totals else "unknown"
        totals[status] += r["percentage"]

    labels = list(totals.keys())
    values = [totals[k] for k in labels]
    colors = [_STATUS_PALETTE[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Percentage of observed patches (%)")
    ax.set_title("Prairie Plant Status Distribution")
    sns.despine(ax=ax)

    out = save_dir / f"native_vs_invasive_{timestamp()}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# Chart 2 — Top-N species by distribution
def plot_top_species(
    results_path: Path | None = None,
    top_n: int = config.TOP_N_SPECIES,
    save_dir: Path = config.ARTIFACTS_DIR,
) -> Path:
    """
    Horizontal bar chart of the top-N species coloured by native/invasive status.
    Returns the saved file path.
    """
    results_path = results_path or _latest_results_csv()
    rows = _load_results(results_path)
    rows = sorted(rows, key=lambda r: r["percentage"], reverse=True)[:top_n]
    ensure_dir(save_dir)

    species = [r["species"].replace("_", " ").title() for r in rows]
    percentages = [r["percentage"] for r in rows]
    colors = [_STATUS_PALETTE.get(r["status"], _STATUS_PALETTE["unknown"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(species[::-1], percentages[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Percentage of observed patches (%)")
    ax.set_title(f"Top {top_n} Prairie Plants by Observed Coverage")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k.title()) for k, v in _STATUS_PALETTE.items()]
    ax.legend(handles=legend_elements, loc="lower right")
    sns.despine(ax=ax)

    out = save_dir / f"top_{top_n}_species_{timestamp()}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# Chart 3 — Run-to-run comparison (top-N species, grouped bars)
def plot_run_comparison(
    top_n: int = config.COMPARE_TOP_N,
    save_dir: Path = config.TREND_CHARTS_DIR,
) -> Path:
    """
    Grouped bar chart comparing the top-N species between the two most
    recent result runs.  Returns the saved file path.
    """
    ensure_dir(save_dir)
    comparator = ResultsEvaluator()
    diff = comparator.compare_latest()

    # Sort by maximum percentage across both runs; take top_n
    all_sp = sorted(
        diff["all_species"],
        key=lambda r: max(r["older_pct"], r["newer_pct"]),
        reverse=True,
    )[:top_n]

    species_labels = [r["species"].replace("_", " ").title() for r in all_sp]
    older_pcts = [r["older_pct"] for r in all_sp]
    newer_pcts = [r["newer_pct"] for r in all_sp]

    newer_name = Path(diff["newer_file"]).stem
    older_name = Path(diff["older_file"]).stem

    x = range(len(species_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width / 2 for i in x], older_pcts, width, label=older_name, color="#78909C")
    ax.bar([i + width / 2 for i in x], newer_pcts, width, label=newer_name, color="#42A5F5")

    ax.set_xticks(list(x))
    ax.set_xticklabels(species_labels, rotation=35, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Top {top_n} Species — Run Comparison")
    ax.legend()
    sns.despine(ax=ax)

    out = save_dir / f"run_comparison_{timestamp()}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# Text comparison report
def generate_comparison_report(save_dir: Path = config.RESULTS_DIR) -> Path:
    """
    Write a plain-text comparison report for the two most recent runs.
    Returns the saved file path.
    """
    ensure_dir(save_dir)
    comparator = ResultsEvaluator()
    diff = comparator.compare_latest()

    ts = timestamp()
    report_path = save_dir / f"comparison_report_{ts}.txt"

    newer_name = Path(diff["newer_file"]).stem
    older_name = Path(diff["older_file"]).stem

    lines = [
        "=" * 60,
        "PRAIRIE WILDFLOWER TRACKING — RUN COMPARISON REPORT",
        "=" * 60,
        f"Newer run : {newer_name}",
        f"Older run : {older_name}",
        "",
    ]

    if diff["added"]:
        lines.append("NEW species (appeared in newer run):")
        for sp in diff["added"]:
            lines.append(f"  + {sp}")
        lines.append("")

    if diff["removed"]:
        lines.append("LOST species (absent from newer run):")
        for sp in diff["removed"]:
            lines.append(f"  - {sp}")
        lines.append("")

    lines.append("CHANGED species (sorted by largest absolute change):")
    changed_sorted = sorted(diff["changed"], key=lambda r: abs(r["delta"]), reverse=True)
    for row in changed_sorted:
        arrow = "▲" if row["delta"] >= 0 else "▼"
        lines.append(
            f"  {arrow} {row['species']:<35s} "
            f"{row['older_pct']:>6.2f}% → {row['newer_pct']:>6.2f}%  "
            f"(Δ {row['delta']:+.2f}%)  [{row['status']}]"
        )

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Comparison report saved: %s", report_path)
    return report_path


def main() -> None:
    ensure_dir(config.ARTIFACTS_DIR)
    ensure_dir(config.TREND_CHARTS_DIR)

    plot_native_vs_invasive()
    plot_top_species()

    try:
        plot_run_comparison()
        generate_comparison_report()
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Skipping comparison charts — %s", exc)


if __name__ == "__main__":
    main()
