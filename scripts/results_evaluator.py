"""
Loads two results CSVs and computes the change in species distribution
between runs.
"""
import csv
from pathlib import Path

from scripts import config
from utils.utils import get_logger, two_most_recent_files

logger = get_logger(__name__)


class ResultsEvaluator:
    """
    Compares two results CSV files (produced by PredictionPipeline) and
    returns a structured diff of species percentage changes.
    """

    def __init__(self, results_dir: Path = config.RESULTS_DIR) -> None:
        self.results_dir = Path(results_dir)

    def compare_latest(self) -> dict:
        """
        Automatically locate the two most recent result files and compare them.
        Returns the comparison dict (see _compare).
        """
        newer, older = two_most_recent_files(self.results_dir, "*.csv")
        logger.info("Comparing:\n  newer → %s\n  older → %s", newer.name, older.name)
        return self.compare(newer, older)

    def compare(self, newer: Path, older: Path) -> dict:
        """
        Compare two result CSV files.

        Returns
        -------
        dict with keys:
            newer_file, older_file,
            added    → species in newer but not older
            removed  → species in older but not newer
            changed  → list of {species, older_pct, newer_pct, delta}
            all_species → combined sorted list for charting
        """
        newer_data = self._load_csv(newer)
        older_data = self._load_csv(older)

        newer_keys = set(newer_data)
        older_keys = set(older_data)

        added = sorted(newer_keys - older_keys)
        removed = sorted(older_keys - newer_keys)
        common = newer_keys & older_keys

        changed = []
        for species in sorted(common):
            old_pct = older_data[species]["percentage"]
            new_pct = newer_data[species]["percentage"]
            changed.append({
                "species": species,
                "older_pct": old_pct,
                "newer_pct": new_pct,
                "delta": round(new_pct - old_pct, 2),
                "status": newer_data[species]["status"],
            })

        # All species across both runs (for comparison charts)
        all_species = sorted(newer_keys | older_keys)
        all_rows = []
        for sp in all_species:
            all_rows.append({
                "species": sp,
                "older_pct": older_data.get(sp, {}).get("percentage", 0.0),
                "newer_pct": newer_data.get(sp, {}).get("percentage", 0.0),
                "status": (newer_data.get(sp) or older_data.get(sp, {})).get("status", "unknown"),
            })

        return {
            "newer_file": str(newer),
            "older_file": str(older),
            "added": added,
            "removed": removed,
            "changed": changed,
            "all_species": all_rows,
        }

    @staticmethod
    def _load_csv(path: Path) -> dict[str, dict]:
        """Load a results CSV and return a dict keyed by species name."""
        data: dict[str, dict] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                species = row["species"].strip()
                data[species] = {
                    "count": int(row.get("count", 0)),
                    "percentage": float(row.get("percentage", 0.0)),
                    "status": row.get("status", "unknown"),
                }
        return data


def main() -> None:
    evaluator = ResultsEvaluator()
    diff = evaluator.compare_latest()
    print(f"Comparing {Path(diff['newer_file']).name} vs {Path(diff['older_file']).name}")
    print(f"  Added species:   {diff['added']}")
    print(f"  Removed species: {diff['removed']}")
    print(f"  Changed species: {len(diff['changed'])}")


if __name__ == "__main__":
    main()
