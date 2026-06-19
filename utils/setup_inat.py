"""
Downloads iNaturalist research-grade plant observations (with photos) for a
North Texas bounding box and saves them into data/iNat_data/<species_name>/.
"""

import time
import urllib.request
from pathlib import Path
import requests
import csv
from tqdm import tqdm

from scripts import config
from utils.utils import ensure_dir, get_logger, load_species_tags

logger = get_logger(__name__)

_OBSERVATIONS_ENDPOINT = f"{config.INAT_API_BASE}/observations"
_PAGE_SIZE = 200
_REQUEST_DELAY = 1.0   # seconds between API calls — avoid flooding the system


class INatDownloader:
    """
    Fetches plant observations via the iNaturalist v1 API and downloads
    the associated photos, organised by species.
    """

    def __init__(
        self,
        output_dir: Path = config.INAT_DIR,
        bbox: dict[str, float] = config.INAT_BBOX,
        taxon_id: int = config.INAT_TAXON_ID,
        max_photos_per_species: int = config.INAT_MAX_PHOTOS_PER_SPECIES,
        quality_grade: str = config.INAT_QUALITY_GRADE,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.bbox = bbox
        self.taxon_id = taxon_id
        self.max_photos = max_photos_per_species
        self.quality_grade = quality_grade

    def run(self) -> None:
        ensure_dir(self.output_dir)
        logger.info(f"Starting iNat download — bounding box: {self.bbox}")

        observations = self._fetch_all_observations()
        logger.info(f"Fetched {len(observations)} observations.")

        species_map = self._group_by_species(observations)
        logger.info(f"Unique species found: {len(species_map)}")

        self._update_species_tags(species_map)

        for species, data in tqdm(species_map.items(), desc="Downloading species", unit="species"):
            self._download_species_photos(species, data["urls"])

        logger.info("iNat download complete.")


    def _fetch_all_observations(self) -> list[dict]:
        """Page through the iNat API and return all matching observations."""
        observations: list[dict] = []
        page = 1

        while True:
            params = {
                "taxon_id": self.taxon_id,
                "quality_grade": self.quality_grade,
                "photos": "true",
                "per_page": _PAGE_SIZE,
                "page": page,
                "swlat": self.bbox["swlat"],
                "swlng": self.bbox["swlng"],
                "nelat": self.bbox["nelat"],
                "nelng": self.bbox["nelng"],
            }

            response = requests.get(_OBSERVATIONS_ENDPOINT, params=params, timeout=30)
            if response.status_code == 403: # we have hit the 10k limit for iNat
                logger.info("We have hit the 10k download limit for iNat, stopping download...")
                break
            elif response.status_code != 200:  # catch any other http errors
                logger.warning(f"Error while downloading: {response.status_code}")
                continue

            response.raise_for_status()

            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            observations.extend(results)
            logger.info(f"  Page {page}: {len(results)} observations (total so far: {len(observations)})")

            if len(observations) >= data.get("total_results", 0):
                break

            page += 1
            time.sleep(_REQUEST_DELAY)

        return observations

    def _group_by_species(self, observations: list[dict]) -> dict[str, dict]:
        """
        Return a dict mapping species_name -> {urls: [...], status: str}.
        Caps each species at self.max_photos.
        """
        species_map: dict[str, dict] = {}

        for obs in observations:
            taxon = obs.get("taxon")
            if not taxon:
                continue

            name: str = taxon.get("name", "").strip().replace(" ", "_").lower()
            if not name:
                continue

            establishment = taxon.get("establishment_means", {}) or {}
            status = establishment.get("establishment_means", "").lower()
            if status == "introduced":
                status = "invasive"
            elif status not in ("native", "endemic"):
                status = ""

            photos = obs.get("photos", [])
            species_map.setdefault(name, {"urls": [], "status": status})
            for photo in photos:
                url = photo.get("url", "")
                if not url:
                    continue
                url = url.replace("/square", "/large")
                if len(species_map[name]["urls"]) < self.max_photos:
                    species_map[name]["urls"].append(url)

        return species_map

    def _update_species_tags(self, species_map: dict[str, dict]) -> None:
        """
        Write any newly discovered species and their iNat status into
        species_tags.csv. Existing rows are preserved and never overwritten.
        """
        csv_path = config.SPECIES_TAGS_CSV
        existing = load_species_tags(csv_path)

        new_rows = []
        for name, data in species_map.items():
            readable_name = name.replace("_", " ")
            if readable_name not in existing:
                new_rows.append({
                    "species_name": readable_name,
                    "status": data["status"],
                })

        if not new_rows:
            logger.info("species_tags.csv is already up to date.")
            return

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["species_name", "status"])
            writer.writerows(new_rows)

        logger.info(f"Added {len(new_rows)} new species to species_tags.csv.")

    def _download_species_photos(self, species: str, urls: list[str]) -> None:
        """Download photos for a single species into its own subdirectory."""
        species_dir = ensure_dir(self.output_dir / species)
        existing = len(list(species_dir.glob("*.jpg")))

        downloaded = 0
        for idx, url in enumerate(urls):
            dest = species_dir / f"{species}_{idx:04d}.jpg"
            if dest.exists():
                continue
            try:
                urllib.request.urlretrieve(url, dest)
                downloaded += 1
                time.sleep(0.2)
            except Exception as exc:
                logger.warning(f"Failed to download {url}: {exc}")

        logger.info(f"{species} — downloaded {downloaded} new photos ({existing} already present)")


def main() -> None:
    downloader = INatDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
