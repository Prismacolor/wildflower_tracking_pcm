"""
Downloads iNaturalist research-grade angiosperm (flowering plant) observations
for a configured iNat "Place", and saves the photos into
data/iNat_data/<place_id>/<species_name>/.

Scoping downloads by place_id (rather than hard-coding one location) means
this project can be reused for other restoration sites — just change
INAT_PLACE_ID in config.py and re-run the download.
"""

import csv
import time
import urllib.request
from pathlib import Path

import requests
from tqdm import tqdm

from scripts import config
from utils.utils import ensure_dir, get_logger, load_species_tags

logger = get_logger(__name__)

_OBSERVATIONS_ENDPOINT = f"{config.INAT_API_BASE}/observations"
_PAGE_SIZE = 200
_INAT_MAX_OFFSET = 10_000   # iNat hard cap — API returns 403 beyond this
_REQUEST_DELAY = 1.0        # seconds between API calls — be polite to iNat


class INatDownloader:
    """
    Fetches plant observations via the iNaturalist v1 API, scoped to a
    configured iNat place, and downloads the associated photos, organised
    by place_id then species. Also auto-populates that place's
    species_tags_<place_id>.csv with native/invasive status when iNat
    provides it.
    """

    def __init__(
        self,
        output_dir: Path = config.INAT_DIR,
        place_id: str = config.INAT_PLACE_ID,
        taxon_id: int = config.INAT_TAXON_ID,
        max_photos_per_species: int = config.INAT_MAX_PHOTOS_PER_SPECIES,
        quality_grade: str = config.INAT_QUALITY_GRADE,
    ) -> None:
        self.place_id = place_id
        self.output_dir = Path(output_dir) / self.place_id
        self.taxon_id = taxon_id
        self.max_photos = max_photos_per_species
        self.quality_grade = quality_grade
        self.species_tags_path = config.DATA_DIR / f"species_tags_{self.place_id}.csv"


    def run(self) -> None:
        """Download photos for all species found at the configured place."""
        ensure_dir(self.output_dir)
        logger.info(f"Starting iNat download — place_id: {self.place_id}")
        logger.info(f"Saving photos to: {self.output_dir}")

        observations = self._fetch_all_observations()
        logger.info(f"Fetched {len(observations)} observations.")

        species_map = self._group_by_species(observations)
        logger.info(f"Unique species found: {len(species_map)}")

        self._update_species_tags(species_map)

        for species, data in tqdm(species_map.items(), desc="Downloading species", unit="species"):
            self._download_species_photos(species, data["urls"])

        logger.info("iNat download complete.")


    def _fetch_all_observations(self) -> list[dict]:
        """
        Page through the iNat API and return all matching observations.
        Stops gracefully at iNat's 10,000 observation hard cap.
        """
        observations: list[dict] = []
        page = 1

        while True:
            # iNat counts from offset = (page-1) * per_page
            # Stop before we hit the 403 wall
            if (page - 1) * _PAGE_SIZE >= _INAT_MAX_OFFSET:
                logger.info(
                    "Reached iNat's 10,000 observation cap — stopping pagination. "
                    "To get more data, narrow your place scope or filter by taxon."
                )
                break

            params = {
                "taxon_id": self.taxon_id,
                "place_id": self.place_id,
                "quality_grade": self.quality_grade,
                "photos": "true",
                "per_page": _PAGE_SIZE,
                "page": page,
            }

            response = requests.get(_OBSERVATIONS_ENDPOINT, params=params, timeout=30)

            if response.status_code == 403:   # we have hit the 10k limit for iNat
                logger.info("We have hit the 10k download limit for iNat, stopping download...")
                break
            elif response.status_code != 200:  # catch any other http errors
                logger.warning(f"Error while downloading: {response.status_code}")
                continue

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            observations.extend(results)
            total_available = data.get("total_results", 0)
            logger.info(
                f"  Page {page}: {len(results)} observations "
                f"(total so far: {len(observations)} / {total_available} available)"
            )

            if len(observations) >= total_available:
                break

            page += 1
            time.sleep(_REQUEST_DELAY)

        return observations

    def _group_by_species(self, observations: list[dict]) -> dict[str, dict]:
        """
        Return a dict mapping species_name -> {urls: [...], status: str}.
        Caps each species at self.max_photos.

        status is derived from iNat's establishment_means field when
        available: 'introduced' -> 'invasive', 'native'/'endemic' -> 'native',
        otherwise left blank for manual review.
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
                status = ""   # leave blank if unknown — fill in manually later

            photos = obs.get("photos", [])
            species_map.setdefault(name, {"urls": [], "status": status})
            for photo in photos:
                url = photo.get("url", "")
                if not url:
                    continue
                # iNat thumbnail URLs end in /square — swap for /large
                url = url.replace("/square", "/large")
                if len(species_map[name]["urls"]) < self.max_photos:
                    species_map[name]["urls"].append(url)

        return species_map

    def _update_species_tags(self, species_map: dict[str, dict]) -> None:
        """
        Write any newly discovered species and their iNat status into
        this place's species_tags_<place_id>.csv. Existing rows are
        preserved and never overwritten.
        """
        csv_path = self.species_tags_path
        existing = load_species_tags(csv_path)

        if not csv_path.exists():
            ensure_dir(csv_path.parent)
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["species_name", "status"])
                writer.writeheader()

        new_rows = []
        for name, data in species_map.items():
            readable_name = name.replace("_", " ")
            if readable_name not in existing:
                new_rows.append({
                    "species_name": readable_name,
                    "status": data["status"],
                })

        if not new_rows:
            logger.info(f"{csv_path.name} is already up to date.")
            return

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["species_name", "status"])
            writer.writerows(new_rows)

        logger.info(f"Added {len(new_rows)} new species to {csv_path.name}.")

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
                time.sleep(0.2)   # light throttle for photo downloads
            except Exception as exc:
                logger.warning(f"Failed to download {url}: {exc}")

        logger.info(f"  {species} — downloaded {downloaded} new photos ({existing} already present)")


def main() -> None:
    downloader = INatDownloader()
    downloader.run()


if __name__ == "__main__":
    main()