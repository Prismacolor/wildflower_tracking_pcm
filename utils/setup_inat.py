"""
Downloads iNaturalist research-grade angiosperm (flowering plant) observations
for multiples spaces, and saves the photos into
data/iNat_data/<place_id>/<species_name>/.

Scoping downloads by place_id (rather than hard-coding one location) means
this project can be reused for other restoration sites — just change
INAT_PLACE_IDs in config.py and re-run the download.
"""

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
_REQUEST_DELAY = 1.5       # seconds between API calls — be polite to iNat


class INatDownloader:
    """
    Fetches plant observations via the iNaturalist v1 API for one place
    and downloads the associated photos, organised by species.

    INatDownloader is scoped to a single place_id. The top-level
    download_all_places() function loops over all configured places.
    """

    def __init__(
            self,
            place_id: str,
            output_dir: Path = config.INAT_DIR,
            taxon_id: int = config.INAT_TAXON_ID,
            max_photos_per_species: int = config.INAT_MAX_PHOTOS_PER_SPECIES,
            quality_grade: str = config.INAT_QUALITY_GRADE,
    ) -> None:
        self.place_id = place_id
        self.output_dir = Path(output_dir) / self.place_id
        self.taxon_id = taxon_id
        self.max_photos = max_photos_per_species
        self.quality_grade = quality_grade

    def run(self) -> None:
        """Download photos for all species found at this place."""
        ensure_dir(self.output_dir)
        logger.info(f"Starting iNat download — place_id: {self.place_id}")
        logger.info(f"Saving photos to: {self.output_dir}")

        observations = self._fetch_all_observations()
        logger.info(f"Fetched {len(observations)} observations.")

        species_map = self._group_by_species(observations)
        logger.info(f"Unique species found: {len(species_map)}")

        for species, urls in tqdm(
                species_map.items(),
                desc=f"Downloading place {self.place_id}",
                unit="species",
        ):
            self._download_species_photos(species, urls)

        logger.info(f"Download complete for place_id: {self.place_id}")

    def _fetch_all_observations(self) -> list[dict]:
        """
        Page through the iNat API and return all matching observations.
        Stops gracefully at iNat's 10,000 observation hard cap.
        """
        observations: list[dict] = []
        page = 1

        while True:
            if (page - 1) * _PAGE_SIZE >= _INAT_MAX_OFFSET:
                logger.info("Reached iNat's 10,000 observation cap — stopping pagination.")
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

            if response.status_code == 403:
                logger.info("We have hit the 10k download limit for iNat, stopping download...")
                break
            elif response.status_code != 200:
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

    def _group_by_species(self, observations: list[dict]) -> dict[str, list[str]]:
        """
        Return a dict mapping species_name -> list of photo URLs.
        Caps each species at self.max_photos.
        """
        species_map: dict[str, list[str]] = {}

        for obs in observations:
            taxon = obs.get("taxon")
            if not taxon:
                continue

            name: str = taxon.get("name", "").strip().replace(" ", "_").lower()
            if not name:
                continue

            photos = obs.get("photos", [])
            species_map.setdefault(name, [])
            for photo in photos:
                url = photo.get("url", "")
                if not url:
                    continue
                url = url.replace("/square", "/large")
                if len(species_map[name]) < self.max_photos:
                    species_map[name].append(url)

        return species_map

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

        logger.info(
            f"  {species} — downloaded {downloaded} new photos "
            f"({existing} already present)"
        )

def download_all_places(place_ids: list[str] = config.INAT_PLACE_IDS) -> None:
    """
    Download iNat photos for every place in place_ids, one at a time.
    Already-downloaded photos are skipped, so this is safe to re-run.
    """
    logger.info(f"Downloading data for {len(place_ids)} place(s): {place_ids}")
    for idx, place_id in enumerate(place_ids, start=1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Place {idx} of {len(place_ids)}: {place_id}")
        logger.info(f"{'=' * 50}")
        INatDownloader(place_id=place_id).run()
    logger.info("All places downloaded.")


def main() -> None:
    download_all_places()


if __name__ == "__main__":
    main()
