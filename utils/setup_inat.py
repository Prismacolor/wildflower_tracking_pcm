"""
setup_inat.py
Downloads iNaturalist research-grade plant observations (with photos) for a
North Texas bounding box and saves them into data/iNat_data/<species_name>/.

Usage (from project root):
    python -m scripts.setup_inat
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import requests

from scripts import config
from scripts.utils import ensure_dir, get_logger

logger = get_logger(__name__)

_OBSERVATIONS_ENDPOINT = f"{config.INAT_API_BASE}/observations"
_PAGE_SIZE = 200
_REQUEST_DELAY = 1.0   # seconds between API calls — be polite to iNat


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Download photos for all species found in the bounding box."""
        ensure_dir(self.output_dir)
        logger.info("Starting iNat download — bounding box: %s", self.bbox)

        observations = self._fetch_all_observations()
        logger.info("Fetched %d observations.", len(observations))

        species_map = self._group_by_species(observations)
        logger.info("Unique species found: %d", len(species_map))

        for species, photo_urls in species_map.items():
            self._download_species_photos(species, photo_urls)

        logger.info("iNat download complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            observations.extend(results)
            logger.info("  Page %d: %d observations (total so far: %d)", page, len(results), len(observations))

            if len(observations) >= data.get("total_results", 0):
                break

            page += 1
            time.sleep(_REQUEST_DELAY)

        return observations

    def _group_by_species(self, observations: list[dict]) -> dict[str, list[str]]:
        """
        Return a dict mapping species_name → list of photo URLs.
        Caps each species at self.max_photos.
        """
        species_map: dict[str, list[str]] = {}

        for obs in observations:
            taxon = obs.get("taxon")
            if not taxon:
                continue

            name: str = (
                taxon.get("name", "").strip().replace(" ", "_").lower()
            )
            if not name:
                continue

            photos = obs.get("photos", [])
            for photo in photos:
                url = photo.get("url", "")
                if not url:
                    continue
                # iNat thumbnail URLs end in /square — swap for /large
                url = url.replace("/square", "/large")
                species_map.setdefault(name, [])
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
                time.sleep(0.2)   # light throttle for photo downloads
            except Exception as exc:   # noqa: BLE001
                logger.warning("Failed to download %s: %s", url, exc)

        logger.info("  %s — downloaded %d new photos (%d already present)", species, downloaded, existing)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    downloader = INatDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
