"""
test_inat.py
Tests for scripts/setup_inat.py — INatDownloader (non-network logic only).
We test the data-transformation methods without hitting the live API.
"""

from __future__ import annotations

import pytest

from scripts.setup_inat import INatDownloader


@pytest.fixture()
def downloader(tmp_path):
    return INatDownloader(
        output_dir=tmp_path / "iNat_data",
        max_photos_per_species=5,
    )


class TestGroupBySpecies:

    def _make_obs(self, name: str, photo_urls: list[str]) -> dict:
        return {
            "taxon": {"name": name},
            "photos": [{"url": u} for u in photo_urls],
        }

    def test_groups_by_species(self, downloader):
        obs = [
            self._make_obs("Echinacea purpurea", ["http://a.com/square.jpg"]),
            self._make_obs("Echinacea purpurea", ["http://b.com/square.jpg"]),
            self._make_obs("Lythrum salicaria",  ["http://c.com/square.jpg"]),
        ]
        result = downloader._group_by_species(obs)
        assert "echinacea_purpurea" in result
        assert len(result["echinacea_purpurea"]) == 2
        assert "lythrum_salicaria" in result

    def test_replaces_square_with_large(self, downloader):
        obs = [self._make_obs("Solidago canadensis", ["http://x.com/photos/1/square.jpg"])]
        result = downloader._group_by_species(obs)
        assert result["solidago_canadensis"][0].endswith("/large.jpg")

    def test_caps_at_max_photos(self, downloader):
        urls = [f"http://x.com/{i}/square.jpg" for i in range(20)]
        obs = [self._make_obs("Rosa multiflora", urls)]
        result = downloader._group_by_species(obs)
        assert len(result["rosa_multiflora"]) == downloader.max_photos

    def test_skips_obs_without_taxon(self, downloader):
        obs = [{"taxon": None, "photos": [{"url": "http://x.com/square.jpg"}]}]
        result = downloader._group_by_species(obs)
        assert result == {}

    def test_skips_obs_with_empty_name(self, downloader):
        obs = [{"taxon": {"name": "  "}, "photos": [{"url": "http://x.com/square.jpg"}]}]
        result = downloader._group_by_species(obs)
        assert result == {}
