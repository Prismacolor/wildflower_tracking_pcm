"""
test_extractor.py
Tests for scripts/extractor.py — VideoExtractor.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.extractor import VideoExtractor


def _make_extractor(tmp_path: Path) -> tuple[VideoExtractor, Path, Path]:
    videos_dir = tmp_path / "videos"
    stills_dir = tmp_path / "stills"
    videos_dir.mkdir()
    extractor = VideoExtractor(
        videos_dir=videos_dir,
        stills_base_dir=stills_dir,
        fps=1,
    )
    return extractor, videos_dir, stills_dir


def test_run_creates_output_dir(tmp_path):
    extractor, videos_dir, stills_dir = _make_extractor(tmp_path)
    batch = videos_dir / "batch_20260601"
    batch.mkdir()
    # Place a dummy mp4 so glob finds something
    (batch / "test.mp4").write_bytes(b"")

    with patch.object(extractor, "_extract_frames", return_value=5) as mock_extract:
        out = extractor.run()
    assert out.is_dir()
    assert "prairiecreek_" in out.name
    mock_extract.assert_called_once()


def test_run_warns_on_no_videos(tmp_path, caplog):
    extractor, videos_dir, stills_dir = _make_extractor(tmp_path)
    batch = videos_dir / "batch_20260601"
    batch.mkdir()   # empty — no video files

    import logging
    with caplog.at_level(logging.WARNING, logger="scripts.extractor"):
        out = extractor.run()

    assert out.is_dir()
    assert "No video files" in caplog.text


def test_extract_frames_bad_file(tmp_path):
    extractor, _, _ = _make_extractor(tmp_path)
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # OpenCV will fail to open it; should return 0 gracefully
    result = extractor._extract_frames(fake_video, out_dir)
    assert result == 0


def test_frame_interval_calculation(tmp_path):
    """Verify the frame interval math is correct for different fps settings."""
    extractor, _, _ = _make_extractor(tmp_path)
    extractor.fps = 1

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0   # native 30 fps video

    # Simulate 60 frames then stop
    frame_count = [0]
    def read_side_effect():
        if frame_count[0] < 60:
            frame_count[0] += 1
            return True, MagicMock()
        return False, None
    mock_cap.read.side_effect = read_side_effect

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imwrite") as mock_write:
        saved = extractor._extract_frames(tmp_path / "dummy.mp4", out_dir)

    # 60 frames at 30 fps native, extracting every 30th frame → 2 frames
    assert saved == 2
