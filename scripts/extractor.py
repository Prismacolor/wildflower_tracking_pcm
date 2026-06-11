"""
Converts video files into still frames using OpenCV.
"""

import cv2
from pathlib import Path

from scripts import config
from utils.utils import (
    datestamp,
    ensure_dir,
    get_logger,
    latest_subdirectory,
)


logger = get_logger(__name__)


class VideoExtractor:
    """Extract still frames from every video in a source directory."""
    def __init__(
        self,
        videos_dir: Path = config.VIDEOS_DIR,
        stills_base_dir: Path = config.STILLS_DIR,
        fps: int = config.FRAMES_PER_SECOND,
        image_format: str = config.STILLS_FORMAT,
    ) -> None:
        self.videos_dir = Path(videos_dir)
        self.stills_base_dir = Path(stills_base_dir)
        self.fps = fps
        self.image_format = image_format


    def run(self) -> Path:
        """
        Find the most recent video batch directory, extract frames, and
        save them.  Returns the output directory path.
        """
        source_dir = latest_subdirectory(self.videos_dir)
        logger.info("Using video source directory: %s", source_dir)

        output_dir = self.stills_base_dir / f"prairiecreek_{datestamp()}"
        ensure_dir(output_dir)
        logger.info("Saving stills to: %s", output_dir)

        video_files = list(source_dir.rglob("*"))
        video_files = [
            f for f in video_files
            if f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        ]

        if not video_files:
            logger.warning("No video files found in %s", source_dir)
            return output_dir

        total_frames = 0
        for video_path in video_files:
            saved = self._extract_frames(video_path, output_dir)
            total_frames += saved
            logger.info("  %s → %d frames", video_path.name, saved)

        logger.info("Extraction complete. Total frames saved: %d", total_frames)
        return output_dir


    def _extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """
        Extract one frame per self.fps seconds from *video_path* and write
        them as images into *output_dir*.  Returns the number of frames saved.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return 0

        native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval: int = max(1, round(native_fps / self.fps))
        stem = video_path.stem

        saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                filename = output_dir / f"{stem}_f{frame_idx:06d}.{self.image_format}"
                cv2.imwrite(str(filename), frame)
                saved += 1
            frame_idx += 1

        cap.release()
        return saved


def main() -> None:
    extractor = VideoExtractor()
    out = extractor.run()
    print(f"Stills saved to: {out}")


if __name__ == "__main__":
    main()
