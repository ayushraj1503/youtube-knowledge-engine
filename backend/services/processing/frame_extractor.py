# backend/services/processing/frame_extractor.py
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import yt_dlp

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class FrameExtractor:
    def __init__(self):
        self.fps = settings.FRAME_EXTRACTION_FPS
        self.max_frames = settings.MAX_FRAMES_PER_VIDEO
        self._available = False
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning("ffmpeg_not_found - frame extraction disabled")
                self._available = False
            else:
                self._available = True
                logger.info("ffmpeg_ready")
        except FileNotFoundError:
            logger.warning("ffmpeg_not_installed - frame extraction disabled")
            self._available = False
        except Exception as e:
            logger.warning("ffmpeg_check_failed", error=str(e))
            self._available = False

    def extract(self, video_id: str, video_url: str) -> List[Tuple[float, Path]]:
        if not self._available:
            return []
        try:
            stream_url = self._get_stream_url(video_url)
            if not stream_url:
                return []

            output_dir = Path(tempfile.gettempdir()) / "yk_frames" / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_pattern = str(output_dir / "frame_%05d.jpg")

            cmd = [
                "ffmpeg", "-i", stream_url,
                "-vf", f"fps={self.fps}",
                "-vframes", str(self.max_frames),
                "-q:v", "3", "-y",
                output_pattern,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return []

            frames: List[Tuple[float, Path]] = []
            frame_interval = 1.0 / self.fps
            for frame_file in sorted(output_dir.glob("frame_*.jpg")):
                idx = int(frame_file.stem.split("_")[1]) - 1
                timestamp = idx * frame_interval
                frames.append((timestamp, frame_file))
            return frames
        except Exception as e:
            logger.error("frame_extraction_error", video_id=video_id, error=str(e))
            return []

    def _get_stream_url(self, video_url: str) -> Optional[str]:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/bestvideo",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info.get("url")
        except Exception as e:
            logger.error("stream_url_error", url=video_url, error=str(e))
            return None

    def cleanup(self, video_id: str) -> None:
        import shutil
        frame_dir = Path(tempfile.gettempdir()) / "yk_frames" / video_id
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)