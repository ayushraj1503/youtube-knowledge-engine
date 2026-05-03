# backend/services/processing/visual_captioner.py
"""
Visual Captioner — wraps BLIP (Bootstrapping Language-Image Pre-training)
for local, offline image captioning.

Why BLIP over GPT-4V?
  - No external API calls = no cost per frame, no rate limits
  - Data stays local (important for enterprise use cases)
  - Fast enough for our FPS-limited extraction (0.5 FPS)
  - BLIP-base is ~450MB — acceptable for a production container

Model loading strategy:
  - Lazy initialization: model loads only when first caption is requested
  - Singleton pattern via module-level instance reuse in ProcessingPipeline
  - Runs on GPU if available, falls back to CPU gracefully

BLIP-2 alternative:
  - Better quality but ~5GB VRAM requirement
  - Switch by changing BLIP_MODEL=Salesforce/blip2-opt-2.7b in .env
"""

from pathlib import Path
from typing import Optional

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VisualCaptioner:
    """
    Lazy-loads BLIP model on first use.
    Thread-safe for use in ThreadPoolExecutor (model inference is GIL-held).
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

    def _load_model(self) -> None:
        """Load BLIP model and processor from HuggingFace Hub (cached locally)."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "blip_loading",
                model=settings.BLIP_MODEL,
                device=self._device,
            )

            self._processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
            self._model = BlipForConditionalGeneration.from_pretrained(
                settings.BLIP_MODEL
            ).to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info("blip_loaded", device=self._device)

        except ImportError:
            logger.error(
                "blip_import_error — install transformers and torch"
            )
            self._loaded = False
        except Exception as e:
            logger.error("blip_load_failed", error=str(e))
            self._loaded = False

    def caption(self, frame_path: Path) -> Optional[str]:
        """
        Generate a short caption for a single frame image.
        Returns None if captioning fails or model not available.
        """
        if not self._loaded:
            self._load_model()

        if not self._loaded or not self._model:
            return None

        try:
            from PIL import Image

            img = Image.open(frame_path).convert("RGB")
            inputs = self._processor(img, return_tensors="pt").to(self._device)

            with __import__("torch").no_grad():
                output = self._model.generate(**inputs, max_new_tokens=50)

            caption = self._processor.decode(output[0], skip_special_tokens=True)
            return caption.strip()

        except Exception as e:
            logger.error("caption_failed", frame=str(frame_path), error=str(e))
            return None

    def batch_caption(self, frame_paths: list) -> list:
        """
        Caption multiple frames. Returns list of Optional[str].
        Useful when processing many frames at once for GPU efficiency.
        """
        return [self.caption(p) for p in frame_paths]
