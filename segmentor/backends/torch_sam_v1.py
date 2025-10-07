"""PyTorch backend for SAM v1."""

import logging
from pathlib import Path

import numpy as np
import torch

from typing import Any
from segmentor.config import SegmentorConfig
from segmentor.exceptions import BackendError, ModelLoadError

logger = logging.getLogger(__name__)


class TorchSAMv1Backend:
    """PyTorch implementation for SAM v1."""

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config
        self.model: Any = None
        self.device: torch.device | None = None

    def load(self) -> None:
        """Load SAM v1 model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            device_str = self.config.runtime.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

            # Get checkpoint
            checkpoint_path = self._get_checkpoint_path()

            # Load model
            model_type = self.config.model.encoder_variant
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)

            # Set precision
            if self.config.runtime.precision == "fp16" and self.device.type == "cuda":
                sam = sam.half()

            self.predictor = SamPredictor(sam)
            logger.info(f"Loaded SAM v1 ({model_type}) on {self.device}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM v1: {e}") from e

    def _get_checkpoint_path(self) -> str:
        """Get or download checkpoint."""
        if self.config.model.checkpoint_path:
            return self.config.model.checkpoint_path

        # Auto-download logic
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }

        variant = self.config.model.encoder_variant
        url = checkpoint_urls.get(variant)
        if not url:
            raise ModelLoadError(f"Unknown encoder variant: {variant}")

        # Download to cache
        cache_dir = Path.home() / ".cache" / "segmentor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / f"sam_v1_{variant}.pth"

        if not checkpoint_path.exists():
            logger.info(f"Downloading SAM v1 checkpoint from {url}")
            import urllib.request

            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"Saved to {checkpoint_path}")

        return str(checkpoint_path)

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        try:
            self.predictor.set_image(image)

            # SAM expects box as numpy array
            box_np = np.array(box)

            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False,
            )

            # Return first mask (we use single mask mode)
            mask = masks[0].astype(np.uint8)
            score = float(scores[0])

            return mask, score

        except Exception as e:
            raise BackendError(f"SAM v1 inference failed: {e}") from e

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts."""
        try:
            self.predictor.set_image(image)

            # Convert points to numpy arrays
            point_coords = np.array([[x, y] for x, y, _ in points])
            point_labels = np.array([label for _, _, label in points])

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

            mask = masks[0].astype(np.uint8)
            score = float(scores[0])

            return mask, score

        except Exception as e:
            raise BackendError(f"SAM v1 inference failed: {e}") from e

    def close(self) -> None:
        """Clean up resources."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()