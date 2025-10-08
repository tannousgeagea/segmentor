"""Core Segmentor class."""

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from segmentor.backends.base import BaseSAMBackend
from segmentor.config import SegmentorConfig
from segmentor.exceptions import BackendError, InvalidPromptError, ModelLoadError
from segmentor.models import SegmentationResult
from segmentor.utils.cache import EmbeddingCache
from segmentor.utils.image_io import load_image
from segmentor.utils.mask_utils import (
    apply_morphology,
    mask_to_png_bytes,
    mask_to_polygons,
    mask_to_rle,
    remove_small_components,
)

logger = logging.getLogger(__name__)


class Segmentor:
    """Main segmentation interface supporting SAM v1 and v2.

    Args:
        config: SegmentorConfig instance or None to use defaults
        **kwargs: Override specific config values
    """

    def __init__(
        self,
        config: SegmentorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize configuration
        if config is None:
            config = SegmentorConfig()

        # Apply kwargs overrides with smart mapping
        # Common shortcuts for user convenience
        shortcuts = {
            "model": ("model", "name"),
            "backend": ("runtime", "backend"),
            "device": ("runtime", "device"),
            "precision": ("runtime", "precision"),
            "batch_size": ("runtime", "batch_size"),
            "checkpoint_path": ("model", "checkpoint_path"),
            "encoder_variant": ("model", "encoder_variant"),
        }

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if "." in key:
                section, field = key.split(".", 1)
                if hasattr(config, section):
                    setattr(getattr(config, section), field, value)
            elif key in shortcuts:
                # Use shortcut mapping
                section_name, field_name = shortcuts[key]
                section = getattr(config, section_name)
                setattr(section, field_name, value)
            else:
                # Try to find which config section this key belongs to
                applied = False
                for section_name in ["model", "runtime", "tiling", "outputs", "thresholds", 
                                     "postprocess", "cache", "server", "logging"]:
                    section = getattr(config, section_name, None)
                    if section and hasattr(section, key):
                        setattr(section, key, value)
                        applied = True
                        break
                
                if not applied:
                    logger.warning(f"Unknown config parameter: {key}")

        self.config = config
        self.backend: BaseSAMBackend | None = None
        self.cache: EmbeddingCache | None = None

        if self.config.cache.enabled:
            self.cache = EmbeddingCache(max_size=self.config.cache.max_items)

        self._load_backend()

    def _load_backend(self) -> None:
        """Load the appropriate backend based on configuration."""
        backend_key = f"{self.config.runtime.backend}_{self.config.model.name}"

        try:
            if backend_key == "torch_sam_v1":
                from segmentor.backends.torch_sam_v1 import TorchSAMv1Backend

                self.backend = TorchSAMv1Backend(self.config)
            elif backend_key == "torch_sam_v2":
                from segmentor.backends.torch_sam_v2 import TorchSAMv2Backend

                self.backend = TorchSAMv2Backend(self.config)
            elif backend_key == "onnx_sam_v1":
                from segmentor.backends.onnx_sam_v1 import ONNXSAMv1Backend

                self.backend = ONNXSAMv1Backend(self.config)
            elif backend_key == "onnx_sam_v2":
                from segmentor.backends.onnx_sam_v2 import ONNXSAMv2Backend

                self.backend = ONNXSAMv2Backend(self.config)
            else:
                raise BackendError(f"Unknown backend: {backend_key}")

            self.backend.load()
            logger.info(f"Loaded backend: {backend_key}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load backend {backend_key}: {e}") from e

    def set_backend(self, backend_name: str) -> None:
        """Switch to a different backend.

        Args:
            backend_name: Name of backend ('torch' or 'onnx')
        """
        if self.backend is not None:
            self.backend.close()

        self.config.runtime.backend = backend_name  # type: ignore
        self._load_backend()

    def segment_from_box(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        box: tuple[int, int, int, int],
        *,
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> SegmentationResult:
        """Segment object from bounding box.

        Args:
            image: Input image (numpy array, PIL Image, bytes, or path)
            box: Bounding box as (x1, y1, x2, y2) in absolute pixels
            output_formats: List of output formats ['rle', 'png', 'polygons', 'numpy']
            return_overlay: Whether to return overlay visualization

        Returns:
            SegmentationResult with requested outputs
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Load and validate image
        img_array = load_image(image)
        self._validate_box(box, img_array.shape)

        if output_formats is None:
            output_formats = self.config.outputs.default_formats

        # Perform segmentation
        if self.backend is None:
            raise BackendError("Backend not loaded")

        mask, score = self.backend.infer_from_box(img_array, box)

        # Postprocess
        mask = self._postprocess_mask(mask, img_array.shape)

        # Generate outputs
        result = self._create_result(
            mask=mask,
            score=score,
            output_formats=output_formats,
            latency_ms=(time.time() - start_time) * 1000,
            request_id=request_id,
        )

        return result

    def segment_from_points(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        points: list[tuple[int, int, int]],
        *,
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> SegmentationResult:
        """Segment object from point prompts.

        Args:
            image: Input image
            points: List of (x, y, label) where label is 1 (positive) or 0 (negative)
            output_formats: List of output formats
            return_overlay: Whether to return overlay

        Returns:
            SegmentationResult with requested outputs
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        img_array = load_image(image)
        self._validate_points(points, img_array.shape)

        if output_formats is None:
            output_formats = self.config.outputs.default_formats

        if self.backend is None:
            raise BackendError("Backend not loaded")

        mask, score = self.backend.infer_from_points(img_array, points)
        mask = self._postprocess_mask(mask, img_array.shape)

        result = self._create_result(
            mask=mask,
            score=score,
            output_formats=output_formats,
            latency_ms=(time.time() - start_time) * 1000,
            request_id=request_id,
        )

        return result

    def segment(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        boxes: list[tuple[int, int, int, int]] | None = None,
        points: list[tuple[int, int, int]] | None = None,
        *,
        strategy: Literal["merge", "largest", "all"] = "largest",
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> list[SegmentationResult]:
        """Segment with multiple prompts.

        Args:
            image: Input image
            boxes: List of bounding boxes
            points: List of point prompts
            strategy: How to handle multiple masks ('merge', 'largest', 'all')
            output_formats: List of output formats
            return_overlay: Whether to return overlay

        Returns:
            List of SegmentationResult instances
        """
        if boxes is None and points is None:
            raise InvalidPromptError("Must provide either boxes or points")

        results = []

        if boxes:
            for box in boxes:
                result = self.segment_from_box(
                    image, box, output_formats=output_formats, return_overlay=return_overlay
                )
                results.append(result)

        if points:
            result = self.segment_from_points(
                image, points, output_formats=output_formats, return_overlay=return_overlay
            )
            results.append(result)

        # Apply strategy
        if strategy == "largest" and len(results) > 1:
            largest = max(results, key=lambda r: r.area)
            return [largest]
        elif strategy == "merge" and len(results) > 1:
            # Merge masks
            merged_mask = np.zeros_like(results[0].mask) if results[0].mask is not None else None
            if merged_mask is not None:
                for r in results:
                    if r.mask is not None:
                        merged_mask = np.logical_or(merged_mask, r.mask)

                # Create merged result
                merged_result = self._create_result(
                    mask=merged_mask.astype(np.uint8),
                    score=np.mean([r.score for r in results]),
                    output_formats=output_formats or self.config.outputs.default_formats,
                    latency_ms=sum(r.latency_ms for r in results),
                    request_id=str(uuid.uuid4()),
                )
                return [merged_result]

        return results

    def warmup(self, image_size: tuple[int, int] | None = None) -> None:
        """Warm up the model with a dummy forward pass.

        Args:
            image_size: Optional image size (height, width)
        """
        if image_size is None:
            image_size = (1024, 1024)

        dummy_image = np.zeros((*image_size, 3), dtype=np.uint8)
        dummy_box = (100, 100, 200, 200)

        logger.info("Warming up model...")
        try:
            self.segment_from_box(dummy_image, dummy_box)
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def close(self) -> None:
        """Clean up resources."""
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def _validate_box(
        self, box: tuple[int, int, int, int], image_shape: tuple[int, int, int]
    ) -> None:
        """Validate bounding box coordinates."""
        x1, y1, x2, y2 = box
        h, w = image_shape[:2]

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise InvalidPromptError(f"Box {box} out of image bounds ({w}x{h})")

        if x2 <= x1 or y2 <= y1:
            raise InvalidPromptError(f"Invalid box dimensions: {box}")

    def _validate_points(
        self, points: list[tuple[int, int, int]], image_shape: tuple[int, int, int]
    ) -> None:
        """Validate point coordinates."""
        h, w = image_shape[:2]
        for x, y, label in points:
            if x < 0 or y < 0 or x >= w or y >= h:
                raise InvalidPromptError(f"Point ({x}, {y}) out of image bounds ({w}x{h})")
            if label not in (0, 1):
                raise InvalidPromptError(f"Point label must be 0 or 1, got {label}")

    def _postprocess_mask(self, mask: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
        """Apply postprocessing to mask."""
        if self.config.postprocess.remove_small_components:
            min_area = int(self.config.outputs.min_area_ratio * mask.size)
            mask = remove_small_components(mask, min_area=min_area)

        if self.config.postprocess.morphological_closing:
            kernel_size = self.config.postprocess.closing_kernel_size
            mask = apply_morphology(mask, operation="closing", kernel_size=kernel_size)

        return mask

    def _create_result(
        self,
        mask: np.ndarray,
        score: float,
        output_formats: list[str],
        latency_ms: float,
        request_id: str,
    ) -> SegmentationResult:
        """Create SegmentationResult from mask and metadata."""
        result = SegmentationResult(
            score=score,
            area=int(np.sum(mask)),
            latency_ms=latency_ms,
            model_info={
                "name": self.config.model.name,
                "backend": self.config.runtime.backend,
                "device": self.config.runtime.device,
            },
            request_id=request_id,
        )

        # Compute bbox
        if np.any(mask):
            coords = np.argwhere(mask)
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            result.bbox = (int(x1), int(y1), int(x2), int(y2))

        # Generate requested outputs
        for fmt in output_formats:
            if fmt == "numpy":
                result.mask = mask
            elif fmt == "rle":
                result.rle = mask_to_rle(mask)
            elif fmt == "polygons":
                result.polygons = mask_to_polygons(mask)
            elif fmt == "png":
                result.png_bytes = mask_to_png_bytes(mask)

        return result

    def __enter__(self) -> "Segmentor":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()