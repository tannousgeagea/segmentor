"""ONNX Runtime backend for SAM v2."""

import logging

import numpy as np

from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ONNXSAMv2Backend:
    """ONNX Runtime implementation for SAM v2."""

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config

    def load(self) -> None:
        """Load SAM v2 ONNX model."""
        raise ModelLoadError(
            "ONNX SAM v2 backend requires pre-exported ONNX models. "
            "This is a placeholder for future implementation."
        )

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        raise NotImplementedError

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up resources."""
        pass