"""Base backend protocol."""

from typing import Protocol, runtime_checkable

import numpy as np

from segmentor.config import SegmentorConfig


@runtime_checkable
class BaseSAMBackend(Protocol):
    """Protocol for SAM backend implementations."""

    def __init__(self, config: SegmentorConfig) -> None:
        """Initialize backend with configuration."""
        ...

    def load(self) -> None:
        """Load model weights and initialize."""
        ...

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box.

        Args:
            image: RGB image as numpy array (HxWx3)
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        ...

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts.

        Args:
            image: RGB image as numpy array (HxWx3)
            points: List of (x, y, label) tuples

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...