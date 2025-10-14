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

    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate masks from multiple bounding boxes efficiently.
        
        Args:
            image: RGB image as numpy array (HxWx3)
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Tuple of (list of binary_masks, list of confidence_scores)
        """
        ...
    
    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate masks from multiple point prompts efficiently.
        
        Args:
            image: RGB image as numpy array (HxWx3)
            points_list: List of point prompt lists
            
        Returns:
            Tuple of (list of binary_masks, list of confidence_scores)
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...