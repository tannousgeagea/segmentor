"""Tests for core Segmentor functionality."""

import numpy as np
import pytest

from perceptra_seg import Segmentor, SegmentorConfig
from perceptra_seg.exceptions import InvalidPromptError


def test_segmentor_initialization(config: SegmentorConfig) -> None:
    """Test Segmentor initialization."""
    segmentor = Segmentor(config=config)
    assert segmentor.backend is not None
    assert segmentor.config == config
    segmentor.close()


def test_segment_from_box(segmentor: Segmentor, sample_image: np.ndarray, sample_box: tuple) -> None:
    """Test segmentation from bounding box."""
    result = segmentor.segment_from_box(
        sample_image,
        sample_box,
        output_formats=["numpy", "rle"],
    )

    assert result.mask is not None
    assert result.mask.shape == sample_image.shape[:2]
    assert result.rle is not None
    assert result.score > 0
    assert result.area > 0
    assert result.bbox is not None


def test_segment_from_points(segmentor: Segmentor, sample_image: np.ndarray) -> None:
    """Test segmentation from points."""
    points = [(250, 200, 1)]  # Positive point in center of square

    result = segmentor.segment_from_points(
        sample_image,
        points,
        output_formats=["numpy"],
    )

    assert result.mask is not None
    assert result.score > 0


def test_invalid_box(segmentor: Segmentor, sample_image: np.ndarray) -> None:
    """Test invalid bounding box."""
    invalid_box = (0, 0, 10000, 10000)  # Out of bounds

    with pytest.raises(InvalidPromptError):
        segmentor.segment_from_box(sample_image, invalid_box)


def test_output_formats(segmentor: Segmentor, sample_image: np.ndarray, sample_box: tuple) -> None:
    """Test different output formats."""
    result = segmentor.segment_from_box(
        sample_image,
        sample_box,
        output_formats=["numpy", "rle", "polygons", "png"],
    )

    assert result.mask is not None
    assert result.rle is not None
    assert result.polygons is not None
    assert result.png_bytes is not None


def test_context_manager() -> None:
    """Test using Segmentor as context manager."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"

    with Segmentor(config=config) as seg:
        assert seg.backend is not None

    # Backend should be closed after exiting context
    assert seg.backend is None