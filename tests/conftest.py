"""Pytest fixtures and configuration."""

import numpy as np
import pytest

from segmentor import Segmentor, SegmentorConfig


@pytest.fixture
def config() -> SegmentorConfig:
    """Create test configuration."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"
    config.runtime.backend = "torch"
    config.model.name = "sam_v1"
    return config


@pytest.fixture
def segmentor(config: SegmentorConfig) -> Segmentor:
    """Create Segmentor instance."""
    seg = Segmentor(config=config)
    yield seg
    seg.close()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create sample test image."""
    # Simple image with white square on black background
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:300, 150:350] = 255  # White square
    return image


@pytest.fixture
def sample_box() -> tuple[int, int, int, int]:
    """Sample bounding box covering the white square."""
    return (150, 100, 350, 300)