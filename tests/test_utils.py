"""Tests for utility functions."""

import numpy as np

from segmentor.utils.image_io import load_image
from segmentor.utils.mask_utils import (
    apply_morphology,
    mask_to_png_bytes,
    mask_to_polygons,
    mask_to_rle,
    remove_small_components,
)


def test_mask_to_rle() -> None:
    """Test RLE encoding."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    rle = mask_to_rle(mask)

    assert "size" in rle
    assert "counts" in rle
    assert rle["size"] == [10, 10]


def test_mask_to_polygons() -> None:
    """Test polygon extraction."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    polygons = mask_to_polygons(mask)

    assert len(polygons) > 0
    assert all(len(poly) >= 3 for poly in polygons)


def test_mask_to_png() -> None:
    """Test PNG encoding."""
    mask = np.ones((50, 50), dtype=np.uint8)

    png_bytes = mask_to_png_bytes(mask)

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    assert png_bytes.startswith(b"\x89PNG")


def test_remove_small_components() -> None:
    """Test small component removal."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # Small component
    mask[30:80, 30:80] = 1  # Large component

    filtered = remove_small_components(mask, min_area=200)

    # Small component should be removed
    assert np.sum(filtered[10:20, 10:20]) == 0
    # Large component should remain
    assert np.sum(filtered[30:80, 30:80]) > 0


def test_load_image_numpy() -> None:
    """Test loading numpy array."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    loaded = load_image(img)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)