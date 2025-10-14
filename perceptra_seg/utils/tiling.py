"""Tiling utilities for large images."""

import numpy as np


def tile_image(
    image: np.ndarray, tile_size: int = 1024, stride: int = 512
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Tile large image into overlapping patches.

    Args:
        image: Input image (HxWxC)
        tile_size: Size of each tile
        stride: Stride between tiles

    Returns:
        List of (tile, bbox) tuples where bbox is (x1, y1, x2, y2)
    """
    h, w = image.shape[:2]
    tiles = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)

            tile = image[y1:y2, x1:x2]
            tiles.append((tile, (x1, y1, x2, y2)))

    return tiles


def merge_tile_masks(
    tile_masks: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    image_shape: tuple[int, int],
    blend_mode: str = "average",
) -> np.ndarray:
    """Merge tiled masks back into full image.

    Args:
        tile_masks: List of (mask, bbox) tuples
        image_shape: Target image shape (H, W)
        blend_mode: 'average' or 'linear' blending

    Returns:
        Merged mask
    """
    h, w = image_shape
    merged = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for mask, (x1, y1, x2, y2) in tile_masks:
        merged[y1:y2, x1:x2] += mask
        counts[y1:y2, x1:x2] += 1

    # Average overlapping regions
    counts = np.maximum(counts, 1)
    merged /= counts

    return (merged > 0.5).astype(np.uint8)