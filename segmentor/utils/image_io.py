"""Image I/O utilities."""

import io
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from segmentor.exceptions import ImageLoadError


def load_image(
    image: np.ndarray | Image.Image | bytes | str | Path, timeout: int = 10
) -> np.ndarray:
    """Load image from various sources.

    Args:
        image: Image as numpy array, PIL Image, bytes, file path, or URL
        timeout: Timeout for URL requests in seconds

    Returns:
        RGB image as numpy array (HxWx3)
    """
    try:
        # Already numpy array
        if isinstance(image, np.ndarray):
            return _ensure_rgb(image)

        # PIL Image
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        # Bytes
        if isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image))
            return np.array(pil_img.convert("RGB"))

        # String path or URL
        if isinstance(image, (str, Path)):
            image_str = str(image)

            # URL
            if image_str.startswith(("http://", "https://")):
                response = requests.get(image_str, timeout=timeout)
                response.raise_for_status()
                pil_img = Image.open(io.BytesIO(response.content))
                return np.array(pil_img.convert("RGB"))

            # File path
            path = Path(image_str)
            if not path.exists():
                raise ImageLoadError(f"Image file not found: {path}")

            pil_img = Image.open(path)
            return np.array(pil_img.convert("RGB"))

        raise ImageLoadError(f"Unsupported image type: {type(image)}")

    except Exception as e:
        raise ImageLoadError(f"Failed to load image: {e}") from e


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB format."""
    if image.ndim == 2:
        # Grayscale to RGB
        return np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA to RGB
        return image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] == 3:
        return image
    else:
        raise ImageLoadError(f"Unexpected image shape: {image.shape}")