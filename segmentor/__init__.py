"""Segmentor: Production-grade segmentation with SAM models.

Basic usage:
    >>> from segmentor import Segmentor
    >>> seg = Segmentor(backend="torch", model="sam_v1")
    >>> result = seg.segment_from_box(image, box=(100, 100, 400, 400))
    >>> print(f"Score: {result.score}, Area: {result.area}")

API usage:
    >>> from segmentor.config import SegmentorConfig
    >>> config = SegmentorConfig.from_yaml("config.yaml")
    >>> seg = Segmentor(config=config)
"""

from segmentor.__version__ import __version__, __author__, __email__
from segmentor.config import SegmentorConfig
from segmentor.core import Segmentor
from segmentor.exceptions import (
    BackendError,
    ConfigError,
    InvalidPromptError,
    ModelLoadError,
    SegmentorError,
)
from segmentor.models import SegmentationResult

__all__ = [
    # Main classes
    "Segmentor",
    "SegmentorConfig",
    "SegmentationResult",
    # Exceptions
    "SegmentorError",
    "ModelLoadError",
    "InvalidPromptError",
    "BackendError",
    "ConfigError",
    # Version info
    "__version__",
    "__author__",
    "__email__",
]

# Package metadata
__title__ = "segmentor"
__description__ = "Production-grade segmentation tool with SAM v1/v2 support"
__url__ = "https://github.com/tannousgeagea/segmentor"
__license__ = "Apache-2.0"