"""Segmentor: Production-grade segmentation with SAM models."""

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

__version__ = "0.1.0"
__all__ = [
    "Segmentor",
    "SegmentorConfig",
    "SegmentationResult",
    "SegmentorError",
    "ModelLoadError",
    "InvalidPromptError",
    "BackendError",
    "ConfigError",
]