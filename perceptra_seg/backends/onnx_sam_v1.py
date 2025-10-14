"""ONNX Runtime backend for SAM v1."""

import logging

import numpy as np

from typing import Any
from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, ModelLoadError

logger = logging.getLogger(__name__)


class ONNXSAMv1Backend:
    """ONNX Runtime implementation for SAM v1."""

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config
        self.session: Any = None

    def load(self) -> None:
        """Load SAM v1 ONNX model."""
        try:
            import onnxruntime as ort

            # Note: ONNX models require separate encoder and decoder
            # This is a simplified implementation
            logger.warning("ONNX SAM v1 backend is a stub - requires ONNX model files")

            providers = ["CPUExecutionProvider"]
            if self.config.runtime.device.startswith("cuda"):
                providers.insert(0, "CUDAExecutionProvider")

            # In production, load actual ONNX files
            # self.encoder_session = ort.InferenceSession("encoder.onnx", providers=providers)
            # self.decoder_session = ort.InferenceSession("decoder.onnx", providers=providers)

            raise ModelLoadError(
                "ONNX backend requires pre-exported ONNX models. "
                "Please export SAM to ONNX format first."
            )

        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX SAM v1: {e}") from e

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        raise BackendError("ONNX backend not fully implemented")

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts."""
        raise BackendError("ONNX backend not fully implemented")

    def close(self) -> None:
        """Clean up resources."""
        pass