import numpy as np
import pytest

from perceptra_seg.config import SegmentorConfig
from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend


def test_torch_sam_v1_backend_init() -> None:
    """Test TorchSAMv1Backend initialization."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"
    
    backend = TorchSAMv1Backend(config)
    assert backend.config == config
    assert backend.device is None  # Not loaded yet


def test_torch_sam_v1_checkpoint_path_resolution() -> None:
    """Test checkpoint path resolution."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"
    config.model.encoder_variant = "vit_h"
    
    backend = TorchSAMv1Backend(config)
    
    # Should return URL or cached path
    path = backend._get_checkpoint_path()
    assert path is not None
    assert isinstance(path, str)


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not installed"
)
def test_backend_inference(config: SegmentorConfig) -> None:
    """Test backend inference (if PyTorch available)."""
    from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend
    
    backend = TorchSAMv1Backend(config)
    backend.load()
    
    # Create dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (100, 100, 300, 300)
    
    mask, score = backend.infer_from_box(image, box)
    
    assert mask.shape == (480, 640)
    assert 0 <= score <= 1
    
    backend.close()