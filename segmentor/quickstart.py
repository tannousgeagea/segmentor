"""Quickstart script to verify installation."""

import sys

import numpy as np


def main() -> None:
    """Run quickstart checks."""
    print("Segmentor Quickstart")
    print("=" * 50)

    # Check imports
    try:
        from segmentor import Segmentor, SegmentorConfig

        print("✓ Segmentor package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import segmentor: {e}")
        sys.exit(1)

    # Check PyTorch backend
    try:
        import torch

        print(f"✓ PyTorch {torch.__version__} available")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not available (install with: pip install segmentor[torch])")

    # Test basic functionality
    try:
        print("\nTesting basic segmentation...")
        config = SegmentorConfig()
        config.runtime.device = "cpu"  # Use CPU for testing

        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_box = (100, 100, 300, 300)

        segmentor = Segmentor(config=config)
        result = segmentor.segment_from_box(dummy_image, dummy_box, output_formats=["numpy"])

        print(f"✓ Segmentation successful!")
        print(f"  Mask shape: {result.mask.shape}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Latency: {result.latency_ms:.1f}ms")

        segmentor.close()

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 50)
    print("All checks passed! Segmentor is ready to use.")


if __name__ == "__main__":
    main()