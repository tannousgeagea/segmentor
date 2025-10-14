"""Batch processing example.

Shows how to efficiently process multiple images.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from perceptra_seg import Segmentor


def generate_test_images(count: int = 5) -> Iterator[tuple[str, np.ndarray]]:
    """Generate test images."""
    for i in range(count):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Random white square
        x, y = np.random.randint(50, 400, 2)
        size = np.random.randint(100, 200)
        img[y:y+size, x:x+size] = 255
        yield f"image_{i}.png", img


def main() -> None:
    """Run batch processing example."""
    
    print("Batch Processing Example")
    print("=" * 50)
    
    # Initialize segmentor once
    print("\nInitializing Segmentor...")
    seg = Segmentor(
        backend="torch",
        model="sam_v1",
        device="cuda",
        cache={"enabled": True}  # Enable caching for better performance
    )
    
    # Warm up the model
    print("Warming up model...")
    seg.warmup(image_size=(480, 640))
    
    # Process images
    print("\nProcessing images...")
    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)
    
    total_time = 0
    results = []
    
    for name, image in generate_test_images(count=10):
        # Find the white square (simple approach)
        coords = np.argwhere(image[:, :, 0] > 0)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        box = (x_min, y_min, x_max, y_max)
        
        # Segment
        result = seg.segment_from_box(
            image,
            box=box,
            output_formats=["png"]
        )
        
        # Save
        mask_path = output_dir / f"mask_{name}"
        Path(mask_path).write_bytes(result.png_bytes)
        
        # Track metrics
        total_time += result.latency_ms
        results.append({
            "name": name,
            "score": result.score,
            "area": result.area,
            "latency_ms": result.latency_ms,
        })
        
        print(f"  {name}: score={result.score:.3f}, "
              f"latency={result.latency_ms:.1f}ms")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Processed {len(results)} images")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Average latency: {total_time / len(results):.1f}ms per image")
    print(f"Throughput: {1000 * len(results) / total_time:.2f} images/sec")
    print(f"\n✓ Results saved to {output_dir}/")
    
    seg.close()


if __name__ == "__main__":
    main()