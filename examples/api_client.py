"""Example API client for Segmentor service.

This demonstrates how to use Segmentor as a REST API.
"""

import base64
import io
from pathlib import Path

import requests
from PIL import Image


class SegmentorClient:
    """Client for Segmentor REST API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None
    ) -> None:
        """Initialize client.
        
        Args:
            base_url: Base URL of the Segmentor service
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def health_check(self) -> bool:
        """Check if service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/v1/healthz")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def segment_from_box(
        self,
        image_path: str | Path,
        box: tuple[int, int, int, int],
        output_formats: list[str] | None = None
    ) -> dict:
        """Segment object from bounding box.
        
        Args:
            image_path: Path to image file
            box: Bounding box as (x1, y1, x2, y2)
            output_formats: List of output formats
            
        Returns:
            Dictionary with segmentation results
        """
        # Load and encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Make request
        response = requests.post(
            f"{self.base_url}/v1/segment/box",
            headers=self.headers,
            json={
                "image": img_b64,
                "box": list(box),
                "output_formats": output_formats or ["rle"],
            }
        )
        response.raise_for_status()
        return response.json()
    
    def segment_from_points(
        self,
        image_path: str | Path,
        points: list[tuple[int, int, int]],
        output_formats: list[str] | None = None
    ) -> dict:
        """Segment object from point prompts.
        
        Args:
            image_path: Path to image file
            points: List of (x, y, label) tuples
            output_formats: List of output formats
            
        Returns:
            Dictionary with segmentation results
        """
        # Load and encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Convert points to API format
        api_points = [
            {"x": x, "y": y, "label": label}
            for x, y, label in points
        ]
        
        # Make request
        response = requests.post(
            f"{self.base_url}/v1/segment/points",
            headers=self.headers,
            json={
                "image": img_b64,
                "points": api_points,
                "output_formats": output_formats or ["rle"],
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_mask_image(self, result: dict) -> Image.Image | None:
        """Extract mask image from result.
        
        Args:
            result: Result dictionary from API
            
        Returns:
            PIL Image or None if no PNG mask in result
        """
        if "png_base64" not in result:
            return None
        
        mask_bytes = base64.b64decode(result["png_base64"])
        return Image.open(io.BytesIO(mask_bytes))


def main() -> None:
    """Run API client example."""
    
    # Initialize client
    client = SegmentorClient(base_url="http://localhost:8080")
    
    # Check health
    print("Checking service health...")
    if not client.health_check():
        print("❌ Service is not available!")
        print("Start the service with: segmentor-cli serve")
        return
    print("✓ Service is healthy")
    
    # Create a test image
    print("\nCreating test image...")
    img = Image.new("RGB", (640, 480), color="black")
    # Draw a white rectangle
    for x in range(150, 350):
        for y in range(100, 300):
            img.putpoint((x, y), (255, 255, 255))
    
    img_path = "test_api_image.png"
    img.save(img_path)
    
    # Segment from box
    print("\nSegmenting from bounding box...")
    result = client.segment_from_box(
        img_path,
        box=(150, 100, 350, 300),
        output_formats=["rle", "png"]
    )
    
    print(f"Score: {result['score']:.3f}")
    print(f"Area: {result['area']} pixels")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    
    # Save mask
    mask_img = client.get_mask_image(result)
    if mask_img:
        mask_img.save("api_output_mask.png")
        print("✓ Saved mask to api_output_mask.png")
    
    # Cleanup
    Path(img_path).unlink()
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()