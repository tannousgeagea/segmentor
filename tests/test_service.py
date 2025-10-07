"""Tests for FastAPI service."""

import base64

import pytest
from fastapi.testclient import TestClient

from segmentor.config import SegmentorConfig
from service.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"
    app = create_app(config)
    return TestClient(app)


@pytest.fixture
def sample_image_b64() -> str:
    """Create base64-encoded test image."""
    import numpy as np
    from PIL import Image
    import io

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    return base64.b64encode(img_bytes).decode()


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_segment_box(client: TestClient, sample_image_b64: str) -> None:
    """Test box segmentation endpoint."""
    payload = {
        "image": sample_image_b64,
        "box": [20, 20, 80, 80],
        "output_formats": ["rle"],
    }

    response = client.post("/v1/segment/box", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "area" in data
    assert "rle" in data


def test_segment_points(client: TestClient, sample_image_b64: str) -> None:
    """Test points segmentation endpoint."""
    payload = {
        "image": sample_image_b64,
        "points": [{"x": 50, "y": 50, "label": 1}],
        "output_formats": ["rle"],
    }

    response = client.post("/v1/segment/points", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "score" in data


def test_invalid_box(client: TestClient, sample_image_b64: str) -> None:
    """Test invalid box returns 400."""
    payload = {
        "image": sample_image_b64,
        "box": [0, 0, 10000, 10000],
    }

    response = client.post("/v1/segment/box", json=payload)

    assert response.status_code == 400