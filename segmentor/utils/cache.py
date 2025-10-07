"""Caching utilities."""

import hashlib
import time
from collections import OrderedDict
from typing import Any


class EmbeddingCache:
    """LRU cache for image embeddings."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """Get cached value if exists and not expired."""
        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = (value, time.time())

    def compute_image_hash(self, image: Any) -> str:
        """Compute hash for image data."""
        import numpy as np

        if isinstance(image, np.ndarray):
            return hashlib.sha256(image.tobytes()).hexdigest()[:16]
        return hashlib.sha256(str(image).encode()).hexdigest()[:16]

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()