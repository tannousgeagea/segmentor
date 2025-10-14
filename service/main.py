"""FastAPI application factory."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from perceptra_seg.config import SegmentorConfig
from service.middleware import LoggingMiddleware
from service.routes import router

logger = logging.getLogger(__name__)


def create_app(config: SegmentorConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Segmentor configuration

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = SegmentorConfig()

    app = FastAPI(
        title="Segmentor API",
        description="Production segmentation service with SAM models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging middleware
    app.add_middleware(LoggingMiddleware)

    # Store config in app state
    app.state.config = config
    app.state.segmentor = None

    # Routes
    app.include_router(router, prefix="/v1")

    # Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Startup/shutdown events
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize segmentor on startup."""
        from perceptra_seg import Segmentor

        logger.info("Initializing Segmentor...")
        app.state.segmentor = Segmentor(config=config)
        logger.info("Segmentor ready")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Clean up resources on shutdown."""
        if app.state.segmentor:
            app.state.segmentor.close()
            logger.info("Segmentor closed")

    return app


# For uvicorn: uvicorn service.main:app
app = create_app()