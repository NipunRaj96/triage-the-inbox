"""FastAPI application factory with lifespan model loading and global error handlers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.models.container import ModelContainer
from app.telemetry import configure_logging, get_logger

logger = get_logger("service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager: loads all runtime models, feature extractors,
    and policy tables once at service startup.
    """
    settings = get_settings()
    configure_logging()
    logger.info(
        "service_startup_beginning",
        app_name=settings.app_name,
        env=settings.environment,
    )

    if not hasattr(app.state, "models") or app.state.models is None:
        container = ModelContainer(settings)
        container.load()
        app.state.models = container
        logger.info("models_loaded_successfully")

    if not hasattr(app.state, "settings") or app.state.settings is None:
        app.state.settings = settings

    yield

    logger.info("service_shutdown_complete")


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI service."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)

    # Global handler for malformed JSON / HTTP envelopes -> HTTP 400
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "malformed_request",
                    "message": "Invalid request envelope or schema violation.",
                    "details": exc.errors(),
                }
            },
        )

    return app


app = create_app()
