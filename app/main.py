"""Entry point for ASGI servers (e.g. uvicorn app.main:app)."""

from app.api.service import app, create_app

__all__ = ["app", "create_app"]
