"""API package exposing service app and route handlers."""

from app.api.service import app, create_app

__all__ = ["app", "create_app"]
