"""Telemetry, logging, and metrics package."""

from app.telemetry.logging import configure_logging, get_logger
from app.telemetry.metrics import (
    TRIAGE_ERRORS_TOTAL,
    TRIAGE_HOSTILE_FLAGS_TOTAL,
    TRIAGE_LATENCY_SECONDS,
    TRIAGE_NEEDS_HUMAN_TOTAL,
    TRIAGE_REQUESTS_TOTAL,
)

__all__ = [
    "TRIAGE_ERRORS_TOTAL",
    "TRIAGE_HOSTILE_FLAGS_TOTAL",
    "TRIAGE_LATENCY_SECONDS",
    "TRIAGE_NEEDS_HUMAN_TOTAL",
    "TRIAGE_REQUESTS_TOTAL",
    "configure_logging",
    "get_logger",
]
