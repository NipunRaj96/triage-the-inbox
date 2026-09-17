"""Unit tests for ModelContainer and graceful degradation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.calibration import ConfidenceCalibrator
from app.models.container import ModelContainer


def test_container_missing_intent_model_raises_error(tmp_path: Path):
    fake_settings = Settings(
        intent_model_dir=tmp_path / "nonexistent_intent",
        intent_classes_path=tmp_path / "nonexistent_classes.json",
    )
    container = ModelContainer(fake_settings)
    with pytest.raises(FileNotFoundError):
        container.load()


def test_calibrator_graceful_fallback(tmp_path: Path):
    # Calibrator should not crash if curve_path is missing; it should degrade gracefully
    cal = ConfidenceCalibrator(curve_path=tmp_path / "missing_curve.pkl")
    cal.load()
    assert cal.is_loaded is True
    assert cal.has_curve is False
    assert cal.calibrate(0.75) == 0.75
    assert cal.calibrate(-0.1) == 0.0
    assert cal.calibrate(1.5) == 1.0


def test_container_loads_runtime_models():
    settings = Settings()
    if (
        not settings.intent_model_dir.exists()
        or not settings.action_model_path.exists()
    ):
        pytest.skip("Model artifacts not present on disk")

    container = ModelContainer(settings)
    assert container.is_loaded is False
    container.load()
    assert container.is_loaded is True
    # Verify idempotency
    container.load()
    assert container.is_loaded is True
