"""
app/models/calibration.py
--------------------------
Isotonic regression confidence calibrator (Phase 4).

Per 04_LOGIC_FLOW.md §8:
  raw_confidence = min(intent_conf, action_conf)   # weaker link dominates
  calibrated_confidence = isotonic_curve.predict(raw_confidence)

The calibration curve is fit on a HELD-OUT slice of dev.jsonl (never the
same rows used to tune classifiers or POLICY_BOOST), targeting empirical
action accuracy per the top-70%/overall error-ratio bar (≤ 0.85).

Thread-safe: calibration curve is read-only after load().
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path

import numpy as np


class ConfidenceCalibrator:
    """
    Maps raw (intent, action) confidence scores to calibrated probabilities
    via a fitted IsotonicRegression curve.

    Usage
    -----
        cal = ConfidenceCalibrator()
        cal.load()
        calibrated_conf = cal.calibrate(raw_confidence)
    """

    DEFAULT_CURVE_PATH = Path("artifacts/models/calibration/calibrator.pkl")

    def __init__(self, curve_path: Path | str | None = None) -> None:
        self._curve_path = Path(curve_path) if curve_path else self.DEFAULT_CURVE_PATH
        self._calibrator = None  # sklearn IsotonicRegression after load()
        self._is_loaded: bool = False
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load calibration curve from disk. Idempotent."""
        with self._lock:
            if self._is_loaded:
                return

            if not self._curve_path.exists():
                # Graceful degradation: identity mapping until curve is trained.
                # This lets the service start and serve requests; calibration
                # is a quality enhancement, not a correctness requirement.
                self._calibrator = None
                self._is_loaded = True
                return

            with open(self._curve_path, "rb") as fh:
                self._calibrator = pickle.load(fh)
            self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def has_curve(self) -> bool:
        """True if a real calibration curve is loaded (vs identity fallback)."""
        return self._calibrator is not None

    def calibrate(self, raw_confidence: float) -> float:
        """
        Map a raw confidence score ∈ [0, 1] to a calibrated probability.

        If no calibration curve is loaded (Phase 2/3 before Phase 4 runs),
        returns raw_confidence clamped to [0, 1] — an identity mapping that
        doesn't break the service but also doesn't pass the top-70% bar.

        Parameters
        ----------
        raw_confidence : float
            Typically min(intent_prob_top1, action_prob_top1).

        Returns
        -------
        calibrated_confidence : float ∈ [0.0, 1.0], rounded to 4 dp.
        """
        clamped = float(np.clip(raw_confidence, 0.0, 1.0))

        if self._calibrator is None:
            return round(clamped, 4)

        try:
            cal = self._calibrator.predict([[clamped]])[0]
            return round(float(np.clip(cal, 0.0, 1.0)), 4)
        except Exception:  # noqa: BLE001 - Production triage service contract requires never crashing
            # Never crash on calibration; fall back to identity
            return round(clamped, 4)

    def raw_to_calibrated_batch(self, raw_scores: list[float]) -> list[float]:
        """
        Batch version — used by the training script to evaluate the
        top-70%/overall error ratio on a held-out calibration slice.
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() before calibrating")

        if self._calibrator is None:
            return [round(float(np.clip(s, 0.0, 1.0)), 4) for s in raw_scores]

        arr = np.clip(raw_scores, 0.0, 1.0).reshape(-1, 1)
        cal = self._calibrator.predict(arr)
        return [round(float(np.clip(v, 0.0, 1.0)), 4) for v in cal]
