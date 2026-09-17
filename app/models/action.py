"""
app/models/action.py
--------------------
LightGBM action classifier with policy-aware soft prior re-scoring.

Interface (per 04_LOGIC_FLOW.md §6-7):
    action, raw_conf, is_deviation = action_model.predict(features, permitted_actions)
    action_probs dict is also available via predict_proba() for downstream calibration.

The policy-aware re-scoring (§7) applies a soft POLICY_BOOST multiplier to
permitted actions rather than a hard mask, preserving recall on the ~9% of
gold labels that are genuinely off-policy.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np


class ActionClassifier:
    """
    LightGBM multiclass classifier for next-action prediction.

    Loads a saved LightGBM Booster from disk and wraps it with:
    - Tabular feature alignment (ordered column list sidecar)
    - Soft policy-aware re-scoring (POLICY_BOOST config value)
    - Thread-safe lazy loading
    """

    DEFAULT_MODEL_PATH = Path("artifacts/models/action/action_model.pkl")
    DEFAULT_CLASSES_PATH = Path("artifacts/models/action/action_classes.json")
    DEFAULT_COLUMNS_PATH = Path("artifacts/models/action/feature_columns.json")

    def __init__(
        self,
        model_path: Path | str | None = None,
        classes_path: Path | str | None = None,
        columns_path: Path | str | None = None,
        policy_boost: float = 1.0,
    ) -> None:
        self._model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self._classes_path = (
            Path(classes_path) if classes_path else self.DEFAULT_CLASSES_PATH
        )
        self._columns_path = (
            Path(columns_path) if columns_path else self.DEFAULT_COLUMNS_PATH
        )
        # 1.0 chosen by POLICY_BOOST sweep (Phase 3): best clean acc 0.7425,
        # violation rate 0.0565 ≤ 2× gold rate.  See results/action_eval.json.
        self.policy_boost = policy_boost

        self._booster = None  # lightgbm.Booster after load()
        self._action_classes: list[str] = []
        self._feature_columns: list[str] = []
        self._is_loaded: bool = False
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model artifacts from disk. Idempotent."""
        with self._lock:
            if self._is_loaded:
                return

            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Action model not found: {self._model_path}\n"
                    "Run:  python scripts/train_action.py"
                )

            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=str(self._model_path))

            self._action_classes = json.loads(
                self._classes_path.read_text(encoding="utf-8")
            )
            self._feature_columns = json.loads(
                self._columns_path.read_text(encoding="utf-8")
            )
            self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def action_classes(self) -> list[str]:
        return list(self._action_classes)

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        """
        Return raw (pre-reweighting) class probabilities for all action labels.

        Parameters
        ----------
        features : dict produced by DialogueFeatureExtractor.extract_state_features()

        Returns
        -------
        dict[action_label -> probability]
        """
        if not self._is_loaded:
            raise RuntimeError("Call ActionClassifier.load() before predict_proba()")

        x = self._align_features(features)

        # LightGBM predict returns shape (1, n_classes) for single sample
        raw = self._booster.predict(x.reshape(1, -1))
        proba = np.asarray(raw, dtype=np.float32).flatten()

        return {cls: float(p) for cls, p in zip(self._action_classes, proba)}

    def predict(
        self,
        features: dict[str, Any],
        permitted_actions: list[str],
    ) -> tuple[str, float, bool]:
        """
        Predict next action with policy-aware soft re-scoring.

        Parameters
        ----------
        features          : feature dict from DialogueFeatureExtractor
        permitted_actions : list of action labels allowed per policy for this intent

        Returns
        -------
        predicted_action  : str   — argmax of adjusted scores
        raw_confidence    : float — max adjusted score (before calibration)
        is_deviation      : bool  — True if predicted action ∉ permitted_actions
        """
        if not self._is_loaded:
            raise RuntimeError("Call ActionClassifier.load() before predict()")

        raw_probs = self.predict_proba(features)

        # § 7 Soft policy re-scoring
        adjusted: dict[str, float] = {}
        for action, prob in raw_probs.items():
            if action in permitted_actions:
                adjusted[action] = prob * (1.0 + self.policy_boost)
            else:
                adjusted[action] = prob

        predicted_action = max(adjusted, key=adjusted.__getitem__)
        raw_confidence = adjusted[predicted_action]
        # Normalise confidence back to [0, 1] after boost
        total = sum(adjusted.values())
        raw_confidence = raw_confidence / total if total > 0 else 0.0
        is_deviation = predicted_action not in permitted_actions

        return predicted_action, float(raw_confidence), is_deviation

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _align_features(self, features: dict[str, Any]) -> np.ndarray:
        """
        Produce a 1-D float32 array aligned to the training column order.
        Missing features default to 0.0; extra keys are silently dropped.
        """
        return np.array(
            [float(features.get(col, 0.0)) for col in self._feature_columns],
            dtype=np.float32,
        )
