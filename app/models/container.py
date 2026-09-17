"""
app/models/container.py
------------------------
ModelContainer: single owner of all model artifacts loaded at startup.

All three models (intent, action, calibrator) plus the policy engine and
feature extractor live here.  The container is attached to `app.state` during
the FastAPI lifespan and retrieved via `request.app.state.models`.

Thread-safety: each individual model already uses threading.Lock for its own
load().  The container is fully initialised before the server accepts requests,
so no additional locking is needed at the container level.

Usage (in lifespan):
    container = ModelContainer(settings)
    container.load()
    app.state.models = container
"""

from __future__ import annotations

import os

# Must be set before importing lightgbm or sentence-transformers to prevent
# the macOS ARM64 OpenMP/Rayon SIGSEGV crash (see .env for context).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


from app.core.config import Settings
from app.models.action import ActionClassifier
from app.models.calibration import ConfidenceCalibrator
from app.models.features import ACTION_VOCABULARY, DialogueFeatureExtractor
from app.models.intent import IntentClassifier
from app.policy.engine import PolicyEngine


class ModelContainer:
    """
    Owns and lazily-initialises all runtime model artefacts.

    Parameters
    ----------
    settings : Settings
        Application settings (paths, thresholds).  Pass the result of
        Settings.load_with_overrides() so the container sees live YAML values.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.intent_clf = IntentClassifier(
            model_dir=str(settings.intent_model_dir),
            classes_path=str(settings.intent_classes_path),
        )
        self.action_clf = ActionClassifier(
            model_path=settings.action_model_path,
            classes_path=settings.action_classes_path,
            columns_path=settings.action_columns_path,
            policy_boost=settings.policy_boost,
        )
        self.calibrator = ConfidenceCalibrator(
            curve_path=settings.calibrator_path,
        )
        self.policy_engine = PolicyEngine(
            guidelines_path=settings.guidelines_path,
            overrides_path=settings.policy_overrides_path,
        )
        self._is_loaded = False

    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load all model artefacts from disk.  Call once during app startup.
        Idempotent — safe to call multiple times (each model guards itself).
        """
        if self._is_loaded:
            return

        self.intent_clf.load()

        # Build feature extractor using the loaded intent class list
        self.feature_extractor = DialogueFeatureExtractor(
            action_vocabulary=ACTION_VOCABULARY,
            intent_classes=self.intent_clf.classes_,
            embedding_dim=384,
        )

        self.action_clf.load()
        self.calibrator.load()

        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
