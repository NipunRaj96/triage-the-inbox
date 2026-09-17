"""
app/models/intent.py
--------------------
Intent classifier — production inference wrapper around a saved SetFit model.

The class is loaded once at service startup (lazy, thread-safe singleton pattern
in app/core/config.py). At request time, predict() is sub-millisecond for CPU
inference after the model is warmed up.

Interface (per 04_LOGIC_FLOW.md §4):
    intent, intent_probs = intent_model.predict(text: str)
    intent_probs is a dict[str, float] — full distribution over all 96 classes,
    values sum to ~1.0, preserving probability mass for the action classifier
    feature vector.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np


class IntentClassifier:
    """
    Production-grade intent classifier wrapping a SavedPretrained SetFit model.

    Thread-safe: the underlying sentence-transformer + head are read-only
    after load(); concurrent predict() calls are safe.
    """

    # Default artifact paths (relative to repo root; override in tests)
    DEFAULT_MODEL_DIR = Path("artifacts/models/intent")
    DEFAULT_CLASSES_PATH = Path("artifacts/models/intent_classes.json")

    def __init__(
        self,
        model_dir: Path | str | None = None,
        classes_path: Path | str | None = None,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._classes_path = (
            Path(classes_path) if classes_path else self.DEFAULT_CLASSES_PATH
        )
        self._model = None  # SetFitModel, after load()
        self._classes: list[str] = []  # ordered label list
        self._is_loaded: bool = False
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load model artifacts from disk.  Call once at service startup.
        Idempotent — calling load() again is a no-op.
        """
        with self._lock:
            if self._is_loaded:
                return

            if not self._model_dir.exists():
                raise FileNotFoundError(
                    f"Intent model directory not found: {self._model_dir}\n"
                    "Run:  python scripts/train_intent.py --train data/train.jsonl.gz "
                    "--dev data/dev.jsonl --noisy data/dev_noisy.jsonl"
                )

            # Lazy import so the module can be imported without torch
            from setfit import SetFitModel

            self._model = SetFitModel.from_pretrained(str(self._model_dir))
            self._classes = self._load_classes()
            self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def classes_(self) -> list[str]:
        """Ordered list of intent labels (same order as probability vector)."""
        return list(self._classes)

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Classify a dialogue context string into an intent subflow.

        Parameters
        ----------
        text : str
            Formatted dialogue context (produced by DialogueFeatureExtractor
            .format_context_for_embedding).  May be empty after sanitisation.

        Returns
        -------
        intent : str
            Top-1 predicted intent label.
        intent_probs : dict[str, float]
            Full probability distribution over all intent classes.
            Sums to ~1.0.  Passed into the action classifier feature vector.
        """
        if not self._is_loaded:
            raise RuntimeError("Call IntentClassifier.load() before predict()")

        # Handle empty / whitespace-only input — sanitizer routes truly empty
        # inputs away, but be defensive here too.
        if not text or not text.strip():
            return self._fallback_prediction()

        try:
            proba = self._model.predict_proba([text])
            # SetFit returns a torch Tensor or np.ndarray; normalise to ndarray
            if hasattr(proba, "numpy"):
                proba = proba.numpy()
            proba = np.asarray(proba, dtype=np.float32)[0]  # shape: (n_classes,)

            # Map to labels — SetFit preserves training label order
            model_labels = self._get_model_labels()
            idx = int(np.argmax(proba))
            top_intent = model_labels[idx]
            prob_dict = {lbl: float(p) for lbl, p in zip(model_labels, proba)}
            return top_intent, prob_dict

        except Exception:  # noqa: BLE001 - Production triage service contract requires never crashing
            # Any unexpected inference error → safe fallback; never crash
            return self._fallback_prediction()

    def encode(self, text_or_texts: str | list[str]) -> np.ndarray:
        """
        Produce sentence embeddings using the underlying sentence transformer backbone.
        Returns 1D array of shape (embedding_dim,) for single str, or 2D array for list[str].
        """
        if not self._is_loaded:
            raise RuntimeError("Call IntentClassifier.load() before encode()")
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else text_or_texts
        try:
            embs = self._model.encode(texts)
            arr = np.asarray(embs, dtype=np.float32)
            return arr[0] if is_single else arr
        except Exception:  # noqa: BLE001 - Production triage service contract requires never crashing
            dim = 384
            return (
                np.zeros(dim, dtype=np.float32)
                if is_single
                else np.zeros((len(texts), dim), dtype=np.float32)
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_classes(self) -> list[str]:
        """Load class list from the JSON sidecar next to the model directory."""
        if self._classes_path.exists():
            return json.loads(self._classes_path.read_text(encoding="utf-8"))
        # Fallback: ask the model's head for its label list (SetFit ≥ 1.0)
        return self._get_model_labels()

    def _get_model_labels(self) -> list[str]:
        """Extract label order from the loaded SetFit model head."""
        if self._model is None:
            return []
        # SetFit stores id2label in the model's config
        head = getattr(self._model, "model_head", None)
        if head is None:
            return list(self._classes) if self._classes else []
        id2label = getattr(head, "id2label", None) or getattr(head, "classes_", None)
        if id2label is None:
            return list(self._classes) if self._classes else []
        if isinstance(id2label, dict):
            # id2label is {0: "class_a", 1: "class_b", ...}
            return [id2label[i] for i in sorted(id2label.keys())]
        # numpy array / list — already ordered
        return [str(c) for c in id2label]

    def _fallback_prediction(self) -> tuple[str, dict[str, float]]:
        """
        Safe fallback: return 'unknown' with uniform distribution.
        Used when input is empty or inference unexpectedly errors.
        """
        if self._classes:
            uniform = 1.0 / len(self._classes)
            return "unknown", {c: uniform for c in self._classes}
        return "unknown", {"unknown": 1.0}
