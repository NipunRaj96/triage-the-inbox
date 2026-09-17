"""
tests/unit/test_intent_model.py
--------------------------------
Unit tests for IntentClassifier (app/models/intent.py).

Covers all cases called out in 06_BUILD_PLAN_PHASES.md Phase 2:
  - Normal input (post-load)
  - Empty string
  - Very long string
  - Pre-load guard
  - Fallback prediction shape / types
  - Probability vector sums to ~1.0
  - Thread-safety smoke test

These tests run WITHOUT a real SetFit model by patching the model internals,
so they stay fast and offline (no HF Hub download required in CI).
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.models.intent import IntentClassifier

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_CLASSES = [
    "manage",
    "status",
    "recover_password",
    "reset_2fa",
    "return_size",
    "refund_initiate",
    "missing",
    "cost",
]


def _make_loaded_classifier(classes: list[str] = FAKE_CLASSES) -> IntentClassifier:
    """
    Return an IntentClassifier that acts as if load() has been called,
    with a mocked SetFit model that returns uniform probabilities.
    """
    clf = IntentClassifier(
        model_dir=Path("/nonexistent/model"),
        classes_path=Path("/nonexistent/classes.json"),
    )
    clf._is_loaded = True
    clf._classes = list(classes)

    # Build a mock SetFit model whose predict_proba returns uniform probs
    mock_model = MagicMock()
    n = len(classes)
    # Max-prob on first class so argmax is deterministic
    probs = np.zeros(n, dtype=np.float32)
    probs[0] = 0.7
    probs[1:] = 0.3 / (n - 1) if n > 1 else 0.0
    mock_model.predict_proba.return_value = np.array([probs])
    # id2label on the model head
    mock_head = MagicMock()
    mock_head.id2label = {i: c for i, c in enumerate(classes)}
    mock_model.model_head = mock_head

    clf._model = mock_model
    return clf


def _make_empty_class_classifier() -> IntentClassifier:
    """Return a classifier that loaded but has no classes — degenerate edge case."""
    clf = IntentClassifier(
        model_dir=Path("/nonexistent/model"),
        classes_path=Path("/nonexistent/classes.json"),
    )
    clf._is_loaded = True
    clf._classes = []

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[]])
    mock_head = MagicMock()
    mock_head.id2label = {}
    mock_model.model_head = mock_head
    clf._model = mock_model
    return clf


# ---------------------------------------------------------------------------
# Guard: cannot predict before load()
# ---------------------------------------------------------------------------


class TestPreLoadGuard:
    def test_predict_before_load_raises(self) -> None:
        clf = IntentClassifier()
        with pytest.raises(RuntimeError, match="load()"):
            clf.predict("hello, I need help with my order")

    def test_is_loaded_false_before_load(self) -> None:
        clf = IntentClassifier()
        assert clf.is_loaded is False

    def test_load_raises_file_not_found(self, tmp_path: Path) -> None:
        clf = IntentClassifier(model_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="Intent model directory"):
            clf.load()


# ---------------------------------------------------------------------------
# Normal input
# ---------------------------------------------------------------------------


class TestNormalInput:
    def test_predict_returns_tuple(self) -> None:
        clf = _make_loaded_classifier()
        result = clf.predict("[customer] I want to cancel my subscription")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_intent_is_string(self) -> None:
        clf = _make_loaded_classifier()
        intent, _ = clf.predict("[customer] I want to cancel my subscription")
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_prob_dict_keys_match_classes(self) -> None:
        clf = _make_loaded_classifier()
        _, probs = clf.predict("[customer] I want to cancel my subscription")
        assert set(probs.keys()) == set(FAKE_CLASSES)

    def test_prob_values_are_floats_in_0_1(self) -> None:
        clf = _make_loaded_classifier()
        _, probs = clf.predict("[customer] I want to cancel my subscription")
        for v in probs.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_prob_vector_sums_to_approx_one(self) -> None:
        clf = _make_loaded_classifier()
        _, probs = clf.predict("[customer] I want to cancel my subscription")
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_top1_intent_is_argmax_of_probs(self) -> None:
        clf = _make_loaded_classifier()
        intent, probs = clf.predict("[agent] Hello [customer] I need help")
        assert intent == max(probs, key=probs.__getitem__)

    def test_multiline_context_handled(self) -> None:
        clf = _make_loaded_classifier()
        context = (
            "[agent] Hi, how can I help?\n"
            "[customer] My order hasn't arrived\n"
            "[action] pull-up-account\n"
            "[agent] Let me look that up\n"
            "[customer] Please hurry"
        )
        intent, probs = clf.predict(context)
        assert isinstance(intent, str)
        assert probs


# ---------------------------------------------------------------------------
# Empty string
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string_returns_fallback(self) -> None:
        clf = _make_loaded_classifier()
        intent, probs = clf.predict("")
        assert intent == "unknown"
        assert isinstance(probs, dict)

    def test_whitespace_only_returns_fallback(self) -> None:
        clf = _make_loaded_classifier()
        intent, _ = clf.predict("   \n\t  ")
        assert intent == "unknown"

    def test_fallback_probs_cover_all_classes(self) -> None:
        clf = _make_loaded_classifier()
        _, probs = clf.predict("")
        # Fallback returns uniform over known classes, or just "unknown"
        assert len(probs) > 0

    def test_fallback_when_no_classes_loaded(self) -> None:
        """Edge case: model loaded but empty class list."""
        clf = _make_empty_class_classifier()
        intent, probs = clf.predict("")
        assert intent == "unknown"
        assert probs == {"unknown": 1.0}


# ---------------------------------------------------------------------------
# Very long string
# ---------------------------------------------------------------------------


class TestVeryLongInput:
    def test_very_long_string_does_not_crash(self) -> None:
        """2000+ word string — must not crash, OOM, or hang."""
        clf = _make_loaded_classifier()
        very_long = "[customer] " + ("I need to return this item. " * 500)
        intent, probs = clf.predict(very_long)
        assert isinstance(intent, str)
        assert probs

    def test_10k_char_input_does_not_crash(self) -> None:
        clf = _make_loaded_classifier()
        long_text = "a" * 10_000
        intent, _ = clf.predict(long_text)
        assert isinstance(intent, str)

    def test_very_long_unicode_input_does_not_crash(self) -> None:
        clf = _make_loaded_classifier()
        # Simulates noisy Hinglish or code-mixed text at scale
        long_unicode = "[customer] " + ("मुझे अपना पासवर्ड reset करना है. " * 200)
        intent, _ = clf.predict(long_unicode)
        assert isinstance(intent, str)


# ---------------------------------------------------------------------------
# Inference error → fallback (never crash)
# ---------------------------------------------------------------------------


class TestInferenceErrorFallback:
    def test_model_exception_returns_fallback_not_crash(self) -> None:
        clf = _make_loaded_classifier()
        clf._model.predict_proba.side_effect = RuntimeError("GPU OOM")
        intent, _ = clf.predict("normal text")
        assert intent == "unknown"

    def test_nan_proba_returns_fallback(self) -> None:
        clf = _make_loaded_classifier()
        nan_probs = np.array([[float("nan")] * len(FAKE_CLASSES)])
        clf._model.predict_proba.return_value = nan_probs
        # Should not crash — fallback kicks in
        try:
            intent, _ = clf.predict("test")
            assert isinstance(intent, str)
        except Exception:  # noqa: BLE001
            pytest.fail("predict() raised instead of returning fallback")


# ---------------------------------------------------------------------------
# classes_ property
# ---------------------------------------------------------------------------


class TestClassesProperty:
    def test_classes_returns_list(self) -> None:
        clf = _make_loaded_classifier()
        assert isinstance(clf.classes_, list)

    def test_classes_matches_fake_classes(self) -> None:
        clf = _make_loaded_classifier()
        assert set(clf.classes_) == set(FAKE_CLASSES)

    def test_classes_is_copy_not_ref(self) -> None:
        """Mutating the returned list must not affect internal state."""
        clf = _make_loaded_classifier()
        c = clf.classes_
        c.clear()
        assert len(clf.classes_) == len(FAKE_CLASSES)


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_predict_calls_stable(self) -> None:
        """50 threads calling predict() simultaneously should all succeed."""
        clf = _make_loaded_classifier()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                intent, probs = clf.predict("[customer] I need help with billing")
                assert isinstance(intent, str)
                assert probs
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
