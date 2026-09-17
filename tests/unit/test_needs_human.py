"""Unit tests for needs_human decision logic."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.needs_human import (
    REASON_ALWAYS_ESC,
    REASON_LOW_CONF,
    REASON_NONE,
    REASON_POLICY_DEV,
    REASON_UNHANDLEABLE,
    decide_needs_human,
)
from app.sanitization.sanitizer import SanitizeResult


@pytest.fixture
def base_settings():
    return Settings(
        human_threshold=0.62,
        policy_deviation_threshold=0.80,
        always_escalate_actions=["escalate", "manager-review"],
    )


def test_trigger_unhandleable_binary_junk(base_settings):
    san = SanitizeResult(clean_text="???", is_binary_junk=True)
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.99,
        is_deviation=False,
        action="search-faq",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is True
    assert reason == REASON_UNHANDLEABLE


def test_trigger_unhandleable_empty(base_settings):
    san = SanitizeResult(clean_text="", is_empty=True)
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.99,
        is_deviation=False,
        action="search-faq",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is True
    assert reason == REASON_UNHANDLEABLE


def test_trigger_always_escalate(base_settings):
    san = SanitizeResult(clean_text="I demand to speak to someone")
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.95,
        is_deviation=False,
        action="escalate",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is True
    assert reason == REASON_ALWAYS_ESC


def test_trigger_manager_review(base_settings):
    san = SanitizeResult(clean_text="Please escalate this order")
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.90,
        is_deviation=False,
        action="manager-review",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is True
    assert reason == REASON_ALWAYS_ESC


def test_trigger_policy_deviation_with_subthreshold_confidence(base_settings):
    # Off-policy (is_deviation=True) and confidence is below 0.80
    san = SanitizeResult(clean_text="Refund my account")
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.75,
        is_deviation=True,
        action="offer-refund",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is True
    assert reason == REASON_POLICY_DEV


def test_policy_deviation_with_high_confidence_does_not_escalate_deviation(
    base_settings,
):
    # Off-policy but very confident (>= 0.80) -> does NOT escalate on deviation
    san = SanitizeResult(clean_text="Refund my account")
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.88,
        is_deviation=True,
        action="offer-refund",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is False
    assert reason == REASON_NONE


def test_trigger_low_confidence(base_settings):
    # On-policy but confidence below human_threshold (0.62)
    san = SanitizeResult(clean_text="Need help with account")
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.55,
        is_deviation=False,
        action="pull-up-account",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is True
    assert reason == REASON_LOW_CONF


def test_all_clear_confident_and_on_policy(base_settings):
    # On-policy, high confidence, clean input -> no human needed
    san = SanitizeResult(clean_text="I want to track order 12345")
    needs_human, reason = decide_needs_human(
        calibrated_conf=0.92,
        is_deviation=False,
        action="track-order",
        sanitize_result=san,
        settings=base_settings,
    )
    assert needs_human is False
    assert reason == REASON_NONE
