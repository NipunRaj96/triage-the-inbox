"""
app/core/needs_human.py
-----------------------
Pure function implementing the four `needs_human` trigger conditions from
04_LOGIC_FLOW.md §9.

Kept deliberately free of side effects so each trigger can be unit-tested
independently without spinning up the full application.
"""

from __future__ import annotations

from app.core.config import Settings
from app.sanitization.sanitizer import SanitizeResult

# Reason codes — surfaced in the response for observability
REASON_UNHANDLEABLE = "unhandleable_input"
REASON_LOW_CONF = "low_confidence"
REASON_POLICY_DEV = "policy_deviation"
REASON_ALWAYS_ESC = "always_escalate"
REASON_NONE = "none"


def decide_needs_human(
    calibrated_conf: float,
    is_deviation: bool,
    action: str,
    sanitize_result: SanitizeResult,
    settings: Settings,
) -> tuple[bool, str]:
    """
    Evaluate whether this request should be escalated to a human agent.

    Triggers are checked in priority order; the first match wins.

    Parameters
    ----------
    calibrated_conf : float
        Isotonic-regression-calibrated confidence ∈ [0, 1].
    is_deviation : bool
        True if the predicted action is not in the policy-permitted set.
    action : str
        Predicted action label.
    sanitize_result : SanitizeResult
        Output of the sanitization step for the last customer turn.
    settings : Settings
        Live policy thresholds from config/policy_overrides.yaml.

    Returns
    -------
    (needs_human: bool, reason: str)
        reason is one of the REASON_* constants above.
    """
    # Trigger 1 — unhandleable input (binary junk or empty after cleaning)
    if sanitize_result.is_binary_junk or sanitize_result.is_empty:
        return True, REASON_UNHANDLEABLE

    # Trigger 4 — always-escalate action set (checked before conf to avoid
    # over-relying on model confidence for hard-coded escalations)
    if action in settings.always_escalate_actions:
        return True, REASON_ALWAYS_ESC

    # Trigger 3 — off-policy AND below the looser deviation threshold
    # (Don't escalate every off-policy call — some are the correct call;
    # only escalate the ones we're also uncertain about.)
    if is_deviation and calibrated_conf < settings.policy_deviation_threshold:
        return True, REASON_POLICY_DEV

    # Trigger 2 — calibrated confidence below the main human threshold
    if calibrated_conf < settings.human_threshold:
        return True, REASON_LOW_CONF

    return False, REASON_NONE
