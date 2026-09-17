"""Unit tests for policy parsing and overrides."""

from app.policy import PolicyEngine


def test_policy_engine_loads_guidelines():
    engine = PolicyEngine(guidelines_path="data/guidelines.json")
    # Subflow "Initiate Refund" should have permitted actions
    actions = engine.get_permitted_actions("initiate_refund")
    assert len(actions) > 0
    assert "pull-up-account" in actions
    assert "offer-refund" in actions


def test_policy_engine_unknown_intent():
    engine = PolicyEngine(guidelines_path="data/guidelines.json")
    actions = engine.get_permitted_actions("non_existent_flow")
    assert actions == []
