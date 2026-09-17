"""Unit tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.core.schemas import TriageRequest, TriageResponse, Turn


def test_valid_triage_request():
    req = TriageRequest(
        id="1234-1",
        context=[
            Turn(speaker="customer", text="I want to return an item"),
            Turn(speaker="agent", text="What is your account id?"),
        ],
        turn_index=1,
    )
    assert req.id == "1234-1"
    assert len(req.context) == 2


def test_invalid_speaker_rejected():
    with pytest.raises(ValidationError):
        Turn(speaker="invalid_role", text="hello")


def test_valid_triage_response():
    resp = TriageResponse(
        id="1234-1",
        intent="return_size",
        action="pull-up-account",
        confidence=0.85,
        needs_human=False,
    )
    assert resp.confidence == 0.85
    assert not resp.needs_human
