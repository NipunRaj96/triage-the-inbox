"""Pydantic schemas strictly conforming to input.schema.json and schema.json."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    """A single turn in the conversation prefix."""

    model_config = ConfigDict(extra="forbid")

    speaker: Literal["agent", "customer", "action"] = Field(
        ..., description="The speaker or turn type"
    )
    text: str = Field(..., description="Turn utterance or action description")


class TriageRequest(BaseModel):
    """Incoming request shape per input.schema.json."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique conversation/turn ID")
    context: list[Turn] = Field(
        ..., max_length=8, description="Last observed conversation turns (up to 8)"
    )
    turn_index: int | None = Field(
        default=None, ge=0, description="Optional dialogue turn index"
    )


class ErrorDetail(BaseModel):
    """Optional error explanation for defined abstention/error envelope."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation")


class TriageResponse(BaseModel):
    """Output response shape per schema.json."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Echoed input row ID")
    intent: str = Field(..., description="Predicted ABCD subflow intent")
    action: str = Field(..., description="Next permitted action, or 'none'")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence [0.0, 1.0]"
    )
    needs_human: bool = Field(
        ..., description="Flag indicating human takeover is required"
    )
    needs_human_reason: str | None = Field(
        default=None, description="Reason code if human takeover is required"
    )
    cost_usd: float | None = Field(
        default=None, ge=0.0, description="Estimated inference cost in USD"
    )
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="Request processing time in ms"
    )
    error: ErrorDetail | None = Field(
        default=None, description="Optional abstention/error info"
    )
