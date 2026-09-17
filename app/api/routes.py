"""FastAPI route definitions for /triage, /healthz, and /metrics with full ML inference pipeline."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, status
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from app.core.config import get_settings
from app.core.needs_human import REASON_UNHANDLEABLE, decide_needs_human
from app.core.schemas import ErrorDetail, TriageRequest, TriageResponse
from app.models.container import ModelContainer
from app.sanitization import resolve_empty_turn_action, sanitize
from app.telemetry import (
    TRIAGE_ERRORS_TOTAL,
    TRIAGE_NEEDS_HUMAN_TOTAL,
    TRIAGE_REQUESTS_TOTAL,
    get_logger,
)

router = APIRouter()
logger = get_logger("triage_api")


def _get_container(request: Request) -> ModelContainer:
    """Safely retrieve the loaded ModelContainer from app.state or initialize on demand."""
    if hasattr(request.app.state, "models") and request.app.state.models is not None:
        return request.app.state.models

    settings = getattr(request.app.state, "settings", None) or get_settings()
    container = ModelContainer(settings)
    container.load()
    request.app.state.models = container
    return container


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Liveness and readiness check."""
    return {"status": "ok", "service": "op06-triage"}


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return PlainTextResponse(generate_latest().decode("utf-8"))


@router.post("/triage", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def triage(req: TriageRequest, request: Request) -> TriageResponse:
    """
    Triage incoming conversation prefix:
    sanitize -> intent -> policy lookup -> feature engineering -> action prediction
    -> calibration -> needs_human decision -> telemetry -> response.
    """
    start_time = time.perf_counter()
    settings = getattr(request.app.state, "settings", None) or get_settings()

    try:
        container = _get_container(request)

        # 1. Resolve the best customer turn text for sanitization.
        #    In noisy inputs the last customer turn may be empty (""); fall back
        #    to the most-recent NON-EMPTY customer turn so we still have signal.
        #    Only bail out if EVERY customer turn is empty or there are none at all.
        all_customer_texts = [t.text for t in req.context if t.speaker == "customer"]
        non_empty_customer_texts = [t for t in all_customer_texts if t.strip()]
        last_turn_text = (
            non_empty_customer_texts[-1] if non_empty_customer_texts else ""
        )
        last_was_empty = bool(all_customer_texts) and not all_customer_texts[-1].strip()
        sanitize_result = sanitize(
            last_turn_text, max_tokens=settings.max_tokens_sanitize
        )

        # 2. Check for truly unhandleable input: binary junk OR no customer text at all.
        #    An empty *last* turn that has prior non-empty turns is NOT unhandleable.
        if sanitize_result.is_binary_junk or (
            sanitize_result.is_empty and not non_empty_customer_texts
        ):
            TRIAGE_NEEDS_HUMAN_TOTAL.labels(reason=REASON_UNHANDLEABLE).inc()
            TRIAGE_REQUESTS_TOTAL.labels(status="unhandleable").inc()
            latency_ms = (time.perf_counter() - start_time) * 1000
            return TriageResponse(
                id=req.id,
                intent="unknown",
                action="none",
                confidence=0.0,
                needs_human=True,
                needs_human_reason=REASON_UNHANDLEABLE,
                cost_usd=0.0,
                latency_ms=round(latency_ms, 3),
                error=ErrorDetail(
                    code="invalid_input",
                    message="Input text is empty, unparseable, or binary payload.",
                ),
            )

        # 3. Build model input text from dialogue context (last 8 turns, speaker formatted)
        model_input = container.feature_extractor.format_context_for_embedding(
            req.context, max_turns=8
        )

        # 4. Intent classification
        pred_intent, intent_probs = container.intent_clf.predict(model_input)

        # 5. Policy lookup
        permitted_actions = container.policy_engine.get_permitted_actions(pred_intent)

        # 6. Feature extraction for action classifier
        # When last customer turn was empty, use full formatted context for embedding
        # (richer signal for the 50 noisy rows with empty final turns).
        if last_was_empty:
            clean_text_for_emb = model_input
        else:
            clean_text_for_emb = (
                sanitize_result.clean_text
                if sanitize_result.clean_text
                else last_turn_text
            )
        last_emb = container.intent_clf.encode(clean_text_for_emb)

        features = container.feature_extractor.extract_state_features(
            context=req.context,
            turn_index=req.turn_index,
            intent_probs=intent_probs,
            permitted_actions=permitted_actions,
            last_turn_embedding=last_emb,
            sanitize_was_truncated=sanitize_result.was_truncated,
            sanitize_had_hostile=sanitize_result.had_hostile_markers,
        )

        # 7. Action prediction with soft policy boost re-weighting
        pred_action, action_conf, is_deviation = container.action_clf.predict(
            features=features,
            permitted_actions=permitted_actions,
        )

        # Agent action cue resolution for empty customer turns
        if last_was_empty:
            cue_action = resolve_empty_turn_action(req.context, pred_action)
            if cue_action != pred_action:
                pred_action = cue_action
                action_conf = 0.90

        # 8. Confidence calibration: calibrated from action prediction confidence
        raw_conf = action_conf
        calibrated_conf = container.calibrator.calibrate(raw_conf)

        # 9. needs_human decision (4 trigger conditions)
        needs_human, reason = decide_needs_human(
            calibrated_conf=calibrated_conf,
            is_deviation=is_deviation,
            action=pred_action,
            sanitize_result=sanitize_result,
            settings=settings,
        )

        # 10. Metrics & structured telemetry
        TRIAGE_REQUESTS_TOTAL.labels(status="success").inc()
        if needs_human:
            TRIAGE_NEEDS_HUMAN_TOTAL.labels(reason=reason).inc()

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "triage_completed",
            request_id=req.id,
            intent=pred_intent,
            action=pred_action,
            confidence=calibrated_conf,
            needs_human=needs_human,
            reason=reason,
            is_deviation=is_deviation,
            latency_ms=round(latency_ms, 3),
        )

        return TriageResponse(
            id=req.id,
            intent=pred_intent,
            action=pred_action,
            confidence=calibrated_conf,
            needs_human=needs_human,
            needs_human_reason=reason if needs_human else None,
            cost_usd=0.0,
            latency_ms=round(latency_ms, 3),
        )

    except Exception as exc:
        TRIAGE_ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
        TRIAGE_NEEDS_HUMAN_TOTAL.labels(reason="internal_error").inc()
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("unhandled_triage_error", request_id=req.id)
        # Bounded fallback: never crash or return 500
        return TriageResponse(
            id=req.id,
            intent="unknown",
            action="none",
            confidence=0.0,
            needs_human=True,
            needs_human_reason="internal_error",
            cost_usd=0.0,
            latency_ms=round(latency_ms, 3),
            error=ErrorDetail(code="internal_error", message=str(exc)),
        )
