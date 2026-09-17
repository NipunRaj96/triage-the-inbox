"""Prometheus metrics for operational health monitoring."""

from prometheus_client import Counter, Histogram

TRIAGE_REQUESTS_TOTAL = Counter(
    "triage_requests_total",
    "Total inbound triage requests processed",
    ["status"],
)

TRIAGE_ERRORS_TOTAL = Counter(
    "triage_errors_total",
    "Total triage errors and unhandled exceptions",
    ["error_type"],
)

TRIAGE_NEEDS_HUMAN_TOTAL = Counter(
    "triage_needs_human_total",
    "Total requests routed to a human agent",
    ["reason"],
)

TRIAGE_LATENCY_SECONDS = Histogram(
    "triage_latency_seconds",
    "End-to-end request latency in seconds",
    ["stage"],
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0],
)

TRIAGE_HOSTILE_FLAGS_TOTAL = Counter(
    "triage_hostile_flags_total",
    "Hostile and anomalous inputs detected by sanitization layer",
    ["flag_type"],
)
