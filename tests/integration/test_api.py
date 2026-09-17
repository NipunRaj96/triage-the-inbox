"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from app.api.service import app

client = TestClient(app)


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_triage_valid_request():
    payload = {
        "id": "test-001",
        "context": [{"speaker": "customer", "text": "hello i want a refund"}],
        "turn_index": 1,
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-001"
    assert "intent" in data
    assert "action" in data
    assert "confidence" in data
    assert "needs_human" in data


def test_triage_malformed_json_returns_400():
    # Missing required 'context' field
    payload = {"id": "test-bad"}
    response = client.post("/triage", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_triage_empty_text_returns_error_envelope():
    payload = {
        "id": "hostile-empty",
        "context": [{"speaker": "customer", "text": ""}],
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["needs_human"] is True
    assert data["intent"] == "unknown"
    assert data["action"] == "none"
    assert data["needs_human_reason"] == "unhandleable_input"


def test_triage_binary_junk_returns_error_envelope():
    payload = {
        "id": "hostile-binary",
        "context": [
            {"speaker": "customer", "text": "\x00\x01\x02\x03\xff\xfe\xfd\x00\x00"}
        ],
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["needs_human"] is True
    assert data["intent"] == "unknown"
    assert data["action"] == "none"
    assert data["needs_human_reason"] == "unhandleable_input"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "triage_requests_total" in response.text


def test_triage_confidence_and_latency_ranges():
    payload = {
        "id": "test-ranges",
        "context": [{"speaker": "customer", "text": "I want to track my order"}],
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] is not None
    assert data["latency_ms"] >= 0.0
