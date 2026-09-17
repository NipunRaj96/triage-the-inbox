"""Locust load test scenario for OP-06 Triage Service.

Hits POST /triage with a representative cyclic sample of dev.jsonl rows.
Measures end-to-end response latency under stated concurrency.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from locust import HttpUser, between, task

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEV_DATA = REPO_ROOT / "data" / "dev.jsonl"
if not DEV_DATA.exists():
    DEV_DATA = (
        REPO_ROOT
        / "Deployment.inc-Hiring-Problems-main"
        / "references"
        / "OP-06"
        / "dev.jsonl"
    )


def _load_sample_payloads() -> list[dict]:
    payloads = []
    if not DEV_DATA.exists():
        return payloads
    with open(DEV_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            payloads.append(
                {
                    "id": row.get("id"),
                    "context": row.get("context", []),
                    "turn_index": row.get("turn_index"),
                    "dialogue_id": row.get("dialogue_id"),
                }
            )
    return payloads


PAYLOADS = _load_sample_payloads()
PAYLOAD_CYCLE = itertools.cycle(PAYLOADS) if PAYLOADS else None


class TriageUser(HttpUser):
    wait_time = between(0.01, 0.05)  # slight realistic think-time between requests

    @task
    def triage_request(self):
        if not PAYLOAD_CYCLE:
            return
        payload = next(PAYLOAD_CYCLE)
        with self.client.post(
            "/triage",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/triage",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "intent" in data and "action" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure(f"Invalid schema: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
