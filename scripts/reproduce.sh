#!/usr/bin/env bash
# scripts/reproduce.sh
# OP-06 Triage: full end-to-end reproduction script.
#
# Usage:
#   bash scripts/reproduce.sh
#
# What it does:
#   1. Activates the virtual environment
#   2. Starts the FastAPI service in the background
#   3. Polls GET /healthz until the service is ready (models loaded)
#   4. Runs the offline evaluation harness (run_eval.py)
#   5. Reports grader results and stops the service
#
# Requirements:
#   - Python 3.11+
#   - .venv/ created via: python -m venv .venv && .venv/bin/pip install -r requirements.txt
#   - Trained model artifacts in artifacts/models/
#   - .env file (see .env.example)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Environment ──────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Load .env if present (PORT, LOG_LEVEL, etc.)
if [[ -f ".env" ]]; then
  set -o allexport
  source .env
  set +o allexport
fi

PORT="${PORT:-8000}"
VENV_PYTHON=".venv/bin/python"

if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "ERROR: .venv not found. Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# ── Start service ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "OP-06 Triage — Reproduction"
echo "============================================================"
echo ""
echo "[1/4] Starting FastAPI service on port $PORT …"
PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --log-level warning \
  &
SERVICE_PID=$!
trap "echo ''; echo 'Stopping service (PID $SERVICE_PID)…'; kill $SERVICE_PID 2>/dev/null || true" EXIT

# ── Wait for readiness ────────────────────────────────────────────────────────
echo "[2/4] Waiting for GET /healthz → 200 …"
MAX_WAIT=120
WAITED=0
until curl -sf "http://localhost:${PORT}/healthz" > /dev/null 2>&1; do
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "ERROR: Service did not become ready in ${MAX_WAIT}s"
    exit 1
  fi
  sleep 2
  WAITED=$((WAITED + 2))
done
echo "    OK Service ready (${WAITED}s startup)"

# Verify healthz response shape
HEALTH_RESP=$(curl -sf "http://localhost:${PORT}/healthz")
echo "    Response: $HEALTH_RESP"

# ── Offline evaluation ────────────────────────────────────────────────────────
echo ""
echo "[3/4] Running evaluation harness (dev + noisy + hostile) …"
PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" scripts/run_eval.py

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Grader report written to results/grader_report.json"
if command -v jq > /dev/null 2>&1; then
  echo ""
  jq '.bars' results/grader_report.json
fi

echo ""
echo "============================================================"
echo "Reproduction complete.  See results/ for full artifacts."
echo "============================================================"
