#!/usr/bin/env python3
"""Phase 5 – Evaluation harness.

Runs dev.jsonl + dev_noisy.jsonl + hostile.jsonl through the full ML
inference pipeline (no HTTP server needed) and calls the official OP-06
grader, producing:

    results/predictions.jsonl   – one pred per row, grader-compatible
    results/grader_report.json  – full grader JSON report
    results/latency_stats.json  – per-row latency breakdown

Usage::

    python scripts/run_eval.py [--data-dir PATH] [--out-dir PATH] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from project root or scripts/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# macOS ARM64: must be set BEFORE importing torch / lightgbm / tokenizers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.core.config import Settings
from app.core.needs_human import decide_needs_human
from app.models.container import ModelContainer
from app.sanitization import resolve_empty_turn_action, sanitize

# Import the official grader so we don't duplicate any metric logic
GRADER_PATH = (
    REPO_ROOT
    / "Deployment.inc-Hiring-Problems-main"
    / "references"
    / "OP-06"
    / "grader.py"
)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("grader", GRADER_PATH)
_grader_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_grader_mod)
grade = _grader_mod.grade
load_jsonl = _grader_mod.load_jsonl
by_id = _grader_mod.by_id
BARS = _grader_mod.BARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    return REPO_ROOT / "Deployment.inc-Hiring-Problems-main" / "references" / "OP-06"


def _predict_row(
    row: dict, container: ModelContainer, settings: Settings
) -> tuple[dict, float]:
    """Run a single row through the inference pipeline. Returns (pred_dict, latency_ms)."""
    t0 = time.perf_counter()
    rid = str(row["id"])

    try:
        context = row.get("context", [])
        turn_index = row.get("turn_index")

        # 1. Resolve best customer turn text (fall back past empty noisy turns)
        customer_turns = [t["text"] for t in context if t.get("speaker") == "customer"]
        non_empty_customer = [t for t in customer_turns if t.strip()]
        last_turn_text = non_empty_customer[-1] if non_empty_customer else ""
        last_was_empty = bool(customer_turns) and not customer_turns[-1].strip()
        san = sanitize(last_turn_text, max_tokens=settings.max_tokens_sanitize)

        # 2. Bail only on binary junk OR truly no customer text at all
        if san.is_binary_junk or (san.is_empty and not non_empty_customer):
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "id": rid,
                "intent": "unknown",
                "action": "none",
                "confidence": 0.0,
                "needs_human": True,
                "cost_usd": 0.0,
                "latency_ms": round(latency_ms, 3),
            }, latency_ms

        # 3. Build Turn objects for feature extractor
        from app.core.schemas import Turn

        turns_obj = [Turn(speaker=t["speaker"], text=t["text"]) for t in context[-8:]]

        # 4. Intent classification
        model_input = container.feature_extractor.format_context_for_embedding(
            turns_obj, max_turns=8
        )
        pred_intent, intent_probs = container.intent_clf.predict(model_input)

        # 5. Policy lookup
        permitted_actions = container.policy_engine.get_permitted_actions(pred_intent)

        # 6. Feature extraction for action classifier
        # When last turn was empty we fell back to a prior customer turn; use the
        # full formatted context for embedding to give richer signal (diagnostic
        # shows +2 noisy correct rows vs isolated prior-turn embedding).
        if last_was_empty:
            emb_text = model_input
        else:
            clean_text = san.clean_text if san.clean_text else last_turn_text
            emb_text = clean_text
        last_emb = container.intent_clf.encode(emb_text)
        features = container.feature_extractor.extract_state_features(
            context=turns_obj,
            turn_index=turn_index,
            intent_probs=intent_probs,
            permitted_actions=permitted_actions,
            last_turn_embedding=last_emb,
            sanitize_was_truncated=san.was_truncated,
            sanitize_had_hostile=san.had_hostile_markers,
        )

        # 7. Action prediction
        pred_action, action_conf, is_deviation = container.action_clf.predict(
            features=features,
            permitted_actions=permitted_actions,
        )

        # Agent action cue resolution for empty customer turns
        if last_was_empty:
            cue_action = resolve_empty_turn_action(context, pred_action)
            if cue_action != pred_action:
                pred_action = cue_action
                action_conf = 0.90

        # 8. Calibration
        calibrated_conf = container.calibrator.calibrate(action_conf)

        # 9. needs_human
        needs_human, _ = decide_needs_human(
            calibrated_conf=calibrated_conf,
            is_deviation=is_deviation,
            action=pred_action,
            sanitize_result=san,
            settings=settings,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "id": rid,
            "intent": pred_intent,
            "action": pred_action,
            "confidence": calibrated_conf,
            "needs_human": needs_human,
            "cost_usd": 0.0,
            "latency_ms": round(latency_ms, 3),
        }, latency_ms

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  [WARN] row {rid} raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return {
            "id": rid,
            "intent": "unknown",
            "action": "none",
            "confidence": 0.0,
            "needs_human": True,
            "cost_usd": 0.0,
            "latency_ms": round(latency_ms, 3),
        }, latency_ms


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * pct / 100), len(sorted_v) - 1)
    return round(sorted_v[idx], 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing dev.jsonl, dev_noisy.jsonl, hostile.jsonl, schema.json",
    )
    ap.add_argument(
        "--out-dir", default="results", help="Output directory (default: results/)"
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 60)
        print("OP-06 Phase 5 — Evaluation Harness")
        print("=" * 60)

    # 1. Load models
    if not args.quiet:
        print("\n[1/5] Loading models …", flush=True)
    settings = Settings.load_with_overrides(
        REPO_ROOT / "config" / "policy_overrides.yaml"
    )
    settings.intent_model_dir = REPO_ROOT / settings.intent_model_dir
    settings.intent_classes_path = REPO_ROOT / settings.intent_classes_path
    settings.action_model_path = REPO_ROOT / settings.action_model_path
    settings.action_classes_path = REPO_ROOT / settings.action_classes_path
    settings.action_columns_path = REPO_ROOT / settings.action_columns_path
    settings.calibrator_path = REPO_ROOT / settings.calibrator_path
    settings.guidelines_path = REPO_ROOT / settings.guidelines_path
    settings.policy_overrides_path = REPO_ROOT / settings.policy_overrides_path

    container = ModelContainer(settings)
    container.load()
    if not args.quiet:
        print("    OK Models loaded")

    # 2. Load eval data
    if not args.quiet:
        print("\n[2/5] Loading eval data …", flush=True)

    dev_path = data_dir / "dev.jsonl"
    noisy_path = data_dir / "dev_noisy.jsonl"
    hostile_path = data_dir / "hostile.jsonl"
    schema_path = data_dir / "schema.json"

    gold_rows = load_jsonl(str(dev_path), "gold")
    noisy_rows = (
        load_jsonl(str(noisy_path), "noisy gold") if noisy_path.exists() else []
    )
    hostile_rows = (
        load_jsonl(str(hostile_path), "hostile") if hostile_path.exists() else []
    )
    schema = (
        json.loads(schema_path.read_text(encoding="utf-8"))
        if schema_path.exists()
        else None
    )

    all_rows = gold_rows + noisy_rows + hostile_rows
    if not args.quiet:
        print(
            f"    OK {len(gold_rows)} clean + {len(noisy_rows)} noisy + {len(hostile_rows)} hostile = {len(all_rows)} total"
        )

    # 3. Run inference
    if not args.quiet:
        print(f"\n[3/5] Running inference on {len(all_rows)} rows …", flush=True)

    predictions: list[dict] = []
    latencies: list[float] = []
    t_total_start = time.perf_counter()

    for i, row in enumerate(all_rows):
        pred, lat_ms = _predict_row(row, container, settings)
        predictions.append(pred)
        latencies.append(lat_ms)
        if not args.quiet and (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_total_start
            rps = (i + 1) / elapsed
            print(f"    {i + 1}/{len(all_rows)} rows  ({rps:.1f} rows/s)", flush=True)

    total_elapsed = time.perf_counter() - t_total_start
    if not args.quiet:
        print(
            f"    OK Done in {total_elapsed:.1f}s  ({len(all_rows) / total_elapsed:.1f} rows/s)"
        )

    # 4. Write predictions JSONL (combined and per-split)
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in predictions)

    # Split predictions for SUBMISSION_SCHEMA.md per-problem requirements
    dev_pred_path = out_dir / "dev_predictions.jsonl"
    noisy_pred_path = out_dir / "noisy_predictions.jsonl"
    hostile_pred_path = out_dir / "hostile_results.jsonl"

    n_gold = len(gold_rows)
    n_noisy = len(noisy_rows)
    dev_preds = predictions[:n_gold]
    noisy_preds = predictions[n_gold : n_gold + n_noisy]
    hostile_preds = predictions[n_gold + n_noisy :]

    with open(dev_pred_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in dev_preds)

    with open(noisy_pred_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in noisy_preds)

    with open(hostile_pred_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in hostile_preds)

    # Keep raw predictions in results/raw/
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "dev-final.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in predictions)

    if not args.quiet:
        print(
            f"\n[4/5] Predictions written to {pred_path} (+ per-split files in {out_dir})"
        )

    # 5. Grade & save
    if not args.quiet:
        print("\n[5/5] Grading …", flush=True)

    preds_by_id = by_id(predictions)
    report = grade(gold_rows, noisy_rows, hostile_rows, preds_by_id, schema)

    report_path = out_dir / "grader_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    lat_stats = {
        "n": len(latencies),
        "mean_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
        "total_elapsed_s": round(total_elapsed, 3),
        "throughput_rows_per_s": round(len(latencies) / total_elapsed, 2)
        if total_elapsed > 0
        else 0.0,
    }
    lat_path = out_dir / "latency_stats.json"
    with open(lat_path, "w", encoding="utf-8") as fh:
        json.dump(lat_stats, fh, indent=2)
        fh.write("\n")

    # Write latency_bench.json per SUBMISSION_SCHEMA.md if not already present with load benchmark
    bench_file = out_dir / "latency_bench.json"
    if not bench_file.exists() or bench_file.stat().st_size < 1000:
        bench_data = {
            "sample_count": len(predictions),
            "concurrency": 1,
            "hardware": "CPU-only (MacBook Air ARM64)",
            "mean_latency_ms": lat_stats["mean_ms"],
            "p50_latency_ms": lat_stats["p50_ms"],
            "p95_latency_ms": lat_stats["p95_ms"],
            "p99_latency_ms": lat_stats["p99_ms"],
            "cost_per_message_usd": 0.0,
            "throughput_rps": lat_stats["throughput_rows_per_s"],
            "durations_ms": [round(l, 2) for l in latencies[:100]],
        }
        with open(bench_file, "w", encoding="utf-8") as fh:
            json.dump(bench_data, fh, indent=2)
            fh.write("\n")

    # Read p95 latency under load from latency_bench.json if available
    claimed_p95_s = round(lat_stats["p95_ms"] / 1000.0, 4)
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as fh:
                loaded_bench = json.load(fh)
                if "p95_latency_ms" in loaded_bench:
                    claimed_p95_s = round(loaded_bench["p95_latency_ms"] / 1000.0, 4)
        except Exception:  # noqa: BLE001, S110
            pass

    # Write results/manifest.json per SUBMISSION_SCHEMA.md
    manifest_data = {
        "claimed": {
            "intent_accuracy": report["intent"]["accuracy_all"],
            "macro_f1": report["intent"]["macro_f1_all"],
            "action_accuracy": report["action"]["accuracy_all"],
            "noisy_accuracy": report["action"]["accuracy_noisy"],
            "schema_valid_pct": report["schema"]["valid_rate_all_inputs"] * 100.0,
            "top70_error_ratio": report["confidence"]["top70_error_ratio"],
            "cost_per_message_usd": 0.0,
            "latency_p95_s": claimed_p95_s,
        },
        "runs": [
            {
                "run_id": "op06-prod-final",
                "seed": 42,
                "model": "SetFit(MiniLM-L12-v2)+LightGBM",
                "status": "completed",
                "raw_path": "results/raw/dev-final.jsonl",
            }
        ],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
        fh.write("\n")

    # Print summary
    if not args.quiet:
        b = report["bars"]
        yn = lambda x: "PASS" if x else ("N/A" if x is None else "FAIL")
        print("\n" + "=" * 60)
        print("GRADER RESULTS")
        print("=" * 60)
        print(
            f"  {'Intent accuracy (clean+noisy)':<38}{report['intent']['accuracy_all']:>9.4f}"
            f"   >= {BARS['intent_accuracy']:<6}  {yn(b['intent_accuracy_all']['pass'])}"
        )
        print(
            f"  {'Action accuracy (clean+noisy)':<38}{report['action']['accuracy_all']:>9.4f}"
            f"   >= {BARS['action_accuracy']:<6}  {yn(b['action_accuracy_all']['pass'])}"
        )
        print(
            f"  {'Intent macro-F1':<38}{report['intent']['macro_f1_all']:>9.4f}"
            f"   >= {BARS['macro_f1']:<6}  {yn(b['macro_f1_all']['pass'])}"
        )
        gap = report["action"]["noisy_gap"]
        gap_str = f"{gap:>9.4f}" if gap is not None else "       N/A"
        print(
            f"  {'Clean-to-noisy action gap':<38}{gap_str}"
            f"   <= {BARS['noisy_gap']:<6}  {yn(b['noisy_gap']['pass'])}"
        )
        print(
            f"  {'Schema valid (incl. hostile)':<38}"
            f"{report['schema']['valid_rate_all_inputs'] * 100:>8.2f}%"
            f"   == {BARS['schema_valid_pct']:<6}  {yn(b['schema_valid_pct']['pass'])}"
        )
        pol = report["policy"]
        print(
            f"  {'Policy violations':<38}{pol['violation_rate']!s:>9}"
            f"   (gold itself: {pol['gold_violation_rate']})"
        )
        ratio = report["confidence"]["top70_error_ratio"]
        print(
            f"  {'Confidence error ratio (top 70%)':<38}{ratio!s:>9}"
            f"   <= {BARS['top70_error_ratio']:<6}  {yn(b['top70_error_ratio']['pass'])}"
        )
        print(
            f"\n  Latency   mean {lat_stats['mean_ms']:.1f}ms  p50 {lat_stats['p50_ms']:.1f}ms  "
            f"p95 {lat_stats['p95_ms']:.1f}ms  p99 {lat_stats['p99_ms']:.1f}ms"
        )
        print(f"  Throughput {lat_stats['throughput_rows_per_s']:.1f} rows/s")

        all_pass = all(v["pass"] for v in b.values() if v["pass"] is not None)
        print("\n" + "=" * 60)
        if all_pass:
            print("ALL BARS PASSED")
        else:
            failed = [k for k, v in b.items() if v["pass"] is False]
            print(f"BARS FAILED: {', '.join(failed)}")
        print("=" * 60)
        print(f"\nReports written to:\n  {report_path}\n  {lat_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
