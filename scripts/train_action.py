#!/usr/bin/env python3
"""
scripts/train_action.py
------------------------
Phase 3 training script: trains LightGBM action classifier with the feature
vector from 04_LOGIC_FLOW.md §6, evaluates at multiple POLICY_BOOST values,
and writes the trade-off table to DECISIONS.md.

Requires: intent model artifacts already trained (scripts/train_intent.py).

Usage
-----
  python scripts/train_action.py \\
      --train   data/train.jsonl.gz \\
      --dev     data/dev.jsonl \\
      --noisy   data/dev_noisy.jsonl \\
      --intent-model   artifacts/models/intent \\
      --intent-classes artifacts/models/intent_classes.json \\
      --out-dir artifacts/models \\
      [--no-embeddings]   # skip embedding features for fast iteration

Outputs
-------
  artifacts/models/action/action_model.pkl
  artifacts/models/action/action_classes.json
  artifacts/models/action/feature_columns.json
  results/action_eval.json
"""

from __future__ import annotations

# ── Thread-pool conflict fix ───────────────────────────────────────────────────
# Must be set BEFORE any HuggingFace / tokenizers import to prevent Rayon
# thread pool from spawning — which then crashes OpenMP (libomp.dylib) in
# LightGBM on macOS ARM64 (SIGSEGV in __kmp_allocate_task_team).
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    else:
        rows = [
            json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    return rows[:limit] if limit else rows


def format_context(row: dict, max_turns: int = 8) -> str:
    tail = (row.get("context") or [])[-max_turns:]
    return "\n".join(f"[{t['speaker']}] {t['text']}" for t in tail)


# ──────────────────────────────────────────────────────────────────────────────
# Policy-aware prediction (for eval at different POLICY_BOOST values)
# ──────────────────────────────────────────────────────────────────────────────


def policy_rescore(
    proba: np.ndarray,  # shape (n_classes,)
    action_classes: list[str],
    permitted_actions: list[str],
    policy_boost: float,
) -> str:
    permitted_set = set(permitted_actions)
    adjusted = np.copy(proba)
    for i, act in enumerate(action_classes):
        if act in permitted_set:
            adjusted[i] *= 1.0 + policy_boost
    return action_classes[int(np.argmax(adjusted))]


# ──────────────────────────────────────────────────────────────────────────────
# DECISIONS.md writer
# ──────────────────────────────────────────────────────────────────────────────

MARKER_BEGIN = "<!-- phase3-action:begin -->"
MARKER_END = "<!-- phase3-action:end -->"


def update_decisions_md(
    boost_table: list[dict],
    chosen_boost: float,
    clean_acc: float,
    noisy_acc: float,
    decisions_path: Path,
) -> None:
    text = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""

    if MARKER_BEGIN in text and MARKER_END in text:
        pre = text[: text.index(MARKER_BEGIN)]
        post = text[text.index(MARKER_END) + len(MARKER_END) :]
    else:
        pre = text.rstrip() + "\n\n"
        post = ""

    lines = [
        MARKER_BEGIN,
        "",
        "## Phase 3 — Action Classifier & Policy Boost Trade-off",
        "",
        "### LightGBM Action Classifier Results",
        "",
        "| POLICY_BOOST | Clean Action Acc | Noisy Action Acc | Policy Violation Rate |",
        "|-------------|-----------------|-----------------|----------------------|",
    ]
    for row in boost_table:
        marker = " ✅ **chosen**" if abs(row["boost"] - chosen_boost) < 0.001 else ""
        lines.append(
            f"| {row['boost']}{marker} "
            f"| {row['clean_acc']:.4f} "
            f"| {row['noisy_acc']:.4f} "
            f"| {row['violation_rate']:.4f} |"
        )

    lines += [
        "",
        f"**Chosen POLICY_BOOST: {chosen_boost}**",
        "",
        "**Reasoning (per 06_BUILD_PLAN_PHASES.md Phase 3):**",
        "- Gold policy violation rate is 8.83% (`gold_violation_rate` from grader).",
        "- We target a violation rate close to gold's natural rate rather than",
        "  driving it to zero (which would cost us ~9% action accuracy on the",
        "  genuinely off-policy gold labels).",
        f"- POLICY_BOOST={chosen_boost} achieves the best action accuracy while",
        "  keeping the violation rate within ~2× of the gold rate.",
        "- See 04_LOGIC_FLOW.md §7: hard masking would guarantee 0% recall on",
        "  the off-policy gold fraction — soft boost is the correct design.",
        "",
        "**Final scores (chosen boost):**",
        f"- Clean action accuracy:  {clean_acc:.4f}",
        f"- Noisy action accuracy:  {noisy_acc:.4f}",
        f"- Gap:                    {clean_acc - noisy_acc:.4f}",
        "",
        MARKER_END,
    ]

    decisions_path.write_text(pre + "\n".join(lines) + post, encoding="utf-8")
    print(f"Updated {decisions_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--intent-model", default="artifacts/models/intent")
    ap.add_argument("--intent-classes", default="artifacts/models/intent_classes.json")
    ap.add_argument("--out-dir", default="artifacts/models")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding features (faster, slightly lower accuracy)",
    )
    ap.add_argument("--decisions", default="DECISIONS.md")
    ap.add_argument("--results-out", default="results/action_eval.json")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) / "action"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load intent model ─────────────────────────────────────────────────
    print("Loading intent model for embeddings + intent probs...")
    from app.core.schemas import Turn
    from app.models.features import ACTION_VOCABULARY, DialogueFeatureExtractor
    from app.models.intent import IntentClassifier

    intent_clf = IntentClassifier(
        model_dir=args.intent_model,
        classes_path=args.intent_classes,
    )
    intent_clf.load()
    intent_classes = intent_clf.classes_

    # Move to CPU for large-batch offline encoding — avoids MPS hang on 20k rows
    intent_clf._model.to("cpu")
    sentence_model = intent_clf._model.model_body if not args.no_embeddings else None

    feature_extractor = DialogueFeatureExtractor(
        action_vocabulary=ACTION_VOCABULARY,
        intent_classes=intent_classes,
        embedding_dim=384,
    )

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    train_rows = load_jsonl(args.train, limit=args.max_train)
    dev_rows = load_jsonl(args.dev)
    noisy_rows = load_jsonl(args.noisy)
    print(f"Train: {len(train_rows)}, Dev: {len(dev_rows)}, Noisy: {len(noisy_rows)}")

    # ── Build feature matrix ──────────────────────────────────────────────
    def build_features(
        rows: list[dict], split_name: str
    ) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
        """Returns X, y_action, permitted_lists, raw_proba_matrix"""
        print(f"\nBuilding features for {split_name} ({len(rows)} rows)...")
        t0 = time.perf_counter()

        # Batch encode all contexts
        context_texts = [format_context(r) for r in rows]
        last_customer_texts = []
        for r in rows:
            ctx = r.get("context") or []
            last_cust = next(
                (t["text"] for t in reversed(ctx) if t["speaker"] == "customer"), ""
            )
            last_customer_texts.append(last_cust)

        # Get intent probs for all rows (batch, CPU, chunked for memory)
        print(f"  Computing intent probabilities for {len(rows)} rows...")
        chunk = 512
        proba_chunks = []
        for start in range(0, len(context_texts), chunk):
            batch = context_texts[start : start + chunk]
            p = intent_clf._model.predict_proba(batch, as_numpy=True)
            proba_chunks.append(np.asarray(p, dtype=np.float32))
            if start % 2048 == 0:
                print(f"    intent proba: {start}/{len(context_texts)}")
        intent_proba_batch = np.concatenate(proba_chunks, axis=0)

        # Get sentence embeddings for last customer turn (CPU, chunked)
        emb_batch = None
        if sentence_model is not None:
            print(f"  Computing sentence embeddings for {len(rows)} rows...")
            emb_batch = sentence_model.encode(
                last_customer_texts,
                batch_size=256,
                show_progress_bar=True,
                device="cpu",
            )
            emb_batch = np.asarray(emb_batch, dtype=np.float32)

        # Build feature dicts
        X_list = []
        y_list = []
        perms = []
        model_labels = intent_clf._get_model_labels()

        for i, row in enumerate(rows):
            ctx_turns = [
                Turn(speaker=t["speaker"], text=t["text"])
                for t in (row.get("context") or [])
            ]
            iprobs = {
                lbl: float(intent_proba_batch[i, j])
                for j, lbl in enumerate(model_labels)
            }
            emb = emb_batch[i] if emb_batch is not None else None

            feat = feature_extractor.extract_state_features(
                context=ctx_turns,
                turn_index=row.get("turn_index"),
                intent_probs=iprobs,
                permitted_actions=row.get("permitted_actions", []),
                last_turn_embedding=emb,
            )
            X_list.append(feat)
            y_list.append(row["action"])
            perms.append(row.get("permitted_actions", []))

        # Get column order from first sample
        col_names = list(X_list[0].keys())
        X = np.array(
            [[feat.get(col, 0.0) for col in col_names] for feat in X_list],
            dtype=np.float32,
        )

        elapsed = time.perf_counter() - t0
        print(f"  Feature matrix: {X.shape}, elapsed: {elapsed:.1f}s")
        return X, y_list, perms, intent_proba_batch, col_names

    X_train, y_train, _perms_train, _iproba_train, col_names = build_features(
        train_rows, "train"
    )
    X_dev, y_dev, perms_dev, _iproba_dev, _ = build_features(dev_rows, "dev")
    X_noisy, y_noisy, perms_noisy, _iproba_noisy, _ = build_features(
        noisy_rows, "noisy"
    )

    # Save feature column list (sidecar for inference alignment)
    col_path = out_dir / "feature_columns.json"
    col_path.write_text(json.dumps(col_names, indent=2), encoding="utf-8")

    # ── Encode action labels ──────────────────────────────────────────────
    action_classes = sorted(set(y_train))
    action_idx = {a: i for i, a in enumerate(action_classes)}
    y_train_enc = np.array([action_idx.get(a, 0) for a in y_train])

    classes_path = out_dir / "action_classes.json"
    classes_path.write_text(json.dumps(action_classes, indent=2), encoding="utf-8")
    print(f"\nAction classes ({len(action_classes)}): {action_classes[:5]}...")

    # ── Train LightGBM ────────────────────────────────────────────────────
    import lightgbm as lgb

    print("\nTraining LightGBM action classifier...")
    t0 = time.perf_counter()

    dtrain = lgb.Dataset(X_train, label=y_train_enc)

    params = {
        "objective": "multiclass",
        "num_class": len(action_classes),
        "metric": "multi_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        # n_jobs=1: OpenMP (libomp.dylib) crashes with SIGSEGV on macOS ARM64
        # when HuggingFace tokenizers' Rayon thread pools are alive in the same
        # process.  Single-threaded LGB is fast enough for this dataset size.
        "n_jobs": 1,
        "seed": 42,
        "verbose": -1,
    }

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=params.pop("n_estimators"),
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(50)],
    )

    train_elapsed = time.perf_counter() - t0
    print(f"Training done in {train_elapsed:.1f}s")

    # Save model
    model_path = out_dir / "action_model.pkl"
    booster.save_model(str(model_path))
    print(f"Model saved → {model_path}")

    # ── Evaluate at multiple POLICY_BOOST values ──────────────────────────
    print("\nEvaluating at multiple POLICY_BOOST values...")
    boost_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

    dev_proba = booster.predict(X_dev)  # (n_dev, n_classes)
    noisy_proba = booster.predict(X_noisy)

    boost_table = []
    for boost in boost_values:
        # Clean predictions
        preds_clean = [
            policy_rescore(dev_proba[i], action_classes, perms_dev[i], boost)
            for i in range(len(y_dev))
        ]
        clean_acc = accuracy_score(y_dev, preds_clean)

        # Violation rate
        violation_rate = sum(
            1 for pred, perm in zip(preds_clean, perms_dev) if pred not in perm
        ) / len(preds_clean)

        # Noisy predictions
        preds_noisy = [
            policy_rescore(noisy_proba[i], action_classes, perms_noisy[i], boost)
            for i in range(len(y_noisy))
        ]
        noisy_acc = accuracy_score(y_noisy, preds_noisy)

        boost_table.append(
            {
                "boost": boost,
                "clean_acc": round(clean_acc, 4),
                "noisy_acc": round(noisy_acc, 4),
                "violation_rate": round(violation_rate, 4),
                "gap": round(clean_acc - noisy_acc, 4),
            }
        )

        bar_pass = "✅" if clean_acc >= 0.55 else "❌"
        print(
            f"  BOOST={boost:.2f}: clean={clean_acc:.4f}{bar_pass}  "
            f"noisy={noisy_acc:.4f}  violation={violation_rate:.4f}"
        )

    # Choose boost: best action accuracy that keeps violation rate ≤ 2× gold (0.0883)
    MAX_VIOLATION = 0.20
    eligible = [r for r in boost_table if r["violation_rate"] <= MAX_VIOLATION]
    chosen = max(eligible, key=lambda r: r["clean_acc"]) if eligible else boost_table[0]
    chosen_boost = chosen["boost"]
    print(f"\n✅ Chosen POLICY_BOOST: {chosen_boost}")
    print(
        f"   Clean acc={chosen['clean_acc']:.4f}  "
        f"Noisy acc={chosen['noisy_acc']:.4f}  "
        f"Violation={chosen['violation_rate']:.4f}"
    )

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "chosen_policy_boost": chosen_boost,
        "train_elapsed_s": round(train_elapsed, 1),
        "feature_dim": X_train.shape[1],
        "boost_table": boost_table,
    }
    results_path = Path(args.results_out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved → {results_path}")

    # ── Update DECISIONS.md ───────────────────────────────────────────────
    update_decisions_md(
        boost_table=boost_table,
        chosen_boost=chosen_boost,
        clean_acc=chosen["clean_acc"],
        noisy_acc=chosen["noisy_acc"],
        decisions_path=Path(args.decisions),
    )

    print("\n✅ Phase 3 training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
