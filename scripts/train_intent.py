#!/usr/bin/env python3
"""
scripts/train_intent.py
-----------------------
Phase 2 training script: A/B two sentence-transformer backbones with SetFit,
evaluate both on clean + noisy splits separately, pick the winner, save
artifacts, and log results to DECISIONS.md.

Usage
-----
  python scripts/train_intent.py \\
      --train   data/train.jsonl.gz \\
      --dev     data/dev.jsonl \\
      --noisy   data/dev_noisy.jsonl \\
      --out-dir artifacts/models \\
      [--max-train 20000]   # cap rows for fast iteration
      [--no-bge]            # skip the second backbone (time-saver after winner chosen)

Outputs
-------
  artifacts/models/intent/   -- winning SetFit model directory (load offline)
  artifacts/models/intent_classes.json  -- ordered class list
  results/intent_eval.json   -- both backbones' full eval metrics
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
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
    if limit:
        rows = rows[:limit]
    return rows


def format_context(row: dict, max_turns: int = 8) -> str:
    """Format the last N turns with speaker labels — same as inference path."""
    tail = (row.get("context") or [])[-max_turns:]
    return "\n".join(f"[{t['speaker']}] {t['text']}" for t in tail)


def build_xy(rows: list[dict]) -> tuple[list[str], list[str]]:
    texts = [format_context(r) for r in rows]
    labels = [r["intent"] for r in rows]
    return texts, labels


# ──────────────────────────────────────────────────────────────────────────────
# Backbone training & evaluation
# ──────────────────────────────────────────────────────────────────────────────


def sample_stratified(
    texts: list[str], labels: list[str], shots_per_class: int, seed: int = 42
) -> tuple[list[str], list[str]]:
    """
    Sample at most `shots_per_class` examples per label for contrastive fine-tuning.
    SetFit paper uses 8-shot by default; more gives diminishing returns on the
    sentence-encoder but massively increases pair-generation cost (O(n^2 * C)).
    The classification head is then trained on the FULL dataset.
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        buckets[lbl].append(i)
    sampled_idx = []
    for lbl, idxs in buckets.items():
        sampled_idx.extend(rng.sample(idxs, min(shots_per_class, len(idxs))))
    sampled_idx.sort()
    return [texts[i] for i in sampled_idx], [labels[i] for i in sampled_idx]


def train_and_eval(
    backbone: str,
    train_texts: list[str],
    train_labels: list[str],
    dev_texts: list[str],
    dev_labels: list[str],
    noisy_texts: list[str],
    noisy_labels: list[str],
    out_dir: Path,
    num_epochs: int = 1,
    shots_per_class: int = 16,
    max_steps: int = 300,
) -> dict:
    """
    Train a SetFit model with the given backbone, evaluate on clean + noisy splits.

    Strategy (per SetFit paper best practice):
    - Contrastive sentence-encoder fine-tuning on a STRATIFIED SUBSET
      (shots_per_class * n_classes rows) with bounded max_steps.
    - Classification head trained on ALL training data (full 20k rows).
    """
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    short_name = backbone.split("/")[-1]
    model_path = out_dir / short_name
    print(f"\n{'=' * 60}")
    print(f"  Training backbone: {backbone}")
    print(f"  Full train: {len(train_texts)} rows")
    print(
        f"  Contrastive subset: {shots_per_class} shots/class "
        f"(~{shots_per_class * len(set(train_labels))} rows), max_steps: {max_steps}"
    )
    print(f"  Dev: {len(dev_texts)}, Noisy: {len(noisy_texts)}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()

    # ── Step 1: Contrastive fine-tuning on stratified subset ──────────────
    subset_texts, subset_labels = sample_stratified(
        train_texts, train_labels, shots_per_class=shots_per_class
    )
    print(f"  Contrastive subset size: {len(subset_texts)} rows")

    model = SetFitModel.from_pretrained(backbone)

    contrastive_args = TrainingArguments(
        output_dir=str(model_path / "contrastive"),
        num_epochs=num_epochs,
        batch_size=32,
        max_steps=max_steps,
        seed=42,
    )

    contrastive_trainer = Trainer(
        model=model,
        args=contrastive_args,
        train_dataset=Dataset.from_dict({"text": subset_texts, "label": subset_labels}),
        metric="accuracy",
    )
    print("  [1/2] Contrastive fine-tuning...")
    contrastive_trainer.train_embeddings(
        *contrastive_trainer.dataset_to_parameters(contrastive_trainer.train_dataset),
        args=contrastive_args,
    )
    contrastive_elapsed = time.perf_counter() - t0
    print(f"  Contrastive done in {contrastive_elapsed:.1f}s")

    # ── Step 2: Train classification head on FULL data ────────────────────
    print("  [2/2] Training classification head on full data...")
    t1 = time.perf_counter()
    model.fit(train_texts, train_labels, num_epochs=1)
    head_elapsed = time.perf_counter() - t1
    print(f"  Head training done in {head_elapsed:.1f}s")
    train_elapsed = time.perf_counter() - t0

    # ── Evaluate on clean dev ──────────────────────────────────────────────
    t1 = time.perf_counter()
    pred_clean = model.predict(dev_texts)
    pred_clean = [str(p) for p in pred_clean]
    clean_acc = accuracy_score(dev_labels, pred_clean)
    clean_f1 = f1_score(dev_labels, pred_clean, average="macro", zero_division=0)
    print(f"  Clean  → acc={clean_acc:.4f}  macro-F1={clean_f1:.4f}")

    # ── Evaluate on noisy dev ─────────────────────────────────────────────
    pred_noisy = model.predict(noisy_texts)
    pred_noisy = [str(p) for p in pred_noisy]
    noisy_acc = accuracy_score(noisy_labels, pred_noisy)
    noisy_f1 = f1_score(noisy_labels, pred_noisy, average="macro", zero_division=0)
    gap = clean_acc - noisy_acc
    inf_elapsed = time.perf_counter() - t1
    print(f"  Noisy  → acc={noisy_acc:.4f}  macro-F1={noisy_f1:.4f}  gap={gap:.4f}")
    print(f"  Inference (both splits) in {inf_elapsed:.1f}s")

    # ── Probability distributions ─────────────────────────────────────────
    # SetFit .predict_proba returns a tensor/array of probabilities
    proba_clean = model.predict_proba(dev_texts)
    if hasattr(proba_clean, "numpy"):
        proba_clean = proba_clean.numpy()
    proba_clean = np.array(proba_clean)
    mean_max_prob_clean = float(proba_clean.max(axis=1).mean())

    proba_noisy = model.predict_proba(noisy_texts)
    if hasattr(proba_noisy, "numpy"):
        proba_noisy = proba_noisy.numpy()
    proba_noisy = np.array(proba_noisy)
    mean_max_prob_noisy = float(proba_noisy.max(axis=1).mean())

    # ── Save model ────────────────────────────────────────────────────────
    model.save_pretrained(str(model_path))
    print(f"  Model saved → {model_path}")

    return {
        "backbone": backbone,
        "short_name": short_name,
        "model_path": str(model_path),
        "train_rows": len(train_texts),
        "train_elapsed_s": round(train_elapsed, 1),
        "clean": {
            "accuracy": round(clean_acc, 4),
            "macro_f1": round(clean_f1, 4),
            "mean_max_prob": round(mean_max_prob_clean, 4),
        },
        "noisy": {
            "accuracy": round(noisy_acc, 4),
            "macro_f1": round(noisy_f1, 4),
            "mean_max_prob": round(mean_max_prob_noisy, 4),
        },
        "gap": round(gap, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# DECISIONS.md writer
# ──────────────────────────────────────────────────────────────────────────────

DECISIONS_MARKER_BEGIN = "<!-- phase2-intent:begin -->"
DECISIONS_MARKER_END = "<!-- phase2-intent:end -->"


def update_decisions_md(
    results: list[dict], winner: dict, decisions_path: Path
) -> None:
    text = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""

    # Strip previous block
    if DECISIONS_MARKER_BEGIN in text and DECISIONS_MARKER_END in text:
        pre = text[: text.index(DECISIONS_MARKER_BEGIN)]
        post = text[text.index(DECISIONS_MARKER_END) + len(DECISIONS_MARKER_END) :]
    else:
        pre = text.rstrip() + "\n\n"
        post = ""

    lines = [
        f"{DECISIONS_MARKER_BEGIN}",
        "",
        "## Phase 2 — Intent Classifier Backbone A/B Results",
        "",
        "### Evaluation Results",
        "",
        "| Backbone | Clean Acc | Clean F1 | Noisy Acc | Noisy F1 | Gap | Train time |",
        "|----------|-----------|----------|-----------|----------|-----|------------|",
    ]
    for r in results:
        winner_marker = (
            " ✅ **WINNER**" if r["short_name"] == winner["short_name"] else ""
        )
        lines.append(
            f"| `{r['backbone']}`{winner_marker} "
            f"| {r['clean']['accuracy']:.4f} "
            f"| {r['clean']['macro_f1']:.4f} "
            f"| {r['noisy']['accuracy']:.4f} "
            f"| {r['noisy']['macro_f1']:.4f} "
            f"| {r['gap']:.4f} "
            f"| {r['train_elapsed_s']}s |"
        )

    lines += [
        "",
        "### Choice: Backbone Decision",
        "",
        f"**Winner: `{winner['backbone']}`**",
        "",
        "**Reasoning:**",
        (
            f"- Clean accuracy: **{winner['clean']['accuracy']:.4f}** "
            f"(baseline was 0.3748 — +{winner['clean']['accuracy'] - 0.3748:.4f} lift)"
        ),
        (
            f"- Noisy accuracy: **{winner['noisy']['accuracy']:.4f}** "
            f"(baseline was 0.3000 — +{winner['noisy']['accuracy'] - 0.3000:.4f} lift)"
        ),
        (
            f"- Clean-to-noisy gap: **{winner['gap']:.4f}** "
            f"(qualification bar is ≤0.10 on *action* accuracy; this intent gap gives "
            f"headroom for the action layer)"
        ),
        "",
        "**Selection criterion (per 03_STACK.md §2 and 06_BUILD_PLAN_PHASES.md Phase 2):**",
        "Pick the backbone that best balances clean accuracy against the clean-to-noisy gap,",
        "*not* just raw clean accuracy. The paraphrase-multilingual model covers 50+ languages",
        "including Hindi, directly targeting the code-mixed Hinglish noisy slice. The gap is",
        "the deciding factor when clean accuracy is comparable.",
        "",
        f"Model artifact saved to: `{winner['model_path']}`",
        "",
        f"{DECISIONS_MARKER_END}",
    ]

    new_block = "\n".join(lines)
    decisions_path.write_text(pre + new_block + post, encoding="utf-8")
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
    ap.add_argument("--out-dir", default="artifacts/models")
    ap.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Cap training rows (useful for fast dev iterations)",
    )
    ap.add_argument(
        "--shots",
        type=int,
        default=16,
        help="Shots per class for contrastive fine-tuning (default: 16)",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Max contrastive steps (default: 300)",
    )
    ap.add_argument(
        "--no-bge",
        action="store_true",
        help="Skip BAAI/bge-small-en-v1.5 (faster; run after winner is known)",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--decisions", default="DECISIONS.md")
    ap.add_argument("--results-out", default="results/intent_eval.json")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_rows = load_jsonl(args.train, limit=args.max_train)
    dev_rows = load_jsonl(args.dev)
    noisy_rows = load_jsonl(args.noisy)

    train_texts, train_labels = build_xy(train_rows)
    dev_texts, dev_labels = build_xy(dev_rows)
    noisy_texts, noisy_labels = build_xy(noisy_rows)

    print(
        f"Train: {len(train_texts)}, Dev: {len(dev_texts)}, Noisy: {len(noisy_texts)}"
    )

    # Save class list derived from training data (preserves order for inference)
    all_classes = sorted(set(train_labels))
    classes_path = out_dir / "intent_classes.json"
    classes_path.write_text(json.dumps(all_classes, indent=2), encoding="utf-8")
    print(f"Saved {len(all_classes)} intent classes → {classes_path}")

    # ── Train backbones ────────────────────────────────────────────────────
    backbones = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    if not args.no_bge:
        backbones.append("BAAI/bge-small-en-v1.5")

    results = []
    for backbone in backbones:
        try:
            r = train_and_eval(
                backbone=backbone,
                train_texts=train_texts,
                train_labels=train_labels,
                dev_texts=dev_texts,
                dev_labels=dev_labels,
                noisy_texts=noisy_texts,
                noisy_labels=noisy_labels,
                out_dir=out_dir,
                num_epochs=args.epochs,
                shots_per_class=args.shots,
                max_steps=args.max_steps,
            )
            results.append(r)
        except Exception as exc:
            print(f"ERROR training {backbone}: {exc}")
            raise

    # ── Pick winner ────────────────────────────────────────────────────────
    # Score = clean_acc - 1.5 * gap  (penalise instability on noisy split)
    def score(r: dict) -> float:
        return r["clean"]["accuracy"] - 1.5 * r["gap"]

    winner = max(results, key=score)
    print(f"\n✅ Winner: {winner['backbone']}")
    print(
        f"   Clean acc={winner['clean']['accuracy']:.4f}  "
        f"Noisy acc={winner['noisy']['accuracy']:.4f}  "
        f"Gap={winner['gap']:.4f}"
    )

    # ── Copy winner to canonical path ──────────────────────────────────────
    import shutil

    canonical = out_dir / "intent"
    if canonical.exists():
        shutil.rmtree(canonical)
    shutil.copytree(winner["model_path"], canonical)
    winner["model_path"] = str(canonical)
    print(f"   Canonical model path: {canonical}")

    # ── Save eval JSON ─────────────────────────────────────────────────────
    eval_out = Path(args.results_out)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"winner": winner, "all_runs": results}
    eval_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved → {eval_out}")

    # ── Update DECISIONS.md ────────────────────────────────────────────────
    update_decisions_md(results, winner, Path(args.decisions))

    print("\n✅ Phase 2 training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
