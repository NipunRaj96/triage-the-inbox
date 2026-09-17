#!/usr/bin/env python3
"""
scripts/analyze_failures.py
----------------------------
Join prediction file(s) against gold dev/noisy splits, bucket every error
into one of four categories, and print a Markdown failure-taxonomy table.

Also writes the table directly into EXPERIMENT_LOG.md under a stamped header.

Usage
-----
  python scripts/analyze_failures.py \\
      --pred      results/baseline_predictions.jsonl \\
      --gold      data/dev.jsonl \\
      --noisy     data/dev_noisy.jsonl \\
      [--hostile  data/hostile.jsonl] \\
      [--out-log  EXPERIMENT_LOG.md] \\
      [--tag      "Baseline v0 (bag-of-words nearest-centroid)"]

Error categories (from 06_BUILD_PLAN_PHASES.md Phase 1)
---------------------------------------------------------
  A  wrong_intent_right_action   — intent wrong, action happens to match gold
  B  right_intent_wrong_action   — intent correct, action wrong
  C  both_wrong_off_policy       — both wrong AND predicted action ∉ permitted
  D  both_wrong_on_policy        — both wrong but predicted action ∈ permitted

For noisy rows the breakdown is additionally split by noise_kind:
  typo | code_mixed | two_intents | empty | long | screenshot_description
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    return [
        json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()
    ]


def bucket(
    gold_intent: str,
    gold_action: str,
    pred_intent: str,
    pred_action: str,
    permitted_actions: list[str],
) -> str:
    """Return one of: A, B, C, D, or CORRECT."""
    intent_ok = pred_intent == gold_intent
    action_ok = pred_action == gold_action
    on_policy = pred_action in permitted_actions

    if intent_ok and action_ok:
        return "CORRECT"
    if not intent_ok and action_ok:
        return "A"  # wrong intent, right action (lucky)
    if intent_ok and not action_ok:
        return "B"  # right intent, wrong action
    # both wrong
    if not on_policy:
        return "C"  # both wrong + off-policy
    return "D"  # both wrong but on-policy


def pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num / denom * 100:.1f}%"


def fmt_row(*cols) -> str:
    return "| " + " | ".join(str(c) for c in cols) + " |"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    preds: dict[str, dict],
    gold_rows: list[dict],
    split_name: str,
    noise_col: bool = False,
) -> tuple[dict, str]:
    """
    Returns (stats_dict, markdown_table_str).
    stats_dict keys: total, correct, A, B, C, D, and noise breakdown if noise_col.
    """
    bucket_counts: Counter = Counter()
    noise_bucket: defaultdict[str, Counter] = defaultdict(Counter)

    n_total = 0
    for row in gold_rows:
        rid = str(row["id"])
        if rid not in preds:
            continue
        pred = preds[rid]
        b = bucket(
            gold_intent=row["intent"],
            gold_action=row["action"],
            pred_intent=pred.get("intent", ""),
            pred_action=pred.get("action", ""),
            permitted_actions=row.get("permitted_actions", []),
        )
        bucket_counts[b] += 1
        n_total += 1
        if noise_col:
            nk = row.get("noise_kind", "unknown")
            noise_bucket[nk][b] += 1

    correct = bucket_counts["CORRECT"]
    n_err = n_total - correct
    stats = dict(bucket_counts)
    stats["total"] = n_total
    stats["correct"] = correct
    stats["n_err"] = n_err

    # ---- main table ----
    lines = []
    lines.append(
        f"### `{split_name}` split — {n_total} rows, {correct} correct "
        f"({pct(correct, n_total)} accuracy)\n"
    )
    lines.append("| Category | Label | Count | % of errors | % of total |")
    lines.append("|----------|-------|------:|------------:|-----------:|")
    cat_defs = [
        ("A", "Wrong intent, right action (lucky match)"),
        ("B", "Right intent, wrong action"),
        ("C", "Both wrong + off-policy"),
        ("D", "Both wrong + on-policy"),
    ]
    for cat, label in cat_defs:
        cnt = bucket_counts.get(cat, 0)
        lines.append(fmt_row(cat, label, cnt, pct(cnt, n_err), pct(cnt, n_total)))
    lines.append(fmt_row("—", "**Total errors**", n_err, "100%", pct(n_err, n_total)))
    lines.append("")

    if noise_col and noise_bucket:
        lines.append("#### Noisy errors by `noise_kind`\n")
        all_nk = sorted(noise_bucket.keys())
        header_cats = ["A", "B", "C", "D", "CORRECT", "total"]
        lines.append("| noise_kind | " + " | ".join(header_cats) + " |")
        lines.append("|------------|" + "|".join(["------:"] * len(header_cats)) + "|")
        for nk in all_nk:
            counts = noise_bucket[nk]
            tot = sum(counts.values())
            row_vals = [counts.get(c, 0) for c in ["A", "B", "C", "D", "CORRECT"]] + [
                tot
            ]
            lines.append(
                "| " + nk + " | " + " | ".join(str(v) for v in row_vals) + " |"
            )
        lines.append("")

    return stats, "\n".join(lines)


# ---------------------------------------------------------------------------
# EXPERIMENT_LOG.md writer
# ---------------------------------------------------------------------------

MARKER_BEGIN = "<!-- analyze_failures:begin -->"
MARKER_END = "<!-- analyze_failures:end -->"


def update_experiment_log(log_path: Path, new_section: str) -> None:
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    # Strip previous run block if present
    if MARKER_BEGIN in existing and MARKER_END in existing:
        pre = existing[: existing.index(MARKER_BEGIN)]
        post = existing[existing.index(MARKER_END) + len(MARKER_END) :]
    else:
        pre = existing.rstrip() + "\n\n"
        post = ""

    stamped = (
        f"{MARKER_BEGIN}\n"
        f"## Failure Taxonomy Run — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{new_section}\n"
        f"{MARKER_END}\n"
    )
    log_path.write_text(pre + stamped + post, encoding="utf-8")
    print(f"Updated {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pred", required=True, help="Prediction JSONL file")
    ap.add_argument("--gold", required=True, help="Gold dev.jsonl")
    ap.add_argument("--noisy", default=None, help="Gold dev_noisy.jsonl")
    ap.add_argument(
        "--hostile", default=None, help="Gold hostile.jsonl (schema check only)"
    )
    ap.add_argument(
        "--out-log",
        default="EXPERIMENT_LOG.md",
        help="Path to EXPERIMENT_LOG.md to update",
    )
    ap.add_argument(
        "--tag", default="", help="Human-readable label for this run (e.g. model name)"
    )
    args = ap.parse_args(argv)

    # Load predictions keyed by id
    pred_rows = load_jsonl(args.pred)
    preds = {str(r["id"]): r for r in pred_rows}

    sections: list[str] = []

    if args.tag:
        sections.append(f"> **Run tag:** {args.tag}\n")

    # 1. Clean split
    gold_clean = load_jsonl(args.gold)
    stats_c, table_c = analyze(preds, gold_clean, "clean (dev.jsonl)")
    sections.append(table_c)

    # 2. Noisy split
    if args.noisy:
        gold_noisy = load_jsonl(args.noisy)
        stats_n, table_n = analyze(
            preds, gold_noisy, "noisy (dev_noisy.jsonl)", noise_col=True
        )
        sections.append(table_n)

    # 3. Summary delta
    sections.append("### Delta summary\n")
    if args.noisy:
        c_acc = stats_c["correct"] / max(stats_c["total"], 1)
        n_acc = stats_n["correct"] / max(stats_n["total"], 1)
        sections.append(
            f"| Split | Accuracy | Cat-B (intent✓ action✗) | Cat-C (both✗ off-policy) |\n"
            f"|-------|----------|-------------------------|---------------------------|\n"
            f"| clean | {c_acc:.4f} | {stats_c.get('B', 0)} ({pct(stats_c.get('B', 0), stats_c['n_err'])}) "
            f"| {stats_c.get('C', 0)} ({pct(stats_c.get('C', 0), stats_c['n_err'])}) |\n"
            f"| noisy | {n_acc:.4f} | {stats_n.get('B', 0)} ({pct(stats_n.get('B', 0), stats_n['n_err'])}) "
            f"| {stats_n.get('C', 0)} ({pct(stats_n.get('C', 0), stats_n['n_err'])}) |\n"
            f"| **gap** | {c_acc - n_acc:.4f} | — | — |\n"
        )

    full_section = "\n".join(sections)

    # Print to stdout too
    print(full_section)

    # Update EXPERIMENT_LOG.md
    log_path = Path(args.out_log)
    update_experiment_log(log_path, full_section)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
