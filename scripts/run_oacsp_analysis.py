"""Run the OACSP analysis on the v1.0.0 saved predictions.

This script is the v1.1.0 novel-piece entry point. It does NOT retrain anything.
It loads the per-sample softmax + label CSVs already saved by scripts/evaluate.py
from the v1.0.0 baseline run, then applies three abstention rules and prints
the comparison table that will go into the v1.1.0 report.

Inputs (must exist):
    predictions.csv      — from scripts/evaluate.py, has columns:
                           id, label, pred, prob_0, prob_1, prob_2, prob_3, prob_4
    (we re-use the same file for val and test by splitting based on the
     `split` column if present; otherwise pass --val-csv and --test-csv)

Usage:

    python scripts/run_oacsp_analysis.py \\
        --val-predictions outputs/temperature_scaling_predictions.csv \\
        --test-predictions outputs/temperature_scaling_predictions.csv \\
        --target-coverage 0.80 \\
        --output-dir outputs/oacsp_v1.1.0

Author: Khan, Muhammad Shahnawaz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.selective.class_conditional import (
    DEFAULT_CLASS_COST_MULTIPLIER,
    DEFAULT_TARGET_RECALL,
    apply_equalized_recall,
    apply_global_threshold,
    apply_ordinal_cost,
    build_comparison_table,
    calibrate_equalized_recall,
    calibrate_global_threshold,
    calibrate_ordinal_cost,
)


def _load_predictions_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions CSV and return (probs[N, 5], y[N]).

    Accepts either `prob_0`..`prob_4` or `p_0`..`p_4` as the softmax column
    naming convention. Requires a `label` column with the true class.
    """
    df = pd.read_csv(path)
    # Try both naming conventions
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_")])
    if not prob_cols:
        prob_cols = sorted([c for c in df.columns if c.startswith("p_") and c[2:].isdigit()])
    if not prob_cols:
        raise ValueError(
            f"No prob_* or p_* columns found in {path}. Got columns: {list(df.columns)}"
        )
    if len(prob_cols) != 5:
        raise ValueError(
            f"Expected 5 softmax columns for 5-class DR; got {len(prob_cols)}: {prob_cols}"
        )
    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {path}. Got: {list(df.columns)}")
    probs = df[prob_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return probs, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions", required=True, type=Path)
    ap.add_argument("--test-predictions", required=True, type=Path)
    ap.add_argument("--target-coverage", type=float, default=0.80)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--target-recall",
        type=str,
        default=None,
        help=(
            "Optional JSON dict of per-class target recall, e.g. "
            '\'{"0": 0.9, "1": 0.85, "2": 0.9, "3": 0.95, "4": 0.95}\'. '
            "Defaults to DEFAULT_TARGET_RECALL."
        ),
    )
    ap.add_argument(
        "--cost-multiplier",
        type=str,
        default=None,
        help=(
            "Optional JSON dict of per-class cost multiplier. "
            "Defaults to DEFAULT_CLASS_COST_MULTIPLIER."
        ),
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_recall = DEFAULT_TARGET_RECALL
    if args.target_recall:
        target_recall = {int(k): float(v) for k, v in json.loads(args.target_recall).items()}

    class_cost_multiplier = DEFAULT_CLASS_COST_MULTIPLIER
    if args.cost_multiplier:
        class_cost_multiplier = {
            int(k): float(v) for k, v in json.loads(args.cost_multiplier).items()
        }

    print(f"Loading val predictions from {args.val_predictions}", flush=True)
    val_probs, val_y = _load_predictions_csv(args.val_predictions)
    print(f"  -> {len(val_y)} samples", flush=True)

    print(f"Loading test predictions from {args.test_predictions}", flush=True)
    test_probs, test_y = _load_predictions_csv(args.test_predictions)
    print(f"  -> {len(test_y)} samples", flush=True)

    print()
    print("Target coverage:", args.target_coverage)
    print("Target recall per class:", target_recall)
    print("Cost multiplier per class:", class_cost_multiplier)
    print()

    # Headline comparison table for the paper
    df = build_comparison_table(
        val_probs=val_probs,
        val_y=val_y,
        test_probs=test_probs,
        test_y=test_y,
        target_coverage=args.target_coverage,
        target_recall=target_recall,
        class_cost_multiplier=class_cost_multiplier,
    )

    table_path = args.output_dir / "oacsp_comparison.csv"
    df.to_csv(table_path, index=False)

    # Pretty-print to stdout — this is what user pastes back to Claude
    print("=" * 78)
    print("OACSP COMPARISON TABLE (v1.1.0 headline result)")
    print("=" * 78)
    print(df.to_string(index=False))
    print()
    print(f"Saved CSV: {table_path}")

    # Individual full results
    print("\nCalibrating global baseline...", flush=True)
    tau_global = calibrate_global_threshold(val_probs, val_y, args.target_coverage)
    r_global = apply_global_threshold(test_probs, test_y, tau_global, class_cost_multiplier)

    print("Calibrating OACSP equalized-recall...", flush=True)
    tau_eq = calibrate_equalized_recall(val_probs, val_y, target_recall)
    r_eq = apply_equalized_recall(
        test_probs, test_y, tau_eq, target_recall, class_cost_multiplier
    )

    print("Calibrating OACSP ordinal-cost-weighted...", flush=True)
    tau_cost = calibrate_ordinal_cost(
        val_probs, val_y, args.target_coverage, class_cost_multiplier
    )
    r_cost = apply_ordinal_cost(test_probs, test_y, tau_cost, class_cost_multiplier)

    detailed = {
        "config": {
            "target_coverage": args.target_coverage,
            "target_recall": target_recall,
            "class_cost_multiplier": class_cost_multiplier,
            "val_n": int(len(val_y)),
            "test_n": int(len(test_y)),
        },
        "global_threshold_baseline": r_global.to_dict(),
        "oacsp_equalized_recall": r_eq.to_dict(),
        "oacsp_ordinal_cost": r_cost.to_dict(),
    }
    detailed_path = args.output_dir / "oacsp_detailed.json"
    detailed_path.write_text(json.dumps(detailed, indent=2, default=float))
    print(f"Saved detailed JSON: {detailed_path}")

    # Sanity check the headline claim
    print()
    print("=" * 78)
    print("HEADLINE CHECK — paste this section back to Claude")
    print("=" * 78)
    print(f"Coverage matched at {args.target_coverage}? "
          f"global={r_global.overall_coverage:.3f}  eq={r_eq.overall_coverage:.3f}  "
          f"cost={r_cost.overall_coverage:.3f}")
    print()
    print("Per-class abstention rate (LOWER is better for severe classes):")
    print(f"{'class':<8}{'global':>10}{'equalized':>12}{'ordinal_cost':>14}")
    for c in range(5):
        g = r_global.per_class_abstention_rate[c]
        e = r_eq.per_class_abstention_rate[c]
        o = r_cost.per_class_abstention_rate[c]
        print(f"{c:<8}{g:>10.3f}{e:>12.3f}{o:>14.3f}")
    print()
    print("Cost-weighted AURC (LOWER is better):")
    print(f"  global threshold:       {r_global.cost_weighted_aurc:.4f}")
    print(f"  OACSP equalized recall: {r_eq.cost_weighted_aurc:.4f}")
    print(f"  OACSP ordinal cost:     {r_cost.cost_weighted_aurc:.4f}")
    print()
    print("Overall selective QWK (HIGHER is better):")
    print(f"  global threshold:       {r_global.overall_selective_qwk:.4f}")
    print(f"  OACSP equalized recall: {r_eq.overall_selective_qwk:.4f}")
    print(f"  OACSP ordinal cost:     {r_cost.overall_selective_qwk:.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
