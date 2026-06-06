"""One-shot script: make stratified APTOS train/val/test splits and save CSVs.

Run once. CSVs are committed to git so every later run uses identical splits.

Usage:
    python scripts/prepare_aptos_splits.py \
        --train-csv /path/to/aptos/train.csv \
        --output-dir data/splits/aptos2019
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.splits import make_stratified_splits, save_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, type=Path,
                    help="Path to APTOS train.csv (id_code, diagnosis).")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write train.csv, val.csv, test.csv.")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    if not {"id_code", "diagnosis"}.issubset(df.columns):
        raise SystemExit("train.csv must have columns: id_code, diagnosis")

    train_df, val_df, test_df = make_stratified_splits(
        df, label_col="diagnosis",
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    save_splits(train_df, val_df, test_df, args.output_dir)

    print(f"Wrote splits to {args.output_dir}")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:<5} n={len(d):>5}   "
              f"class counts: {d['diagnosis'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
