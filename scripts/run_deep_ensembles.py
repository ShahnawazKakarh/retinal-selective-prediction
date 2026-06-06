"""Run deep ensemble inference using M pre-trained checkpoints.

This script does **not** train the ensemble — train M members yourself by
running scripts/train.py M times with different seeds and different
experiment.name values. Then point this script at the M run directories.

Example workflow:
    # Train M=5 members (loop or Kaggle notebooks)
    for seed in 1 2 3 4 5; do
        python scripts/train.py --config configs/ensemble_member.yaml \
            --splits-dir data/splits/aptos2019 \
            --images-dir /path/to/images \
            # then edit config seed + name between runs, or use --override
    done

    # Aggregate
    python scripts/run_deep_ensembles.py \
        --member-dirs experiments/runs/ensemble_seed{1,2,3,4,5} \
        --splits-dir data/splits/aptos2019 \
        --images-dir /path/to/images \
        --out-dir experiments/runs/ensemble_aggregate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.aptos import APTOS2019Dataset
from src.data.splits import load_splits
from src.data.transforms import build_eval_transform
from src.selective.risk_coverage import (
    aurc,
    excess_aurc,
    selective_accuracy_at_coverage,
)
from src.uncertainty.deep_ensembles import ensemble_predict
from src.utils.config import load_config
from src.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-dirs", required=True, nargs="+", type=Path,
                    help="Run directories of M trained ensemble members.")
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write ensemble outputs.")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if len(args.member_dirs) < 2:
        raise SystemExit(f"Need at least 2 members; got {len(args.member_dirs)}")

    # Use the first member's config as the reference (all members share splits + backbone)
    cfg = load_config(args.member_dirs[0] / "config.yaml")
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Ensemble M = {len(args.member_dirs)}")

    # ---- Data ----
    _train_df, _val_df, test_df = load_splits(args.splits_dir)
    test_ds = APTOS2019Dataset(
        test_df, args.images_dir,
        transform=build_eval_transform(cfg.data.image_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    # ---- Aggregate predictions ----
    ckpt_paths = [d / "best.pt" for d in args.member_dirs]
    for p in ckpt_paths:
        if not p.exists():
            raise SystemExit(f"Missing checkpoint: {p}")

    out = ensemble_predict(
        ckpt_paths,
        test_loader,
        device,
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )

    probs = out["probs_mean"]
    labels = out["labels"]
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.int32)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-sample predictions ----
    pred_df = pd.DataFrame({
        "id_code": out["id_codes"],
        "label": labels,
        "pred_ens": preds,
        "conf_ens": probs.max(axis=1),
        "predictive_entropy": out["predictive_entropy"],
        "expected_entropy": out["expected_entropy"],
        "mutual_information": out["mutual_information"],
        **{f"p_{i}": probs[:, i] for i in range(probs.shape[1])},
        **{f"var_{i}": out["probs_var"][:, i] for i in range(probs.shape[1])},
    })
    pred_df.to_csv(args.out_dir / "ensemble_predictions.csv", index=False)

    # ---- Selective prediction summary ----
    # Variance is summarized as the mean over classes (a scalar per sample)
    mean_var = out["probs_var"].mean(axis=1)
    signals = {
        "neg_max_confidence": -probs.max(axis=1),
        "predictive_entropy": out["predictive_entropy"],
        "expected_entropy": out["expected_entropy"],
        "mutual_information": out["mutual_information"],
        "mean_class_variance": mean_var,
    }
    target_coverages = [0.70, 0.80, 0.90, 0.95]

    selective_summary = {
        "ensemble_size_M": len(ckpt_paths),
        "member_dirs": [str(d) for d in args.member_dirs],
        "n_test": int(len(labels)),
        "overall_accuracy": float(correct.mean()),
        "signals": {
            name: {
                "aurc": aurc(correct, unc),
                "excess_aurc": excess_aurc(correct, unc),
                "selective_accuracy": {
                    f"coverage_{int(c * 100)}": selective_accuracy_at_coverage(correct, unc, c)
                    for c in target_coverages
                },
            }
            for name, unc in signals.items()
        },
    }
    with open(args.out_dir / "ensemble_selective.json", "w") as f:
        json.dump(selective_summary, f, indent=2)

    # ---- Print ----
    print(f"\nOverall test accuracy: {selective_summary['overall_accuracy']:.4f}")
    print(f"{'signal':<22} {'AURC':>8} {'excess':>8}   selective acc @ coverage")
    for name, entry in selective_summary["signals"].items():
        acc_str = "  ".join(
            f"{int(c * 100):>3}%={entry['selective_accuracy'][f'coverage_{int(c * 100)}']['selective_accuracy']:.3f}"
            for c in target_coverages
        )
        print(f"{name:<22} {entry['aurc']:>8.4f} {entry['excess_aurc']:>8.4f}   {acc_str}")

    print(f"\nWrote: {args.out_dir / 'ensemble_predictions.csv'}")
    print(f"       {args.out_dir / 'ensemble_selective.json'}")


if __name__ == "__main__":
    main()
