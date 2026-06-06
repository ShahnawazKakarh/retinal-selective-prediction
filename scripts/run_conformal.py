"""Conformal prediction — calibrate on val, evaluate on test.

Uses the baseline checkpoint's softmax probabilities. Compares SplitConformal
(score = 1 - p_y) with APS (Adaptive Prediction Sets).

Outputs:
  conformal_predictions.csv  — id_code, label, top1_pred, set_size_split, set_size_aps,
                               in_set_split, in_set_aps, p_0..p_4
  conformal_summary.json     — q_hat, empirical coverage, avg set size,
                               selective metrics using set_size as the uncertainty signal.

Usage:
    python scripts/run_conformal.py \
        --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
        --splits-dir data/splits/aptos2019 \
        --images-dir /path/to/aptos/train_images \
        --alpha 0.10
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
from src.models.backbone import RetinalClassifier
from src.selective.risk_coverage import (
    aurc,
    excess_aurc,
    selective_accuracy_at_coverage,
)
from src.uncertainty.conformal import APS, SplitConformal
from src.utils.config import load_config
from src.utils.seed import seed_everything


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["id_code"])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Miscoverage rate (target coverage = 1 - alpha).")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.run_dir / "config.yaml")
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  alpha = {args.alpha}")

    _train_df, val_df, test_df = load_splits(args.splits_dir)
    val_ds = APTOS2019Dataset(val_df, args.images_dir,
                              transform=build_eval_transform(cfg.data.image_size))
    test_ds = APTOS2019Dataset(test_df, args.images_dir,
                               transform=build_eval_transform(cfg.data.image_size))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers, pin_memory=True)

    model = RetinalClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=False,
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    val_probs, val_labels, _ = collect_probs(model, val_loader, device)
    test_probs, test_labels, test_ids = collect_probs(model, test_loader, device)

    split_cp = SplitConformal(alpha=args.alpha).calibrate(val_probs, val_labels)
    aps_cp = APS(alpha=args.alpha).calibrate(val_probs, val_labels)

    sets_split = split_cp.predict_sets(test_probs)
    sets_aps = aps_cp.predict_sets(test_probs)
    sizes_split = np.array([len(s) for s in sets_split], dtype=np.int32)
    sizes_aps = np.array([len(s) for s in sets_aps], dtype=np.int32)
    in_split = np.array([int(y in s) for s, y in zip(sets_split, test_labels)])
    in_aps = np.array([int(y in s) for s, y in zip(sets_aps, test_labels)])

    top1 = test_probs.argmax(axis=1)
    correct = (top1 == test_labels).astype(np.int32)

    # Selective signal: larger set ⇒ more uncertain
    eps = 1e-12
    pred_entropy = -np.sum(test_probs * np.log(test_probs + eps), axis=1)
    signals = {
        "set_size_split": sizes_split.astype(np.float64),
        "set_size_aps": sizes_aps.astype(np.float64),
        "neg_max_confidence": -test_probs.max(axis=1),
        "predictive_entropy": pred_entropy,
    }
    target_coverages = [0.70, 0.80, 0.90, 0.95]
    summary = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_val": int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "split_conformal": {
            "q_hat": split_cp.q_hat,
            "empirical_coverage": float(in_split.mean()),
            "avg_set_size": float(sizes_split.mean()),
            "median_set_size": int(np.median(sizes_split)),
            "singleton_fraction": float((sizes_split == 1).mean()),
        },
        "aps": {
            "q_hat": aps_cp.q_hat,
            "empirical_coverage": float(in_aps.mean()),
            "avg_set_size": float(sizes_aps.mean()),
            "median_set_size": int(np.median(sizes_aps)),
            "singleton_fraction": float((sizes_aps == 1).mean()),
        },
        "top1_accuracy": float(correct.mean()),
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

    pred_df = pd.DataFrame({
        "id_code": test_ids,
        "label": test_labels,
        "top1_pred": top1,
        "set_size_split": sizes_split,
        "set_size_aps": sizes_aps,
        "in_set_split": in_split,
        "in_set_aps": in_aps,
        **{f"p_{i}": test_probs[:, i] for i in range(test_probs.shape[1])},
    })
    pred_df.to_csv(args.run_dir / "conformal_predictions.csv", index=False)
    with open(args.run_dir / "conformal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTarget coverage: {1 - args.alpha:.2f}")
    print(f"SplitConformal: emp. coverage={summary['split_conformal']['empirical_coverage']:.3f}  "
          f"avg |C|={summary['split_conformal']['avg_set_size']:.2f}  "
          f"singleton frac={summary['split_conformal']['singleton_fraction']:.3f}")
    print(f"APS:            emp. coverage={summary['aps']['empirical_coverage']:.3f}  "
          f"avg |C|={summary['aps']['avg_set_size']:.2f}  "
          f"singleton frac={summary['aps']['singleton_fraction']:.3f}")
    print(f"\nTop-1 accuracy: {summary['top1_accuracy']:.4f}")
    for name, entry in summary["signals"].items():
        acc_str = "  ".join(
            f"{int(c * 100):>3}%={entry['selective_accuracy'][f'coverage_{int(c * 100)}']['selective_accuracy']:.3f}"
            for c in target_coverages
        )
        print(f"  {name:<22} AURC={entry['aurc']:.4f}  excess={entry['excess_aurc']:.4f}  {acc_str}")


if __name__ == "__main__":
    main()
