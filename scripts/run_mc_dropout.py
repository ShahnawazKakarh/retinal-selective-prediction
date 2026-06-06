"""MC Dropout uncertainty pass on the test set.

Loads a checkpoint trained by scripts/train.py, runs T stochastic forward
passes with dropout active, and writes:

  mc_dropout_predictions.csv  — id_code, label, pred_mc, conf_mc,
                                 p_0..p_4, predictive_entropy,
                                 expected_entropy, mutual_information
  mc_dropout_selective.json   — AURC + excess-AURC for each uncertainty signal,
                                 plus selective accuracy at coverage 0.7/0.8/0.9/0.95.

Usage:
    python scripts/run_mc_dropout.py \
        --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
        --splits-dir data/splits/aptos2019 \
        --images-dir /path/to/aptos/train_images \
        --n-samples 30
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
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.utils.config import load_config
from src.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=30,
                    help="Number of MC Dropout forward passes (T).")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.run_dir / "config.yaml")
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # ---- Model + checkpoint ----
    model = RetinalClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=False,
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')} "
          f"(val_kappa={ckpt.get('val_kappa'):.4f})")

    # ---- MC Dropout ----
    out = mc_dropout_predict(model, test_loader, device, n_samples=args.n_samples)

    probs = out["probs_mean"]
    labels = out["labels"]
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.int32)

    # ---- Persist per-sample predictions ----
    pred_df = pd.DataFrame({
        "id_code": out["id_codes"],
        "label": labels,
        "pred_mc": preds,
        "conf_mc": probs.max(axis=1),
        "predictive_entropy": out["predictive_entropy"],
        "expected_entropy": out["expected_entropy"],
        "mutual_information": out["mutual_information"],
        **{f"p_{i}": probs[:, i] for i in range(probs.shape[1])},
    })
    pred_df.to_csv(args.run_dir / "mc_dropout_predictions.csv", index=False)

    # ---- Selective prediction summary ----
    # Three candidate uncertainty signals; lower = more confident.
    # We negate confidence so that "higher = more uncertain" holds.
    signals = {
        "neg_max_confidence": -probs.max(axis=1),
        "predictive_entropy": out["predictive_entropy"],
        "expected_entropy": out["expected_entropy"],
        "mutual_information": out["mutual_information"],
    }
    target_coverages = [0.70, 0.80, 0.90, 0.95]

    selective_summary: dict = {
        "n_samples_T": args.n_samples,
        "n_test": int(len(labels)),
        "overall_accuracy": float(correct.mean()),
        "signals": {},
    }
    for name, unc in signals.items():
        entry = {
            "aurc": aurc(correct, unc),
            "excess_aurc": excess_aurc(correct, unc),
            "selective_accuracy": {
                f"coverage_{int(c * 100)}": selective_accuracy_at_coverage(
                    correct, unc, c,
                )
                for c in target_coverages
            },
        }
        selective_summary["signals"][name] = entry

    with open(args.run_dir / "mc_dropout_selective.json", "w") as f:
        json.dump(selective_summary, f, indent=2)

    # ---- Print a compact summary ----
    print(f"\nOverall test accuracy: {selective_summary['overall_accuracy']:.4f}")
    print(f"{'signal':<22} {'AURC':>8} {'excess':>8}   selective acc @ coverage")
    for name, entry in selective_summary["signals"].items():
        acc_str = "  ".join(
            f"{int(c * 100):>3}%={entry['selective_accuracy'][f'coverage_{int(c * 100)}']['selective_accuracy']:.3f}"
            for c in target_coverages
        )
        print(f"{name:<22} {entry['aurc']:>8.4f} {entry['excess_aurc']:>8.4f}   {acc_str}")

    print(f"\nWrote: {args.run_dir / 'mc_dropout_predictions.csv'}")
    print(f"       {args.run_dir / 'mc_dropout_selective.json'}")


if __name__ == "__main__":
    main()
