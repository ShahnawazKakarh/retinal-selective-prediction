"""Fit temperature on val set, evaluate on test set.

Loads a baseline checkpoint, collects logits on val + test, fits T on val,
applies it to test, and writes:

  temperature_scaling.json         — fitted T, NLL before/after, ECE before/after
  temperature_scaling_predictions.csv — id_code, label, pred, conf, p_0..p_4 (calibrated)
  temperature_scaling_selective.json  — AURC + selective accuracy on calibrated probs

Usage:
    python scripts/run_temperature_scaling.py \
        --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
        --splits-dir data/splits/aptos2019 \
        --images-dir /path/to/aptos/train_images
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.aptos import APTOS2019Dataset
from src.data.splits import load_splits
from src.data.transforms import build_eval_transform
from src.evaluation.metrics import expected_calibration_error
from src.models.backbone import RetinalClassifier
from src.selective.risk_coverage import (
    aurc,
    excess_aurc,
    selective_accuracy_at_coverage,
)
from src.uncertainty.temperature_scaling import TemperatureScaler, collect_logits
from src.utils.config import load_config
from src.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.run_dir / "config.yaml")
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    _train_df, val_df, test_df = load_splits(args.splits_dir)
    val_ds = APTOS2019Dataset(
        val_df, args.images_dir,
        transform=build_eval_transform(cfg.data.image_size),
    )
    test_ds = APTOS2019Dataset(
        test_df, args.images_dir,
        transform=build_eval_transform(cfg.data.image_size),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers, pin_memory=True)

    # ---- Model + checkpoint ----
    model = RetinalClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=False,
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # ---- Collect logits ----
    print("Collecting val logits...")
    val_logits, val_labels, val_ids = collect_logits(model, val_loader, device)
    print("Collecting test logits...")
    test_logits, test_labels, test_ids = collect_logits(model, test_loader, device)

    # ---- Fit temperature on val ----
    scaler = TemperatureScaler()
    fit_info = scaler.fit(val_logits, val_labels, device=device)
    print(f"Fitted T = {fit_info['temperature']:.4f}  "
          f"(val NLL {fit_info['nll_before']:.4f} → {fit_info['nll_after']:.4f})")

    # ---- Apply to test ----
    probs_uncal = torch.softmax(torch.from_numpy(test_logits).float(), dim=1).numpy()
    probs_cal = scaler.calibrate_probs(test_logits)

    ece_before = expected_calibration_error(probs_uncal, test_labels)
    ece_after = expected_calibration_error(probs_cal, test_labels)

    preds = probs_cal.argmax(axis=1)
    correct = (preds == test_labels).astype(np.int32)

    # ---- Persist diagnostics ----
    info = {
        **fit_info,
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "test_accuracy": float(correct.mean()),
    }
    with open(args.run_dir / "temperature_scaling.json", "w") as f:
        json.dump(info, f, indent=2)

    # ---- Persist per-sample predictions ----
    pred_df = pd.DataFrame({
        "id_code": test_ids,
        "label": test_labels,
        "pred": preds,
        "confidence": probs_cal.max(axis=1),
        **{f"p_{i}": probs_cal[:, i] for i in range(probs_cal.shape[1])},
    })
    pred_df.to_csv(args.run_dir / "temperature_scaling_predictions.csv", index=False)

    # ---- Also persist CALIBRATED val predictions (needed for downstream OACSP calibration) ----
    probs_val_cal = scaler.calibrate_probs(val_logits)
    val_preds_cal = probs_val_cal.argmax(axis=1)
    val_pred_df = pd.DataFrame({
        "id_code": val_ids,
        "label": val_labels,
        "pred": val_preds_cal,
        "confidence": probs_val_cal.max(axis=1),
        **{f"p_{i}": probs_val_cal[:, i] for i in range(probs_val_cal.shape[1])},
    })
    val_pred_df.to_csv(args.run_dir / "temperature_scaling_val_predictions.csv", index=False)

    # ---- Selective summary on calibrated probs ----
    # Single signal: negative max-confidence (since T-scaling doesn't introduce
    # a separate epistemic measure; it just recalibrates the existing one).
    eps = 1e-12
    pred_entropy = -np.sum(probs_cal * np.log(probs_cal + eps), axis=1)
    signals = {
        "neg_max_confidence": -probs_cal.max(axis=1),
        "predictive_entropy": pred_entropy,
    }
    target_coverages = [0.70, 0.80, 0.90, 0.95]
    summary = {
        "n_test": int(len(test_labels)),
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
    with open(args.run_dir / "temperature_scaling_selective.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print ----
    print(f"\nECE: {ece_before:.4f} → {ece_after:.4f}  (Δ {ece_after - ece_before:+.4f})")
    print(f"Test accuracy: {info['test_accuracy']:.4f}")
    for name, entry in summary["signals"].items():
        acc_str = "  ".join(
            f"{int(c * 100):>3}%={entry['selective_accuracy'][f'coverage_{int(c * 100)}']['selective_accuracy']:.3f}"
            for c in target_coverages
        )
        print(f"  {name:<22} AURC={entry['aurc']:.4f}  excess={entry['excess_aurc']:.4f}  {acc_str}")

    print(f"\nWrote: {args.run_dir / 'temperature_scaling.json'}")
    print(f"       {args.run_dir / 'temperature_scaling_predictions.csv'}")
    print(f"       {args.run_dir / 'temperature_scaling_val_predictions.csv'}")
    print(f"       {args.run_dir / 'temperature_scaling_selective.json'}")


if __name__ == "__main__":
    main()
