"""EDL inference + selective summary on the internal test set.

Usage:
    python scripts/run_evidential.py \
        --run-dir experiments/runs/evidential_efficientnet_b0_aptos \
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
from src.selective.risk_coverage import (
    aurc,
    excess_aurc,
    selective_accuracy_at_coverage,
)
from src.uncertainty.evidential import EvidentialClassifier, evidential_predict
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

    _train_df, _val_df, test_df = load_splits(args.splits_dir)
    test_ds = APTOS2019Dataset(test_df, args.images_dir,
                               transform=build_eval_transform(cfg.data.image_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers, pin_memory=True)

    model = EvidentialClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=False,
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint epoch={ckpt.get('epoch')} val_acc={ckpt.get('val_accuracy'):.4f}")

    out = evidential_predict(model, test_loader, device, cfg.model.num_classes)
    probs = out["probs_mean"]
    labels = out["labels"]
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.int32)

    pred_df = pd.DataFrame({
        "id_code": out["id_codes"],
        "label": labels,
        "pred_edl": preds,
        "conf_edl": probs.max(axis=1),
        "vacuity": out["vacuity"],
        **{f"p_{i}": probs[:, i] for i in range(probs.shape[1])},
        **{f"alpha_{i}": out["alpha"][:, i] for i in range(out["alpha"].shape[1])},
    })
    pred_df.to_csv(args.run_dir / "evidential_predictions.csv", index=False)

    eps = 1e-12
    pred_entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    signals = {
        "neg_max_confidence": -probs.max(axis=1),
        "predictive_entropy": pred_entropy,
        "vacuity": out["vacuity"],
    }
    target_coverages = [0.70, 0.80, 0.90, 0.95]
    summary = {
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
    with open(args.run_dir / "evidential_selective.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall test accuracy: {summary['overall_accuracy']:.4f}")
    for name, entry in summary["signals"].items():
        acc_str = "  ".join(
            f"{int(c * 100):>3}%={entry['selective_accuracy'][f'coverage_{int(c * 100)}']['selective_accuracy']:.3f}"
            for c in target_coverages
        )
        print(f"  {name:<22} AURC={entry['aurc']:.4f}  excess={entry['excess_aurc']:.4f}  {acc_str}")


if __name__ == "__main__":
    main()
