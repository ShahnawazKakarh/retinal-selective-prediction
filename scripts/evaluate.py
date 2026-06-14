"""Evaluation entry point.

Loads a checkpoint, runs on the internal test set, writes:
  * metrics.json   — kappa, accuracy, per-class P/R/F1, ECE, NLL, Brier
  * predictions.csv — id_code, label, pred, conf, p_0..p_4 (for downstream selective work)
  * confusion_matrix.png
  * reliability_diagram.png

Usage:
    python scripts/evaluate.py \
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
)
from torch.utils.data import DataLoader

from src.data.aptos import APTOS2019Dataset
from src.data.splits import load_splits
from src.data.transforms import build_eval_transform
from src.evaluation.calibration import plot_reliability_diagram
from src.evaluation.metrics import expected_calibration_error
from src.models.backbone import RetinalClassifier
from src.utils.config import load_config
from src.utils.seed import seed_everything


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (probs, labels, id_codes)."""
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["id_code"])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def plot_confusion(cm: np.ndarray, class_names: tuple[str, ...], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (internal test)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Run directory containing config.yaml and best.pt.")
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
    eval_tf = build_eval_transform(cfg.data.image_size)
    val_ds = APTOS2019Dataset(val_df, args.images_dir, transform=eval_tf)
    test_ds = APTOS2019Dataset(test_df, args.images_dir, transform=eval_tf)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    # ---- Model ----
    model = RetinalClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=False,  # we're loading our own weights
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')} "
          f"(val_kappa={ckpt.get('val_kappa'):.4f})")

    # ---- Predict on val (for downstream selective methods that need calibration) ----
    val_probs, val_labels, val_id_codes = predict(model, val_loader, device)
    val_preds = val_probs.argmax(axis=1)
    val_confidences = val_probs.max(axis=1)

    # ---- Predict on test ----
    probs, labels, id_codes = predict(model, test_loader, device)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # ---- Metrics ----
    metrics: dict = {
        "n_test": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "kappa_quadratic": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "nll": float(log_loss(labels, probs, labels=list(range(cfg.data.num_classes)))),
        "brier": float(np.mean(np.sum((probs - np.eye(cfg.data.num_classes)[labels]) ** 2, axis=1))),
        "ece_15bins": expected_calibration_error(probs, labels, n_bins=15),
        "per_class": classification_report(
            labels, preds,
            labels=list(range(cfg.data.num_classes)),
            target_names=list(APTOS2019Dataset.CLASS_NAMES),
            output_dict=True, zero_division=0,
        ),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))

    # ---- Persist ----
    out_dir = args.run_dir
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_cols = {f"p_{i}": probs[:, i] for i in range(cfg.data.num_classes)}
    pd.DataFrame({
        "id_code": id_codes,
        "label": labels,
        "pred": preds,
        "confidence": confidences,
        **pred_cols,
    }).to_csv(out_dir / "predictions.csv", index=False)

    val_pred_cols = {f"p_{i}": val_probs[:, i] for i in range(cfg.data.num_classes)}
    pd.DataFrame({
        "id_code": val_id_codes,
        "label": val_labels,
        "pred": val_preds,
        "confidence": val_confidences,
        **val_pred_cols,
    }).to_csv(out_dir / "val_predictions.csv", index=False)

    cm = confusion_matrix(labels, preds, labels=list(range(cfg.data.num_classes)))
    plot_confusion(cm, APTOS2019Dataset.CLASS_NAMES, out_dir / "confusion_matrix.png")
    plot_reliability_diagram(
        probs, labels,
        out_dir / "reliability_diagram.png",
        title=f"Reliability — {cfg.experiment.name}",
    )

    print(f"\nWrote: {out_dir / 'metrics.json'}")
    print(f"       {out_dir / 'predictions.csv'}")
    print(f"       {out_dir / 'val_predictions.csv'}")
    print(f"       {out_dir / 'confusion_matrix.png'}")
    print(f"       {out_dir / 'reliability_diagram.png'}")


if __name__ == "__main__":
    main()
