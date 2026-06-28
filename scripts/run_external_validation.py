"""External validation runner: evaluate the APTOS-trained checkpoint on IDRiD.

This script does NOT retrain. It loads the trained EfficientNet-B0 from a
v1.0.0 / v1.1.0 run directory and evaluates it zero-shot on the IDRiD test set.

After running this, the produced predictions CSVs feed straight into the
existing OACSP analysis script via the standard CLI, enabling the v1.2.0
"OACSP transfer test" headline result.

Usage on Kaggle (after IDRiD dataset attached):

    python scripts/run_external_validation.py \\
        --run-dir experiments/runs/baseline_efficientnet_b0_aptos \\
        --idrid-root "/kaggle/input/idrid-dataset/B. Disease Grading" \\
        --output-dir experiments/runs/baseline_efficientnet_b0_aptos/idrid_external

Outputs:
    metrics.json
    predictions.csv                          per-sample softmax + label
    temperature_scaling_predictions.csv      calibrated using T from APTOS val
    oacsp_v1.1.0/oacsp_comparison.csv        OACSP transfer test (val=APTOS, test=IDRiD)

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
import torch
from torch.utils.data import DataLoader

from src.data.idrid import build_idrid_test_dataset
from src.data.transforms import build_eval_transform
from src.models.backbone import RetinalClassifier
from src.utils.config import load_yaml
from src.evaluation.metrics import compute_test_metrics


def predict(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Forward pass over the entire loader, return (probs, labels, id_codes)."""
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(batch["label"].numpy().tolist())
            all_ids.extend(batch["id_code"])
    return np.concatenate(all_probs, axis=0), np.array(all_labels, dtype=int), all_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="APTOS run directory containing best.pt and saved val_predictions")
    ap.add_argument("--idrid-root", type=Path, required=False, default=None,
                    help="IDRiD root dir. If omitted, auto-detects canonical or Kaggle-flat layout.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model
    cfg = load_yaml(args.config)
    model = RetinalClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=False,  # we're loading our own weights below
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt_path = args.run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {ckpt_path}")

    # --- Build IDRiD test loader
    eval_tf = build_eval_transform(cfg.data.image_size)
    idrid_ds = build_idrid_test_dataset(args.idrid_root, transform=eval_tf)
    idrid_loader = DataLoader(
        idrid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    print(f"IDRiD test set: {len(idrid_ds)} samples")

    # --- Predict
    print("Running inference on IDRiD...")
    probs, labels, id_codes = predict(model, idrid_loader, device)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # --- Save raw predictions
    pred_cols = {f"p_{i}": probs[:, i] for i in range(probs.shape[1])}
    pred_df = pd.DataFrame({
        "id_code": id_codes,
        "label": labels,
        "pred": preds,
        "confidence": confidences,
        **pred_cols,
    })
    pred_df.to_csv(args.output_dir / "predictions.csv", index=False)
    print(f"Wrote predictions.csv ({len(pred_df)} rows)")

    # --- Compute test metrics
    metrics = compute_test_metrics(
        labels=labels, preds=preds, probs=probs, n_classes=probs.shape[1]
    )
    metrics_out = {
        "n_test": int(len(labels)),
        **{k: float(v) for k, v in metrics.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print(f"\nIDRiD external validation metrics:")
    print(json.dumps(metrics_out, indent=2))

    # --- Apply APTOS-fitted temperature
    ts_json = args.run_dir / "temperature_scaling.json"
    if ts_json.exists():
        T = json.loads(ts_json.read_text()).get("temperature", 1.0)
        print(f"\nApplying APTOS-fitted T = {T:.4f} to IDRiD logits...")
        # Recompute calibrated probs from cached logits is awkward; instead
        # we re-run inference to capture logits, then divide.
        # Simpler: re-load and re-infer, dividing logits by T.
        all_logits = []
        model.eval()
        with torch.no_grad():
            for batch in idrid_loader:
                imgs = batch["image"].to(device)
                logits = model(imgs)
                all_logits.append((logits / T).softmax(dim=1).cpu().numpy())
        probs_cal = np.concatenate(all_logits, axis=0)
        preds_cal = probs_cal.argmax(axis=1)
        ts_df = pd.DataFrame({
            "id_code": id_codes,
            "label": labels,
            "pred": preds_cal,
            "confidence": probs_cal.max(axis=1),
            **{f"p_{i}": probs_cal[:, i] for i in range(probs_cal.shape[1])},
        })
        ts_df.to_csv(args.output_dir / "temperature_scaling_predictions.csv", index=False)
        print(f"Wrote temperature_scaling_predictions.csv")
    else:
        print(f"\n[WARN] No temperature_scaling.json found at {ts_json}. Skipping TS step.")

    print("\n[OK] External validation complete.")
    print(f"\nNext step: run OACSP transfer test with:")
    print(f"  python scripts/run_oacsp_analysis.py \\")
    print(f"      --val-predictions  {args.run_dir}/temperature_scaling_val_predictions.csv \\")
    print(f"      --test-predictions {args.output_dir}/temperature_scaling_predictions.csv \\")
    print(f"      --target-coverage  0.80 \\")
    print(f"      --output-dir       {args.output_dir}/oacsp_transfer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
