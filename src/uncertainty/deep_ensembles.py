"""Deep Ensembles for predictive uncertainty.

Lakshminarayanan, Pritzel & Blundell (NeurIPS 2017): train M independent
networks with different random seeds (different init + different shuffle
order). At inference, average their softmax outputs. The disagreement
across members is an epistemic-uncertainty signal.

This module is the *inference-time* glue — it expects M checkpoints already
trained by scripts/train.py with different seeds. The training side is
handled by scripts/run_deep_ensembles.py which launches the M runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.backbone import RetinalClassifier


@torch.no_grad()
def ensemble_predict(
    checkpoint_paths: list[str | Path],
    loader: DataLoader,
    device: torch.device,
    *,
    backbone: str,
    num_classes: int,
    dropout: float,
) -> dict[str, np.ndarray]:
    """Run inference with M ensemble members and aggregate.

    Args:
        checkpoint_paths: Paths to M `best.pt` checkpoints, one per ensemble member.
        loader: Deterministic eval loader.
        device: Compute device.
        backbone: timm backbone name (must match how members were trained).
        num_classes: Number of classes.
        dropout: Dropout prob used in head construction (weights override anyway).

    Returns:
        Dict:
          probs_mean:           (N, C)  average softmax across members
          probs_var:            (N, C)  variance across members
          predictive_entropy:   (N,)    H[mean]
          expected_entropy:     (N,)    E_m H[p_m]
          mutual_information:   (N,)    predictive_entropy - expected_entropy
          labels:               (N,)
          id_codes:             list[str]
    """
    if len(checkpoint_paths) < 2:
        raise ValueError(f"Need at least 2 ensemble members, got {len(checkpoint_paths)}")

    eps = 1e-12
    sums_p = None
    sums_p_sq = None
    sums_entropy = None
    labels: list[np.ndarray] = []
    id_codes: list[str] = []
    n_total = len(loader.dataset)

    for m_idx, ckpt_path in enumerate(tqdm(checkpoint_paths, desc="Ensemble members")):
        model = RetinalClassifier(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        batch_offset = 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + eps), axis=1)

            if m_idx == 0:
                if sums_p is None:
                    sums_p = np.zeros((n_total, probs.shape[1]), dtype=np.float64)
                    sums_p_sq = np.zeros((n_total, probs.shape[1]), dtype=np.float64)
                    sums_entropy = np.zeros(n_total, dtype=np.float64)
                bs = probs.shape[0]
                sums_p[batch_offset:batch_offset + bs] += probs
                sums_p_sq[batch_offset:batch_offset + bs] += probs ** 2
                sums_entropy[batch_offset:batch_offset + bs] += entropy
                labels.append(batch["label"].numpy())
                id_codes.extend(batch["id_code"])
                batch_offset += bs
            else:
                bs = probs.shape[0]
                sums_p[batch_offset:batch_offset + bs] += probs
                sums_p_sq[batch_offset:batch_offset + bs] += probs ** 2
                sums_entropy[batch_offset:batch_offset + bs] += entropy
                batch_offset += bs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    m = len(checkpoint_paths)
    probs_mean = sums_p / m
    probs_var = (sums_p_sq / m) - probs_mean ** 2
    expected_entropy = sums_entropy / m
    predictive_entropy = -np.sum(probs_mean * np.log(probs_mean + eps), axis=1)
    mutual_information = predictive_entropy - expected_entropy

    return {
        "probs_mean": probs_mean.astype(np.float32),
        "probs_var": probs_var.astype(np.float32),
        "predictive_entropy": predictive_entropy.astype(np.float32),
        "expected_entropy": expected_entropy.astype(np.float32),
        "mutual_information": mutual_information.astype(np.float32),
        "labels": np.concatenate(labels),
        "id_codes": id_codes,
    }
