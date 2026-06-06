"""Monte Carlo Dropout for predictive uncertainty.

Gal & Ghahramani (2016): keep dropout active at inference and average T forward
passes. The mean is the prediction; spread across passes is epistemic uncertainty.

We report three uncertainty signals; later we evaluate which one drives the
best selective prediction policy:

  * predictive_entropy: H[E_t p_t]            (total uncertainty)
  * expected_entropy:   E_t H[p_t]            (aleatoric component)
  * mutual_information: predictive_entropy
                        - expected_entropy    (epistemic component, BALD)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 30,
) -> dict[str, np.ndarray]:
    """Run T stochastic forward passes per input.

    Args:
        model: A `RetinalClassifier` (must expose `enable_mc_dropout()`).
        loader: Deterministic eval loader (no shuffling, deterministic transforms).
        device: Compute device.
        n_samples: Number of MC samples (T).

    Returns:
        Dict with keys:
          probs_mean:           (N, C) softmax mean across T passes
          predictive_entropy:   (N,)
          expected_entropy:     (N,)
          mutual_information:   (N,)
          labels:               (N,)
          id_codes:             list[str]
    """
    if not hasattr(model, "enable_mc_dropout"):
        raise ValueError("model must implement enable_mc_dropout() — use RetinalClassifier.")

    model.eval()
    model.enable_mc_dropout()  # flip dropout layers back to train mode

    all_id_codes: list[str] = []
    all_labels: list[np.ndarray] = []
    sums = None              # running sum of probs across T passes:  (N, C)
    sum_entropies = None     # running sum of per-pass entropies:      (N,)
    seen_batches = 0

    # We loop over T outer passes, each time iterating the loader.
    # Iterating the loader inside the T-loop preserves memory; the loader
    # must be deterministic (shuffle=False, deterministic transforms).
    eps = 1e-12
    for t in tqdm(range(n_samples), desc=f"MC dropout (T={n_samples})"):
        batch_offset = 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            entropy_t = -np.sum(probs * np.log(probs + eps), axis=1)  # (B,)

            if t == 0:
                # First pass: allocate aggregates, collect labels and ids
                if sums is None:
                    n_total = len(loader.dataset)
                    sums = np.zeros((n_total, probs.shape[1]), dtype=np.float64)
                    sum_entropies = np.zeros(n_total, dtype=np.float64)
                bs = probs.shape[0]
                sums[batch_offset:batch_offset + bs] += probs
                sum_entropies[batch_offset:batch_offset + bs] += entropy_t
                all_labels.append(batch["label"].numpy())
                all_id_codes.extend(batch["id_code"])
                batch_offset += bs
            else:
                bs = probs.shape[0]
                sums[batch_offset:batch_offset + bs] += probs
                sum_entropies[batch_offset:batch_offset + bs] += entropy_t
                batch_offset += bs
        seen_batches += 1

    probs_mean = sums / n_samples
    expected_entropy = sum_entropies / n_samples
    predictive_entropy = -np.sum(probs_mean * np.log(probs_mean + eps), axis=1)
    mutual_information = predictive_entropy - expected_entropy

    return {
        "probs_mean": probs_mean.astype(np.float32),
        "predictive_entropy": predictive_entropy.astype(np.float32),
        "expected_entropy": expected_entropy.astype(np.float32),
        "mutual_information": mutual_information.astype(np.float32),
        "labels": np.concatenate(all_labels),
        "id_codes": all_id_codes,
    }
