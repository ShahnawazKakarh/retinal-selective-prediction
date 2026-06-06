"""Classification + calibration metrics.

ECE (Expected Calibration Error) is the headline calibration metric for
selective prediction work — a well-calibrated softmax is the precondition
for trustworthy abstention thresholds.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE with equal-width bins on max-confidence.

    Args:
        probs: (N, C) softmax probabilities.
        labels: (N,) integer ground-truth labels.
        n_bins: Number of confidence bins.

    Returns:
        ECE in [0, 1].
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, C), got shape {probs.shape}")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Per-bin confidence vs. accuracy, for reliability diagrams."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_count = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        bin_centers.append((lo + hi) / 2)
        if in_bin.any():
            bin_acc.append(accuracies[in_bin].mean())
            bin_conf.append(confidences[in_bin].mean())
            bin_count.append(int(in_bin.sum()))
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)
    return {
        "bin_centers": np.array(bin_centers),
        "bin_accuracy": np.array(bin_acc),
        "bin_confidence": np.array(bin_conf),
        "bin_count": np.array(bin_count),
    }
