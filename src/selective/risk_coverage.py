"""Risk–coverage analysis for selective prediction.

The core question: if we sort predictions by an uncertainty score and reject
the most uncertain X%, what happens to error on the kept set?

References:
  El-Yaniv & Wiener (2010), "On the foundations of noise-free selective classification"
  Geifman & El-Yaniv (2017), "Selective classification for deep neural networks"

Definitions used here:
  coverage(τ) = fraction of inputs kept (uncertainty ≤ τ)
  risk(τ)     = error rate on the kept subset
  AURC        = ∫ risk(c) dc          (lower is better)

Higher uncertainty score = more uncertain = first to be rejected.
"""
from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    correct: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the full risk–coverage curve.

    Args:
        correct: (N,) bool/int — 1 if the prediction was correct, 0 otherwise.
        uncertainty: (N,) float — higher = more uncertain.

    Returns:
        Dict:
          coverages:  (N,) sorted ascending, from 1/N to 1.0
          risks:      (N,) error rate on the most-certain k items
          thresholds: (N,) the uncertainty threshold used at each coverage
    """
    correct = np.asarray(correct, dtype=np.int32).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if correct.shape != uncertainty.shape:
        raise ValueError("correct and uncertainty must be the same length")
    n = len(correct)
    if n == 0:
        raise ValueError("empty input")

    # Sort by ascending uncertainty (most-certain first)
    order = np.argsort(uncertainty, kind="stable")
    correct_sorted = correct[order]
    unc_sorted = uncertainty[order]

    # As we increase coverage one item at a time, cumulative errors / k = risk
    errors = 1 - correct_sorted
    cum_errors = np.cumsum(errors)
    ks = np.arange(1, n + 1)
    risks = cum_errors / ks
    coverages = ks / n

    return {
        "coverages": coverages,
        "risks": risks,
        "thresholds": unc_sorted,
    }


def aurc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """Area Under the Risk–Coverage curve. Lower is better."""
    curve = risk_coverage_curve(correct, uncertainty)
    return float(np.trapezoid(curve["risks"], curve["coverages"]))


def selective_accuracy_at_coverage(
    correct: np.ndarray,
    uncertainty: np.ndarray,
    target_coverage: float,
) -> dict[str, float]:
    """Accuracy on the kept set when coverage is fixed.

    Args:
        target_coverage: desired coverage in (0, 1].

    Returns:
        Dict with achieved_coverage, selective_accuracy, threshold.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    curve = risk_coverage_curve(correct, uncertainty)
    # smallest k such that k/n >= target_coverage
    k = int(np.ceil(target_coverage * len(correct)))
    k = max(1, min(k, len(correct)))
    idx = k - 1
    return {
        "target_coverage": float(target_coverage),
        "achieved_coverage": float(curve["coverages"][idx]),
        "selective_accuracy": float(1.0 - curve["risks"][idx]),
        "threshold": float(curve["thresholds"][idx]),
    }


def excess_aurc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """AURC minus the AURC of an oracle that knows which predictions are wrong.

    The oracle ranks all correct items before all wrong items, giving the
    minimum achievable AURC. Excess-AURC isolates the contribution of the
    uncertainty signal from the overall error rate. Lower is better.
    """
    actual = aurc(correct, uncertainty)
    # Oracle: assign uncertainty 0 to correct, 1 to wrong
    correct = np.asarray(correct, dtype=np.int32).reshape(-1)
    oracle_unc = (1 - correct).astype(np.float64)
    oracle = aurc(correct, oracle_unc)
    return float(actual - oracle)
