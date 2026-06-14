"""Ordinal-Aware Class-Conditional Selective Prediction (OACSP).

This module is the novel contribution of v1.1.0.

Every published selective-prediction method on diabetic retinopathy uses a SINGLE
confidence threshold across all 5 severity classes. This is wrong for two reasons:

1. DR severity is ORDINAL (0 < 1 < 2 < 3 < 4). Confusing No-DR with Mild-DR is far
   less harmful than confusing No-DR with Proliferative-DR. A single threshold
   treats both errors the same.

2. CLINICAL COST IS ASYMMETRIC. Missing a Severe / Proliferative DR case can blind
   the patient. Missing a Mild DR case has near-zero clinical consequence. Yet a
   global threshold abstains LEAST on the easy classes and MOST on the hard rare
   classes — the opposite of what we want.

OACSP replaces the single threshold tau with per-class thresholds
{tau_0, tau_1, tau_2, tau_3, tau_4} calibrated on the validation set. Two variants:

  A) equalized_recall:
        Each class gets the threshold that achieves a target per-class retained
        recall (e.g. 0.95 for Severe / Proliferative, 0.90 for No-DR / Moderate,
        0.85 for Mild). Operator-meaningful clinical knob.

  B) ordinal_cost_weighted:
        Each sample's confidence signal is multiplied by an ordinal-distance cost
        before global thresholding. Equivalent to per-class thresholds set such
        that the expected cost-weighted risk is minimized at any target coverage.

Both variants operate POST-HOC on already-saved per-sample predictions
(predictions.csv from evaluate.py). No retraining required.

Author: Khan, Muhammad Shahnawaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ClassConditionalVariant = Literal["equalized_recall", "ordinal_cost_weighted"]


# -----------------------------------------------------------------------------
# Default clinical priors for diabetic retinopathy (5-class ICDR)
# -----------------------------------------------------------------------------
#
# These can be overridden by the operator. The defaults below encode a defensible
# clinical position: missing a Severe or Proliferative DR case is much costlier
# than missing a Mild DR case.

DEFAULT_TARGET_RECALL: dict[int, float] = {
    0: 0.90,  # No DR
    1: 0.85,  # Mild
    2: 0.90,  # Moderate
    3: 0.95,  # Severe
    4: 0.95,  # Proliferative DR
}

# Cost matrix C[true_class, abstain] used by the ordinal-cost variant.
# Diagonal ordinal distance: C(true=t, predicted=p) = |t - p|^alpha for alpha=1.
# Abstention has a fixed cost c_abstain that the operator picks; here 0.5 means
# "abstaining is worth half a one-step ordinal error". Severe / Proliferative
# misses get a multiplicative bump.

DEFAULT_CLASS_COST_MULTIPLIER: dict[int, float] = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 2.5,  # Severe — missing this is much worse
    4: 3.0,  # Proliferative — missing this is the worst
}


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------


@dataclass
class OACSPResult:
    """Holds the output of a single OACSP evaluation."""

    variant: ClassConditionalVariant
    per_class_thresholds: dict[int, float]
    target_recall: dict[int, float] | None
    overall_coverage: float
    overall_selective_accuracy: float
    overall_selective_qwk: float
    cost_weighted_aurc: float
    excess_cost_aurc: float
    per_class_retained_recall: dict[int, float]
    per_class_abstention_rate: dict[int, float]
    per_class_kept_count: dict[int, int]
    per_class_total_count: dict[int, int]

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "per_class_thresholds": self.per_class_thresholds,
            "target_recall": self.target_recall,
            "overall_coverage": self.overall_coverage,
            "overall_selective_accuracy": self.overall_selective_accuracy,
            "overall_selective_qwk": self.overall_selective_qwk,
            "cost_weighted_aurc": self.cost_weighted_aurc,
            "excess_cost_aurc": self.excess_cost_aurc,
            "per_class_retained_recall": self.per_class_retained_recall,
            "per_class_abstention_rate": self.per_class_abstention_rate,
            "per_class_kept_count": self.per_class_kept_count,
            "per_class_total_count": self.per_class_total_count,
        }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _ordinal_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """|y_true - y_pred| for ordinal classes."""
    return np.abs(y_true.astype(int) - y_pred.astype(int)).astype(float)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa, implemented locally so this module has
    zero external deps beyond numpy / pandas."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    if len(y_true) == 0:
        return float("nan")

    w = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    O = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    n = O.sum()
    if n == 0:
        return float("nan")
    E = np.outer(act_hist, pred_hist) / n

    num = (w * O).sum()
    den = (w * E).sum()
    if den == 0:
        return 1.0
    return float(1.0 - num / den)


def _global_signal_from_softmax(probs: np.ndarray) -> np.ndarray:
    """Default uncertainty signal = negative max softmax probability.

    Returns array shape (N,), higher value = more uncertain.
    """
    return -probs.max(axis=1)


def _cost_weighted_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kept: np.ndarray,
    class_cost_multiplier: dict[int, float],
) -> float:
    """Expected cost on retained samples + 0 cost on abstained samples (no penalty
    for abstention beyond the lost coverage)."""
    if kept.sum() == 0:
        return 0.0
    y_true_k = y_true[kept]
    y_pred_k = y_pred[kept]
    ord_dist = _ordinal_distance(y_true_k, y_pred_k)
    cls_mult = np.array([class_cost_multiplier[int(c)] for c in y_true_k], dtype=float)
    cost = ord_dist * cls_mult
    return float(cost.mean())


# -----------------------------------------------------------------------------
# Variant A: Equalized-recall per-class thresholds
# -----------------------------------------------------------------------------


def calibrate_equalized_recall(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    target_recall: dict[int, float] | None = None,
    signal_fn=_global_signal_from_softmax,
) -> dict[int, float]:
    """Compute per-class signal thresholds {tau_c} on the val set such that for
    each class c, retained per-class recall >= target_recall[c].

    Logic: For each class c, look only at val samples whose TRUE label is c.
    Sort them by the uncertainty signal (ascending = most confident first).
    Pick the threshold at the target_recall[c] quantile — meaning we KEEP that
    fraction of class-c samples (the most confident), abstain on the rest.

    Returns
    -------
    thresholds : dict[int, float]
        tau_c such that we KEEP val samples of class c with signal <= tau_c.
    """
    if target_recall is None:
        target_recall = DEFAULT_TARGET_RECALL

    val_y = val_y.astype(int)
    val_signal = signal_fn(val_probs)
    thresholds: dict[int, float] = {}
    n_classes = val_probs.shape[1]

    for c in range(n_classes):
        mask = val_y == c
        if mask.sum() == 0:
            thresholds[c] = float("inf")
            continue
        sig_c = np.sort(val_signal[mask])
        q = target_recall.get(c, 0.9)
        q = float(np.clip(q, 0.0, 1.0))
        # We KEEP the q-fraction with the LOWEST signal (= most confident).
        # That corresponds to the q-quantile of the signal distribution.
        idx = int(np.ceil(q * len(sig_c))) - 1
        idx = max(0, min(idx, len(sig_c) - 1))
        thresholds[c] = float(sig_c[idx])
    return thresholds


def apply_equalized_recall(
    test_probs: np.ndarray,
    test_y: np.ndarray,
    thresholds: dict[int, float],
    target_recall: dict[int, float] | None = None,
    class_cost_multiplier: dict[int, float] | None = None,
    signal_fn=_global_signal_from_softmax,
) -> OACSPResult:
    """Apply per-class thresholds to test set and report all metrics.

    The class-conditional rule is applied based on the PREDICTED class
    (since the true label is unknown at deployment time). For each test sample i:
        predicted = argmax(test_probs[i])
        keep iff signal_fn(test_probs[i]) <= thresholds[predicted]
    """
    if class_cost_multiplier is None:
        class_cost_multiplier = DEFAULT_CLASS_COST_MULTIPLIER

    test_y = test_y.astype(int)
    test_pred = test_probs.argmax(axis=1).astype(int)
    test_signal = signal_fn(test_probs)
    n_classes = test_probs.shape[1]

    tau_per_sample = np.array(
        [thresholds.get(int(p), float("inf")) for p in test_pred], dtype=float
    )
    kept = test_signal <= tau_per_sample

    # Overall metrics
    overall_coverage = float(kept.mean())
    if kept.sum() > 0:
        sel_acc = float((test_pred[kept] == test_y[kept]).mean())
        sel_qwk = quadratic_weighted_kappa(test_y[kept], test_pred[kept], n_classes=n_classes)
    else:
        sel_acc = float("nan")
        sel_qwk = float("nan")

    # Per-class retained recall (the operator's clinical knob — did we hit the
    # target on each class?)
    per_class_retained_recall: dict[int, float] = {}
    per_class_abstention_rate: dict[int, float] = {}
    per_class_kept_count: dict[int, int] = {}
    per_class_total_count: dict[int, int] = {}
    for c in range(n_classes):
        cls_mask = test_y == c
        total_c = int(cls_mask.sum())
        per_class_total_count[c] = total_c
        if total_c == 0:
            per_class_retained_recall[c] = float("nan")
            per_class_abstention_rate[c] = float("nan")
            per_class_kept_count[c] = 0
            continue
        kept_and_class = cls_mask & kept
        # retained per-class recall = correctly-predicted class-c samples that were also kept,
        # divided by total class-c samples
        correct_kept = ((test_pred == c) & kept_and_class).sum()
        per_class_retained_recall[c] = float(correct_kept / total_c)
        per_class_abstention_rate[c] = float(1.0 - kept_and_class.sum() / total_c)
        per_class_kept_count[c] = int(kept_and_class.sum())

    # Cost-weighted AURC: integrate cost-weighted risk over coverage sweep,
    # using the global signal sorted ascending (most confident first).
    cw_aurc = _cost_weighted_aurc(test_y, test_pred, test_signal, class_cost_multiplier)

    # Excess = cost AURC minus the oracle-ranking cost AURC (lower bound at
    # this accuracy level).
    excess = cw_aurc - _oracle_cost_aurc(test_y, test_pred, class_cost_multiplier)

    return OACSPResult(
        variant="equalized_recall",
        per_class_thresholds={int(k): float(v) for k, v in thresholds.items()},
        target_recall=target_recall,
        overall_coverage=overall_coverage,
        overall_selective_accuracy=sel_acc,
        overall_selective_qwk=sel_qwk,
        cost_weighted_aurc=cw_aurc,
        excess_cost_aurc=excess,
        per_class_retained_recall=per_class_retained_recall,
        per_class_abstention_rate=per_class_abstention_rate,
        per_class_kept_count=per_class_kept_count,
        per_class_total_count=per_class_total_count,
    )


# -----------------------------------------------------------------------------
# Variant B: Ordinal-cost-weighted global thresholding (induced per-class)
# -----------------------------------------------------------------------------


def calibrate_ordinal_cost(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    target_coverage: float = 0.80,
    class_cost_multiplier: dict[int, float] | None = None,
    signal_fn=_global_signal_from_softmax,
) -> dict[int, float]:
    """Pick per-class thresholds {tau_c} that minimize expected cost-weighted
    risk at the operator's chosen target coverage on the validation set.

    Approach: For each class c, sweep the per-class threshold and pick the value
    that, when applied jointly with the other classes (greedy coordinate descent,
    initialized at the global percentile), minimizes:
        E[cost_weighted_risk on kept | coverage >= target_coverage]
    """
    if class_cost_multiplier is None:
        class_cost_multiplier = DEFAULT_CLASS_COST_MULTIPLIER

    val_y = val_y.astype(int)
    val_pred = val_probs.argmax(axis=1).astype(int)
    val_signal = signal_fn(val_probs)
    n_classes = val_probs.shape[1]

    # Init: per-class quantile equal to global target_coverage
    thresholds: dict[int, float] = {}
    for c in range(n_classes):
        cls_mask = val_pred == c
        if cls_mask.sum() == 0:
            thresholds[c] = float("inf")
            continue
        sig_c = np.sort(val_signal[cls_mask])
        q = float(np.clip(target_coverage, 0.0, 1.0))
        idx = int(np.ceil(q * len(sig_c))) - 1
        idx = max(0, min(idx, len(sig_c) - 1))
        thresholds[c] = float(sig_c[idx])

    # Coordinate descent: for each class, sweep its threshold over the candidate
    # quantiles (per_class_pred sig values), keep the one that gives the lowest
    # cost-weighted risk on kept val samples while overall coverage stays
    # >= target_coverage.
    for _ in range(3):  # 3 passes is plenty for monotone improvement
        for c in range(n_classes):
            cls_mask = val_pred == c
            if cls_mask.sum() == 0:
                continue
            candidates = np.sort(val_signal[cls_mask])
            best_t = thresholds[c]
            best_cost = float("inf")
            for cand in candidates:
                thresholds[c] = float(cand)
                cost, cov = _evaluate_threshold_set(
                    val_y, val_pred, val_signal, thresholds, class_cost_multiplier
                )
                if cov < target_coverage:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_t = float(cand)
            thresholds[c] = best_t
    return thresholds


def apply_ordinal_cost(
    test_probs: np.ndarray,
    test_y: np.ndarray,
    thresholds: dict[int, float],
    class_cost_multiplier: dict[int, float] | None = None,
    signal_fn=_global_signal_from_softmax,
) -> OACSPResult:
    """Same evaluation as Variant A, just labeled variant='ordinal_cost_weighted'."""
    res = apply_equalized_recall(
        test_probs=test_probs,
        test_y=test_y,
        thresholds=thresholds,
        target_recall=None,
        class_cost_multiplier=class_cost_multiplier,
        signal_fn=signal_fn,
    )
    res.variant = "ordinal_cost_weighted"
    return res


# -----------------------------------------------------------------------------
# Baseline: global threshold (single tau)
# -----------------------------------------------------------------------------


def calibrate_global_threshold(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    target_coverage: float,
    signal_fn=_global_signal_from_softmax,
) -> float:
    """Single-tau baseline — pick global threshold so that target_coverage of val
    samples are kept."""
    val_signal = signal_fn(val_probs)
    q = float(np.clip(target_coverage, 0.0, 1.0))
    return float(np.quantile(val_signal, q))


def apply_global_threshold(
    test_probs: np.ndarray,
    test_y: np.ndarray,
    threshold: float,
    class_cost_multiplier: dict[int, float] | None = None,
    signal_fn=_global_signal_from_softmax,
) -> OACSPResult:
    """Apply a single threshold to all classes — the standard published baseline."""
    if class_cost_multiplier is None:
        class_cost_multiplier = DEFAULT_CLASS_COST_MULTIPLIER
    n_classes = test_probs.shape[1]
    thresholds = {c: float(threshold) for c in range(n_classes)}
    res = apply_equalized_recall(
        test_probs=test_probs,
        test_y=test_y,
        thresholds=thresholds,
        target_recall=None,
        class_cost_multiplier=class_cost_multiplier,
        signal_fn=signal_fn,
    )
    res.variant = "equalized_recall"  # cast as same shape; caller knows it's "global"
    return res


# -----------------------------------------------------------------------------
# Cost-weighted AURC implementation
# -----------------------------------------------------------------------------


def _evaluate_threshold_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signal: np.ndarray,
    thresholds: dict[int, float],
    class_cost_multiplier: dict[int, float],
) -> tuple[float, float]:
    """Return (cost_on_kept, coverage) for a given per-class threshold dict."""
    tau_per_sample = np.array(
        [thresholds.get(int(p), float("inf")) for p in y_pred], dtype=float
    )
    kept = signal <= tau_per_sample
    cov = float(kept.mean())
    cost = _cost_weighted_risk(y_true, y_pred, kept, class_cost_multiplier)
    return cost, cov


def _cost_weighted_aurc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signal: np.ndarray,
    class_cost_multiplier: dict[int, float],
) -> float:
    """Cost-weighted AURC: trapezoid-integrate cost over coverage sweeps from
    0 to 1, where at each coverage level we keep the top-confidence fraction."""
    order = np.argsort(signal)  # most confident first
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    n = len(signal)
    if n == 0:
        return 0.0

    cls_mult = np.array([class_cost_multiplier[int(c)] for c in y_true_s], dtype=float)
    per_sample_cost = _ordinal_distance(y_true_s, y_pred_s) * cls_mult

    cum_cost = np.cumsum(per_sample_cost)
    coverages = np.arange(1, n + 1) / n
    risks = cum_cost / np.arange(1, n + 1)  # avg cost on retained set
    return float(np.trapezoid(risks, coverages))


def _oracle_cost_aurc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_cost_multiplier: dict[int, float],
) -> float:
    """Oracle ranking: sort by actual cost ascending (correct kept first)."""
    cls_mult = np.array([class_cost_multiplier[int(c)] for c in y_true], dtype=float)
    per_sample_cost = _ordinal_distance(y_true, y_pred) * cls_mult
    order = np.argsort(per_sample_cost)
    sorted_cost = per_sample_cost[order]
    n = len(per_sample_cost)
    if n == 0:
        return 0.0
    cum = np.cumsum(sorted_cost)
    coverages = np.arange(1, n + 1) / n
    risks = cum / np.arange(1, n + 1)
    return float(np.trapezoid(risks, coverages))


# -----------------------------------------------------------------------------
# Convenience: build the comparison table that goes into the paper
# -----------------------------------------------------------------------------


def build_comparison_table(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    test_probs: np.ndarray,
    test_y: np.ndarray,
    target_coverage: float = 0.80,
    target_recall: dict[int, float] | None = None,
    class_cost_multiplier: dict[int, float] | None = None,
    signal_fn=_global_signal_from_softmax,
) -> pd.DataFrame:
    """Run all three methods (global tau, equalized recall, ordinal cost) on the
    same val/test split and return the comparison table the paper will show."""
    if target_recall is None:
        target_recall = DEFAULT_TARGET_RECALL
    if class_cost_multiplier is None:
        class_cost_multiplier = DEFAULT_CLASS_COST_MULTIPLIER

    # Global baseline — calibrate to the matched target coverage
    tau_global = calibrate_global_threshold(val_probs, val_y, target_coverage, signal_fn)
    r_global = apply_global_threshold(
        test_probs, test_y, tau_global, class_cost_multiplier, signal_fn
    )

    # Variant A
    tau_eq = calibrate_equalized_recall(val_probs, val_y, target_recall, signal_fn)
    r_eq = apply_equalized_recall(
        test_probs, test_y, tau_eq, target_recall, class_cost_multiplier, signal_fn
    )

    # Variant B
    tau_cost = calibrate_ordinal_cost(
        val_probs, val_y, target_coverage, class_cost_multiplier, signal_fn
    )
    r_cost = apply_ordinal_cost(
        test_probs, test_y, tau_cost, class_cost_multiplier, signal_fn
    )

    n_classes = test_probs.shape[1]
    rows = []
    for label, r in [
        ("Global threshold (baseline)", r_global),
        ("OACSP — equalized recall", r_eq),
        ("OACSP — ordinal cost weighted", r_cost),
    ]:
        row = {
            "method": label,
            "coverage": round(r.overall_coverage, 4),
            "selective_acc": round(r.overall_selective_accuracy, 4),
            "selective_qwk": round(r.overall_selective_qwk, 4),
            "cost_weighted_aurc": round(r.cost_weighted_aurc, 4),
            "excess_cost_aurc": round(r.excess_cost_aurc, 4),
        }
        for c in range(n_classes):
            row[f"retained_recall_class_{c}"] = round(r.per_class_retained_recall[c], 4)
        rows.append(row)
    return pd.DataFrame(rows)
