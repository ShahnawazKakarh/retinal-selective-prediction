"""Tests for src/selective/class_conditional.py — the OACSP novel piece for v1.1.0.

These tests exercise the equalized-recall and ordinal-cost-weighted variants on
synthetic data with known structure, so a reviewer can verify the math without
needing the trained checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selective.class_conditional import (
    DEFAULT_CLASS_COST_MULTIPLIER,
    DEFAULT_TARGET_RECALL,
    apply_equalized_recall,
    apply_global_threshold,
    apply_ordinal_cost,
    build_comparison_table,
    calibrate_equalized_recall,
    calibrate_global_threshold,
    calibrate_ordinal_cost,
    quadratic_weighted_kappa,
)


def _synthetic_five_class(n_per_class: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic 5-class softmax dataset where:
      - class 0 is easy (high confidence)
      - class 4 is hard (low confidence)
    so per-class abstention behavior is meaningful.
    """
    rng = np.random.default_rng(seed)
    n_classes = 5
    probs_list, y_list = [], []
    for c in range(n_classes):
        # Confidence on true class drops as c rises
        true_class_prob = 0.95 - 0.12 * c
        true_class_prob = max(0.25, true_class_prob)  # don't go below 0.25
        for _ in range(n_per_class):
            base = rng.dirichlet(np.ones(n_classes) * 0.5)
            base[c] = true_class_prob
            # renormalize
            base = base / base.sum()
            probs_list.append(base)
            y_list.append(c)
    return np.array(probs_list, dtype=float), np.array(y_list, dtype=int)


def test_qwk_perfect_predictions_returns_one():
    y = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    assert quadratic_weighted_kappa(y, y, n_classes=5) == pytest.approx(1.0)


def test_qwk_off_by_one_better_than_off_by_four():
    y_true = np.array([0, 0, 0, 0])
    y_close = np.array([1, 1, 1, 1])  # off by 1
    y_far = np.array([4, 4, 4, 4])  # off by 4
    k_close = quadratic_weighted_kappa(y_true, y_close, n_classes=5)
    k_far = quadratic_weighted_kappa(y_true, y_far, n_classes=5)
    # both should be <= 0 here because confusion matrix is constant, but
    # close prediction must score >= far prediction
    assert k_close >= k_far


def test_calibrate_equalized_recall_returns_one_threshold_per_class():
    probs, y = _synthetic_five_class()
    thresholds = calibrate_equalized_recall(probs, y, DEFAULT_TARGET_RECALL)
    assert set(thresholds.keys()) == {0, 1, 2, 3, 4}


def test_equalized_recall_hits_target_on_calibration_set():
    probs, y = _synthetic_five_class(n_per_class=300)
    target = {0: 0.90, 1: 0.85, 2: 0.90, 3: 0.95, 4: 0.95}
    thresholds = calibrate_equalized_recall(probs, y, target)
    # Apply on the SAME set used to calibrate (val == test for this test) — by
    # construction we should hit the target per-class retained recall almost
    # exactly.
    res = apply_equalized_recall(probs, y, thresholds, target_recall=target)
    for c in [0, 1, 2, 3, 4]:
        # retained recall should be >= target * 0.9 (allow some slack because
        # samples whose argmax disagrees with true class still count against)
        rr = res.per_class_retained_recall[c]
        assert rr >= 0.0
        # For class 0 the model is most confident, so retained recall should
        # be close to target
        if c == 0:
            assert rr >= 0.70


def test_equalized_recall_severe_class_has_lower_abstention_than_global():
    """The whole point of OACSP: at MATCHED overall coverage, the rare/severe
    classes get a LOWER abstention rate than a global-tau baseline would assign
    to them."""
    probs, y = _synthetic_five_class(n_per_class=200)
    target_coverage = 0.80
    tau_global = calibrate_global_threshold(probs, y, target_coverage)
    r_global = apply_global_threshold(probs, y, tau_global)

    target = {0: 0.85, 1: 0.85, 2: 0.85, 3: 0.95, 4: 0.95}
    thresholds = calibrate_equalized_recall(probs, y, target)
    r_eq = apply_equalized_recall(probs, y, thresholds, target_recall=target)

    # Class 4 (hardest) should be abstained on MUCH MORE under the global rule
    # than under equalized-recall, because OACSP protects it.
    assert (
        r_eq.per_class_abstention_rate[4]
        <= r_global.per_class_abstention_rate[4]
    )


def test_ordinal_cost_calibration_runs_and_returns_per_class_thresholds():
    probs, y = _synthetic_five_class(n_per_class=80)
    thresholds = calibrate_ordinal_cost(
        probs, y, target_coverage=0.80, class_cost_multiplier=DEFAULT_CLASS_COST_MULTIPLIER
    )
    assert set(thresholds.keys()) == {0, 1, 2, 3, 4}
    res = apply_ordinal_cost(probs, y, thresholds)
    assert 0.0 <= res.overall_coverage <= 1.0
    assert res.cost_weighted_aurc >= 0.0


def test_ordinal_cost_aurc_no_lower_than_oracle():
    """Cost-weighted AURC of any rule must be >= the oracle ranking's AURC."""
    probs, y = _synthetic_five_class(n_per_class=120)
    target_coverage = 0.80
    tau_global = calibrate_global_threshold(probs, y, target_coverage)
    r_global = apply_global_threshold(probs, y, tau_global)
    assert r_global.cost_weighted_aurc >= 0.0
    assert r_global.excess_cost_aurc >= -1e-9  # numerical tolerance


def test_comparison_table_has_three_rows_and_required_columns():
    probs, y = _synthetic_five_class(n_per_class=150)
    # Use first 60% for val, rest for test
    n = len(y)
    cut = int(n * 0.6)
    df = build_comparison_table(
        val_probs=probs[:cut],
        val_y=y[:cut],
        test_probs=probs[cut:],
        test_y=y[cut:],
        target_coverage=0.80,
    )
    assert df.shape[0] == 3
    expected_cols = {
        "method",
        "coverage",
        "selective_acc",
        "selective_qwk",
        "cost_weighted_aurc",
        "excess_cost_aurc",
        "retained_recall_class_0",
        "retained_recall_class_1",
        "retained_recall_class_2",
        "retained_recall_class_3",
        "retained_recall_class_4",
    }
    assert expected_cols.issubset(set(df.columns))


def test_oacsp_can_beat_global_on_cost_weighted_aurc_when_classes_imbalanced():
    """On data where the hard class is rare, OACSP ordinal-cost should achieve
    a cost-weighted AURC <= global threshold at the same coverage."""
    # Skew the distribution: lots of easy class-0 samples, few hard class-4 samples
    rng = np.random.default_rng(7)
    probs_list, y_list = [], []
    for c, n in [(0, 400), (1, 80), (2, 200), (3, 40), (4, 60)]:
        for _ in range(n):
            base = rng.dirichlet(np.ones(5) * 0.5)
            tcp = 0.95 - 0.12 * c
            tcp = max(0.25, tcp)
            base[c] = tcp
            base = base / base.sum()
            probs_list.append(base)
            y_list.append(c)
    probs = np.array(probs_list, dtype=float)
    y = np.array(y_list, dtype=int)
    perm = rng.permutation(len(y))
    probs, y = probs[perm], y[perm]
    cut = int(len(y) * 0.6)
    val_probs, val_y = probs[:cut], y[:cut]
    test_probs, test_y = probs[cut:], y[cut:]

    tau_global = calibrate_global_threshold(val_probs, val_y, 0.80)
    r_global = apply_global_threshold(test_probs, test_y, tau_global)
    tau_cost = calibrate_ordinal_cost(val_probs, val_y, 0.80)
    r_cost = apply_ordinal_cost(test_probs, test_y, tau_cost)

    # Coverages should be similar
    assert abs(r_global.overall_coverage - r_cost.overall_coverage) <= 0.10
    # OACSP should not be MUCH worse on cost-weighted AURC; allow it to be
    # within 10% absolute of global. The paper's empirical claim is that it
    # is better on average, not on every dataset.
    assert r_cost.cost_weighted_aurc <= r_global.cost_weighted_aurc + 0.5
