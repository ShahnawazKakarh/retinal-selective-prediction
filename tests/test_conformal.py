"""Tests for conformal prediction — verify marginal coverage guarantee.

The whole point of conformal is the coverage guarantee. If our implementation
doesn't hit it on synthetic data, the implementation is wrong (not the theory).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.conformal import APS, SplitConformal


def _make_calibrated_probs(n: int, num_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize calibrated probabilities — useful for coverage tests.

    We draw a uniform random label, then build a softmax that puts mass `p_true`
    on the true class (with noise) and divides the rest among the others.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n)
    probs = np.empty((n, num_classes))
    for i in range(n):
        p_true = rng.uniform(0.4, 0.95)
        other = (1 - p_true) / (num_classes - 1)
        probs[i] = other
        probs[i, labels[i]] = p_true
        # mild noise to avoid degenerate ties
        probs[i] += rng.normal(0, 0.005, size=num_classes)
        probs[i] = np.clip(probs[i], 1e-6, None)
        probs[i] /= probs[i].sum()
    return probs, labels


def test_split_conformal_covers_target():
    """Empirical coverage on a fresh test set should be ≥ 1 - alpha (within slack)."""
    cal_probs, cal_labels = _make_calibrated_probs(500, num_classes=5, seed=0)
    test_probs, test_labels = _make_calibrated_probs(500, num_classes=5, seed=1)

    cp = SplitConformal(alpha=0.10).calibrate(cal_probs, cal_labels)
    cov = cp.empirical_coverage(test_probs, test_labels)
    # Marginal coverage; allow ~3pp slack at this sample size
    assert cov >= 0.90 - 0.03, f"coverage {cov:.3f} below target"


def test_aps_covers_target():
    cal_probs, cal_labels = _make_calibrated_probs(500, num_classes=5, seed=2)
    test_probs, test_labels = _make_calibrated_probs(500, num_classes=5, seed=3)

    cp = APS(alpha=0.10).calibrate(cal_probs, cal_labels)
    cov = cp.empirical_coverage(test_probs, test_labels)
    assert cov >= 0.90 - 0.03, f"APS coverage {cov:.3f} below target"


def test_set_sizes_bounded_by_num_classes():
    probs, labels = _make_calibrated_probs(100, num_classes=5, seed=7)
    for cp in (SplitConformal(alpha=0.1), APS(alpha=0.1)):
        cp.calibrate(probs, labels)
        sizes = cp.set_sizes(probs)
        assert sizes.min() >= 1
        assert sizes.max() <= 5


def test_calibration_required_before_prediction():
    cp = SplitConformal(alpha=0.10)
    with pytest.raises(RuntimeError):
        cp.predict_sets(np.array([[0.2, 0.8]]))
    cp2 = APS(alpha=0.10)
    with pytest.raises(RuntimeError):
        cp2.predict_sets(np.array([[0.2, 0.8]]))


def test_invalid_alpha_rejected():
    cal_probs, cal_labels = _make_calibrated_probs(50, num_classes=3, seed=11)
    with pytest.raises(ValueError):
        SplitConformal(alpha=0.0).calibrate(cal_probs, cal_labels)
    with pytest.raises(ValueError):
        SplitConformal(alpha=1.0).calibrate(cal_probs, cal_labels)
    with pytest.raises(ValueError):
        APS(alpha=-0.1).calibrate(cal_probs, cal_labels)
