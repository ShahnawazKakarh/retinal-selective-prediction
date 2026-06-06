"""Tests for risk-coverage / AURC math.

These are the most-reused functions in the paper, so they get the
most-rigorous tests. Failures here would invalidate every results table.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.selective.risk_coverage import (
    aurc,
    excess_aurc,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)


def test_perfect_uncertainty_yields_oracle_aurc():
    """If uncertainty perfectly ranks correct < incorrect, AURC = oracle."""
    correct = np.array([1, 1, 1, 0, 0])
    # Lower uncertainty on correct items → perfect ranking
    uncertainty = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert excess_aurc(correct, uncertainty) == pytest.approx(0.0, abs=1e-12)


def test_useless_uncertainty_has_positive_excess_aurc():
    """Constant uncertainty must be no better than random ordering."""
    correct = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    uncertainty = np.ones_like(correct, dtype=float)
    # Excess > 0 (uncertainty signal carries no information)
    assert excess_aurc(correct, uncertainty) > 0.0


def test_risk_coverage_shapes_and_bounds():
    n = 50
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, size=n)
    uncertainty = rng.random(n)
    curve = risk_coverage_curve(correct, uncertainty)
    assert curve["coverages"].shape == (n,)
    assert curve["risks"].shape == (n,)
    assert np.isclose(curve["coverages"][-1], 1.0)
    assert (curve["risks"] >= 0).all() and (curve["risks"] <= 1).all()
    assert (curve["coverages"] > 0).all() and (curve["coverages"] <= 1).all()


def test_aurc_within_unit_interval():
    rng = np.random.default_rng(1)
    correct = rng.integers(0, 2, size=100)
    uncertainty = rng.random(100)
    value = aurc(correct, uncertainty)
    assert 0.0 <= value <= 1.0


def test_selective_accuracy_full_coverage_equals_overall():
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    uncertainty = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.7, 0.8])
    result = selective_accuracy_at_coverage(correct, uncertainty, 1.0)
    assert result["selective_accuracy"] == pytest.approx(correct.mean())


def test_selective_accuracy_top_coverage_keeps_only_most_certain():
    # Most certain (lowest unc) sample is correct → selective acc at smallest
    # coverage should be 1.0.
    correct = np.array([1, 0, 1, 0])
    uncertainty = np.array([0.1, 0.9, 0.2, 0.8])
    result = selective_accuracy_at_coverage(correct, uncertainty, 0.25)
    assert result["selective_accuracy"] == 1.0


def test_aurc_input_validation():
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([1, 0]), np.array([0.1]))   # length mismatch
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([]), np.array([]))           # empty
