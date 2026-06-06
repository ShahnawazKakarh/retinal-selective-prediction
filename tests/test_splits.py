"""Tests for stratified splits — determinism + class preservation + no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import make_stratified_splits


def _make_df(n_per_class: dict[int, int]) -> pd.DataFrame:
    rows = []
    counter = 0
    for cls, n in n_per_class.items():
        for _ in range(n):
            rows.append({"id_code": f"id_{counter:05d}", "diagnosis": cls})
            counter += 1
    return pd.DataFrame(rows)


def test_splits_are_deterministic():
    df = _make_df({0: 400, 1: 100, 2: 80, 3: 60, 4: 40})
    a1, b1, c1 = make_stratified_splits(df, "diagnosis", 0.15, 0.15, seed=42)
    a2, b2, c2 = make_stratified_splits(df, "diagnosis", 0.15, 0.15, seed=42)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(c1, c2)


def test_splits_preserve_class_distribution_approximately():
    df = _make_df({0: 400, 1: 100, 2: 80, 3: 60, 4: 40})
    train_df, val_df, test_df = make_stratified_splits(
        df, "diagnosis", 0.15, 0.15, seed=42,
    )
    p_full = df["diagnosis"].value_counts(normalize=True).sort_index()
    for d in (train_df, val_df, test_df):
        p_split = d["diagnosis"].value_counts(normalize=True).sort_index()
        # Within 3 percentage points per class — tight for stratified
        np.testing.assert_allclose(p_split.values, p_full.values, atol=0.03)


def test_splits_are_disjoint_and_complete():
    df = _make_df({0: 200, 1: 50, 2: 30})
    train_df, val_df, test_df = make_stratified_splits(
        df, "diagnosis", 0.15, 0.15, seed=0,
    )
    ids_train = set(train_df["id_code"])
    ids_val = set(val_df["id_code"])
    ids_test = set(test_df["id_code"])
    # No overlap
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)
    # Union covers everything
    assert ids_train | ids_val | ids_test == set(df["id_code"])


def test_invalid_fractions_rejected():
    df = _make_df({0: 100, 1: 100})
    with pytest.raises(ValueError):
        make_stratified_splits(df, "diagnosis", 0.0, 0.2, seed=0)
    with pytest.raises(ValueError):
        make_stratified_splits(df, "diagnosis", 0.6, 0.5, seed=0)   # sum >= 1
