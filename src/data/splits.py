"""Stratified train/val/test splits — deterministic, committed to disk.

Splits are computed once from a fixed seed and saved as CSVs. Every later
experiment reads the same CSVs, guaranteeing identical splits across runs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_stratified_splits(
    df: pd.DataFrame,
    label_col: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split.

    First splits off test, then val from the remainder. Both stratified by label.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Fractions must be in (0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    trainval_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df[label_col],
        random_state=seed,
    )
    relative_val = val_fraction / (1.0 - test_fraction)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=relative_val,
        stratify=trainval_df[label_col],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save split CSVs. These are committed to git so all runs share splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def load_splits(
    splits_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously-saved splits."""
    splits_dir = Path(splits_dir)
    return (
        pd.read_csv(splits_dir / "train.csv"),
        pd.read_csv(splits_dir / "val.csv"),
        pd.read_csv(splits_dir / "test.csv"),
    )
