"""IDRiD (Indian Diabetic Retinopathy Image Dataset) loader for external validation.

This module provides a PyTorch Dataset for IDRiD that matches the same interface
as `APTOS2019Dataset` (returns {'image': tensor, 'label': int, 'id_code': str}).

IDRiD uses the same 5-class ICDR severity grading scheme as APTOS, so no
label re-mapping is required:
    0 = No DR
    1 = Mild non-proliferative DR
    2 = Moderate non-proliferative DR
    3 = Severe non-proliferative DR
    4 = Proliferative DR

## Supported layouts

This loader auto-detects two on-disk layouts:

**Layout A — Kaggle flat** (`mariaherrerot/idrid-dataset` on Kaggle):
    /kaggle/input/datasets/mariaherrerot/idrid-dataset/
        idrid_labels.csv          (single CSV with `id_code`, `diagnosis`, ...)
        Imagenes/Imagenes/        (flat image dir)
            IDRiD_001.jpg         (train images, no "test" suffix)
            IDRiD_001test.jpg     (test images, "test" suffix in id_code)
            ...
    Test images are identified by the presence of "test" in the id_code.

**Layout B — Canonical IDRiD** (official IEEE Dataport bundle):
    <root>/B. Disease Grading/
        1. Original Images/
            a. Training Set/
            b. Testing Set/
        2. Groundtruths/
            a. IDRiD_Disease Grading_Training Labels.csv
            b. IDRiD_Disease Grading_Testing Labels.csv

## License & citation

IDRiD is released under CC BY 4.0. If you use it, cite:
    Porwal et al. (2018). Indian Diabetic Retinopathy Image Dataset (IDRiD):
    A database for diabetic retinopathy screening research. Data 3(3), 25.
    DOI: 10.3390/data3030025

Author: Khan, Muhammad Shahnawaz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Canonical IDRiD bundle defaults
IDRID_CANONICAL_ROOT = Path("/kaggle/input/idrid-dataset/B. Disease Grading")

# Kaggle flat layout defaults (mariaherrerot/idrid-dataset)
IDRID_KAGGLE_FLAT_ROOT = Path("/kaggle/input/datasets/mariaherrerot/idrid-dataset")


def _detect_layout(root: Path) -> str:
    """Auto-detect whether `root` is the canonical or kaggle-flat layout."""
    root = Path(root)
    if (root / "idrid_labels.csv").exists():
        return "kaggle_flat"
    if (root / "2. Groundtruths").exists() or (root / "1. Original Images").exists():
        return "canonical"
    # Try one level up if root pointed at the parent of the dataset
    if root.name == "idrid-dataset" and (root / "idrid_labels.csv").exists():
        return "kaggle_flat"
    raise FileNotFoundError(
        f"Cannot identify IDRiD layout under {root}. "
        f"Expected either idrid_labels.csv (Kaggle flat) or "
        f"'2. Groundtruths/' subdir (canonical bundle)."
    )


def load_idrid_labels(
    root: Path,
    split: str = "test",
    layout: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load IDRiD labels and return (DataFrame, images_dir).

    Returns DataFrame with columns: id_code, label (int 0-4)
    """
    root = Path(root)
    if layout is None:
        layout = _detect_layout(root)

    if layout == "kaggle_flat":
        csv_path = root / "idrid_labels.csv"
        images_dir = root / "Imagenes" / "Imagenes"
        df = pd.read_csv(csv_path)
        # Drop the garbage "Unnamed:" columns
        df = df[[c for c in df.columns if not c.startswith("Unnamed")]].copy()
        # Normalize columns
        df = df.rename(columns={"diagnosis": "label"})
        if "id_code" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"Unexpected columns in {csv_path}: {list(df.columns)}"
            )
        df["label"] = df["label"].astype(int)
        # Split: id_codes containing 'test' are the test set
        is_test = df["id_code"].str.contains("test", case=False, na=False)
        if split == "test":
            df = df[is_test].copy()
        elif split == "train":
            df = df[~is_test].copy()
        elif split == "all":
            pass
        else:
            raise ValueError(f"split must be train/test/all, got {split!r}")
        df = df.reset_index(drop=True)
        return df[["id_code", "label"]], images_dir

    elif layout == "canonical":
        labels_dir = root / "2. Groundtruths"
        if split == "train":
            csv_name = "a. IDRiD_Disease Grading_Training Labels.csv"
            images_dir = root / "1. Original Images" / "a. Training Set"
        elif split == "test":
            csv_name = "b. IDRiD_Disease Grading_Testing Labels.csv"
            images_dir = root / "1. Original Images" / "b. Testing Set"
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        csv_path = labels_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"IDRiD labels not found at {csv_path}.")
        df = pd.read_csv(csv_path)
        col_map = {}
        for col in df.columns:
            if col.lower().replace(" ", "").startswith("image"):
                col_map[col] = "id_code"
            elif "retinopathy" in col.lower() and "grade" in col.lower():
                col_map[col] = "label"
        df = df.rename(columns=col_map)
        df = df[["id_code", "label"]].copy()
        df["label"] = df["label"].astype(int)
        return df, images_dir

    else:
        raise ValueError(f"Unknown IDRiD layout {layout!r}")


class IDRiDDataset(Dataset):
    """IDRiD as a PyTorch Dataset matching the APTOS2019Dataset interface."""

    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_dir: Path,
        transform: Callable | None = None,
    ) -> None:
        self.labels = labels_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Sanity check first few image files exist
        missing = []
        for _, row in self.labels.head(3).iterrows():
            if not (self.images_dir / f"{row['id_code']}.jpg").exists():
                missing.append(row["id_code"])
        if missing:
            raise FileNotFoundError(
                f"Sample image files not found under {self.images_dir}: {missing}."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        row = self.labels.iloc[idx]
        img_path = self.images_dir / f"{row['id_code']}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            import numpy as np
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            import numpy as np
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": int(row["label"]),
            "id_code": str(row["id_code"]),
        }


def build_idrid_test_dataset(
    root: Path | None = None,
    transform: Callable | None = None,
) -> IDRiDDataset:
    """Build the IDRiD test set. Auto-detects layout.

    Searches in order: provided `root` -> canonical default -> kaggle-flat default.
    """
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root))
    candidates.extend([IDRID_CANONICAL_ROOT, IDRID_KAGGLE_FLAT_ROOT])

    last_err = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            labels_df, images_dir = load_idrid_labels(candidate, split="test")
            return IDRiDDataset(labels_df, images_dir, transform=transform)
        except (FileNotFoundError, ValueError) as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"Could not locate IDRiD dataset. Tried: {[str(c) for c in candidates]}. "
        f"Last error: {last_err}"
    )
