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

## Where to get IDRiD on Kaggle

Attach the dataset `mariaherrerot/idrid-dataset` to your Kaggle notebook
(right panel -> Add Data -> search "idrid"). It mounts at
`/kaggle/input/idrid-dataset/` and contains:

    /kaggle/input/idrid-dataset/
        B. Disease Grading/
            1. Original Images/
                a. Training Set/      (413 images, train)
                b. Testing Set/       (103 images, test)
            2. Groundtruths/
                a. IDRiD_Disease Grading_Training Labels.csv
                b. IDRiD_Disease Grading_Testing Labels.csv

We use the IDRiD test set (103 images) as the **external validation set**.
The IDRiD train set is unused here.

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

# Default mount points if attached on Kaggle
IDRID_DEFAULT_ROOT = Path("/kaggle/input/idrid-dataset/B. Disease Grading")
IDRID_DEFAULT_IMAGES = IDRID_DEFAULT_ROOT / "1. Original Images"
IDRID_DEFAULT_LABELS = IDRID_DEFAULT_ROOT / "2. Groundtruths"


def load_idrid_labels(labels_dir: Path, split: str = "test") -> pd.DataFrame:
    """Load IDRiD labels CSV and return a normalized DataFrame.

    Parameters
    ----------
    labels_dir : Path
        Directory containing IDRiD label CSVs.
    split : {"train", "test"}
        Which split to load.

    Returns
    -------
    DataFrame with columns: id_code, label (int 0-4)
    """
    if split == "train":
        csv_name = "a. IDRiD_Disease Grading_Training Labels.csv"
    elif split == "test":
        csv_name = "b. IDRiD_Disease Grading_Testing Labels.csv"
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    csv_path = Path(labels_dir) / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"IDRiD labels not found at {csv_path}. "
            f"Make sure the 'mariaherrerot/idrid-dataset' Kaggle dataset is attached."
        )

    df = pd.read_csv(csv_path)
    # IDRiD label CSV uses different column names — normalize them
    # Original columns: "Image name", "Retinopathy grade", "Risk of macular edema"
    col_map = {}
    for col in df.columns:
        if col.lower().replace(" ", "").startswith("image"):
            col_map[col] = "id_code"
        elif "retinopathy" in col.lower() and "grade" in col.lower():
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    if "id_code" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Could not find id_code and label columns in {csv_path}. "
            f"Got columns: {list(df.columns)}"
        )
    df = df[["id_code", "label"]].copy()
    df["label"] = df["label"].astype(int)
    return df


class IDRiDDataset(Dataset):
    """IDRiD test set as a PyTorch Dataset.

    Returns dicts with keys 'image' (tensor C x H x W), 'label' (int),
    'id_code' (str). Matches the `APTOS2019Dataset` interface so the same
    evaluation harness can be used unchanged.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_dir: Path,
        transform: Callable | None = None,
    ) -> None:
        self.labels = labels_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Sanity check a few image files exist
        missing = []
        for _, row in self.labels.head(3).iterrows():
            if not (self.images_dir / f"{row['id_code']}.jpg").exists():
                missing.append(row["id_code"])
        if missing:
            raise FileNotFoundError(
                f"Sample image files not found under {self.images_dir}: {missing}. "
                f"Check that IDRiD images directory is correct."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        row = self.labels.iloc[idx]
        img_path = self.images_dir / f"{row['id_code']}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            # Albumentations expects numpy HWC uint8
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
    """Convenience constructor for the IDRiD test set with default Kaggle paths."""
    if root is None:
        root = IDRID_DEFAULT_ROOT
    root = Path(root)
    images_dir = root / "1. Original Images" / "b. Testing Set"
    labels_dir = root / "2. Groundtruths"
    labels_df = load_idrid_labels(labels_dir, split="test")
    return IDRiDDataset(labels_df, images_dir, transform=transform)
