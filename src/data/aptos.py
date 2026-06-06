"""APTOS 2019 dataset.

Kaggle: https://www.kaggle.com/competitions/aptos2019-blindness-detection
Labels (5 classes):
  0 - No DR
  1 - Mild
  2 - Moderate
  3 - Severe
  4 - Proliferative DR

The Kaggle download gives:
  train.csv             — id_code, diagnosis
  train_images/         — {id_code}.png
  test.csv / test_images — unlabeled (competition only); we ignore these.

We re-split train.csv into train/val/test ourselves (stratified, fixed seed)
so we have ground-truth labels for our internal test set.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Albumentations compose is the expected transform type, but we keep the
# annotation loose to avoid importing it just for typing.
Transform = object


class APTOS2019Dataset(Dataset):
    """APTOS 2019 fundus image dataset.

    Args:
        df: DataFrame with columns `id_code` and `diagnosis`.
        images_dir: Directory containing `{id_code}.png` files.
        transform: Albumentations transform returning a tensor under key "image".
    """

    NUM_CLASSES = 5
    CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        transform: Transform | None = None,
    ) -> None:
        required = {"id_code", "diagnosis"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['id_code']}.png"

        # OpenCV reads BGR; convert to RGB for ImageNet-pretrained backbones.
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            rgb = self.transform(image=rgb)["image"]
        else:
            # Fallback: plain tensor in CHW float32 [0, 1]
            rgb = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)

        label = int(row["diagnosis"])

        return {
            "image": rgb,
            "label": torch.tensor(label, dtype=torch.long),
            "id_code": str(row["id_code"]),
        }

    @staticmethod
    def class_weights(df: pd.DataFrame) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced training (optional)."""
        counts = df["diagnosis"].value_counts().sort_index()
        weights = counts.sum() / (len(counts) * counts.values)
        return torch.tensor(weights, dtype=torch.float32)
