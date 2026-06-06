"""Image transforms — Albumentations pipelines for train and eval.

Eval transform is deterministic and used for val, internal test, and external sets.
Train transform uses moderate retinal-image-appropriate augmentation.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet stats — backbones are ImageNet-pretrained
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> A.Compose:
    """Training-time augmentation for fundus images.

    Conservative choices: no aggressive color jitter (lesion colors matter),
    no vertical flip (anatomy has top/bottom orientation in some grading schemes),
    horizontal flip is fine (eye laterality is not a grading signal here).
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int = 224) -> A.Compose:
    """Deterministic eval transform — used for val, test, and external datasets."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, value=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
