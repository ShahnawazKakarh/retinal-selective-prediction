"""Global seeding for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Master seed.
        deterministic: If True, force cuDNN deterministic algorithms.
            Slower, but required for paper-grade reproducibility.

    Returns:
        The seed actually used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed
