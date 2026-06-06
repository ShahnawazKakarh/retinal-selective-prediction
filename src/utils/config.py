"""Config loading via OmegaConf. One YAML = one reproducible run."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file into a DictConfig.

    Resolves OmegaConf interpolations (e.g., ${seed}) eagerly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a config snapshot alongside experiment outputs for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
