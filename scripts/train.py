"""Training entry point.

Usage:
    python scripts/train.py --config configs/baseline.yaml \
        --splits-dir data/splits/aptos2019 \
        --images-dir /path/to/aptos/train_images
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.aptos import APTOS2019Dataset
from src.data.splits import load_splits
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.backbone import RetinalClassifier
from src.training.trainer import train
from src.utils.config import load_config, save_config
from src.utils.seed import seed_everything

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B even if API key is set.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    train_df, val_df, _test_df = load_splits(args.splits_dir)
    train_ds = APTOS2019Dataset(
        train_df, args.images_dir,
        transform=build_train_transform(cfg.data.image_size),
    )
    val_ds = APTOS2019Dataset(
        val_df, args.images_dir,
        transform=build_eval_transform(cfg.data.image_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    # ---- Model ----
    model = RetinalClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name}  |  params: {n_params / 1e6:.2f}M")

    # ---- Output dir + config snapshot ----
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    # ---- W&B ----
    wandb_run = None
    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity"),
            name=cfg.experiment.name,
            tags=list(cfg.experiment.get("tags", [])),
            config=dict(cfg),
            dir=str(output_dir),
        )

    # ---- Train ----
    state = train(
        model, train_loader, val_loader,
        epochs=cfg.training.epochs,
        base_lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        label_smoothing=cfg.training.label_smoothing,
        mixed_precision=cfg.training.mixed_precision,
        gradient_clip=cfg.training.gradient_clip,
        early_stopping_patience=cfg.training.early_stopping_patience,
        output_dir=output_dir,
        device=device,
        wandb_run=wandb_run,
    )

    print(f"\nDone. Best val kappa = {state.best_kappa:.4f} at epoch {state.best_epoch + 1}.")
    print(f"Checkpoint: {output_dir / 'best.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
