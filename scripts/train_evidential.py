"""Train an Evidential Deep Learning model.

Same data + backbone as baseline, different head + loss. Selects on val
accuracy (NOT QWK) because EDL probs are Dirichlet means and the loss is
not directly comparable across epochs in the early-annealing regime.

Usage:
    python scripts/train_evidential.py \
        --config configs/evidential.yaml \
        --splits-dir data/splits/aptos2019 \
        --images-dir /path/to/aptos/train_images
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.aptos import APTOS2019Dataset
from src.data.splits import load_splits
from src.data.transforms import build_eval_transform, build_train_transform
from src.uncertainty.evidential import (
    EvidentialClassifier,
    edl_loss,
    evidence_to_alpha,
)
from src.utils.config import load_config, save_config
from src.utils.seed import seed_everything

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def cosine_warmup_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_edl(model, loader, device, num_classes):
    model.eval()
    total_correct, total_n = 0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        evidence = model(x)
        alpha = evidence_to_alpha(evidence)
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        preds = probs.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_n += y.size(0)
    return {"accuracy": total_correct / max(1, total_n)}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df, val_df, _ = load_splits(args.splits_dir)
    train_ds = APTOS2019Dataset(train_df, args.images_dir,
                                transform=build_train_transform(cfg.data.image_size))
    val_ds = APTOS2019Dataset(val_df, args.images_dir,
                              transform=build_eval_transform(cfg.data.image_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)

    model = EvidentialClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model._backbone_name} + EvidentialHead  |  params: {n_params / 1e6:.2f}M")

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity"),
            name=cfg.experiment.name,
            tags=list(cfg.experiment.get("tags", [])),
            config=dict(cfg),
            dir=str(output_dir),
        )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision and device.type == "cuda")
    total_steps = cfg.training.epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)
    annealing_step = cfg.training.get("annealing_step", 10)

    best_acc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.training.epochs}", leave=False)
        for batch in pbar:
            lr = cosine_warmup_lr(step, total_steps, warmup_steps, cfg.training.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda",
                                    enabled=cfg.training.mixed_precision and device.type == "cuda"):
                evidence = model(x)
                losses = edl_loss(evidence, y, cfg.model.num_classes,
                                  epoch=epoch, annealing_step=annealing_step)
                loss = losses["total"]

            scaler.scale(loss).backward()
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * x.size(0)
            epoch_n += x.size(0)
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             mse=f"{losses['mse'].item():.3f}",
                             kl=f"{losses['kl'].item():.3f}",
                             lam=f"{losses['lambda']:.2f}")

        val_metrics = evaluate_edl(model, val_loader, device, cfg.model.num_classes)
        train_loss = epoch_loss / max(1, epoch_n)
        print(f"[epoch {epoch + 1}/{cfg.training.epochs}] "
              f"train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"({time.time() - t0:.1f}s)")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/accuracy": val_metrics["accuracy"],
                "lr": lr,
            })

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": val_metrics["accuracy"],
            }, output_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(best epoch {best_epoch + 1}, acc={best_acc:.4f})")
                break

    print(f"\nDone. Best val accuracy = {best_acc:.4f} at epoch {best_epoch + 1}.")
    print(f"Checkpoint: {output_dir / 'best.pt'}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
