"""Training loop — minimal, deterministic, paper-grade.

Design choices:
  * Mixed precision via torch.amp (Kaggle T4 / A100 friendly).
  * Cosine LR with linear warmup.
  * Early stopping on val quadratic-weighted kappa (the APTOS competition metric)
    AND val loss; we use kappa for selection because it correlates with the
    ordinal nature of DR grading.
  * Per-epoch W&B logging if WANDB_API_KEY is set, otherwise silent.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


@dataclass
class TrainState:
    epoch: int = 0
    best_kappa: float = -1.0
    best_epoch: int = 0
    epochs_without_improvement: int = 0


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss, total_n = 0.0, 0
    preds, targets = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    kappa = cohen_kappa_score(targets, preds, weights="quadratic")
    accuracy = (preds == targets).mean()
    return {
        "loss": total_loss / max(1, total_n),
        "kappa": float(kappa),
        "accuracy": float(accuracy),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    base_lr: float,
    weight_decay: float,
    warmup_epochs: int,
    label_smoothing: float,
    mixed_precision: bool,
    gradient_clip: float,
    early_stopping_patience: int,
    output_dir: str | Path,
    device: torch.device,
    wandb_run=None,
) -> TrainState:
    """Run training. Saves best checkpoint to `output_dir/best.pt`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    state = TrainState()
    global_step = 0

    for epoch in range(epochs):
        state.epoch = epoch
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * x.size(0)
            epoch_n += x.size(0)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        train_loss = epoch_loss / max(1, epoch_n)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch + 1:>3}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_kappa={val_metrics['kappa']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"({elapsed:.1f}s)"
        )

        if wandb_run is not None and _WANDB_AVAILABLE:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/kappa": val_metrics["kappa"],
                "val/accuracy": val_metrics["accuracy"],
                "lr": lr,
            })

        if val_metrics["kappa"] > state.best_kappa:
            state.best_kappa = val_metrics["kappa"]
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_kappa": val_metrics["kappa"],
                "val_loss": val_metrics["loss"],
            }, output_dir / "best.pt")
        else:
            state.epochs_without_improvement += 1
            if state.epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(best epoch {state.best_epoch + 1}, "
                      f"kappa={state.best_kappa:.4f})")
                break

    return state
