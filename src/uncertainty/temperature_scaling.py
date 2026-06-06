"""Temperature scaling — post-hoc softmax calibration.

Guo et al. (ICML 2017): introduce a single scalar T on the logits before
softmax, then optimize T on a held-out *validation* set to minimize NLL.
T > 1 softens overconfident softmax outputs; T < 1 sharpens.

This is the simplest, cheapest, and often surprisingly competitive
calibration method. We use LBFGS as in the original paper.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class TemperatureScaler(nn.Module):
    """Wraps logits with a learnable scalar temperature.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)
        calibrated_probs = scaler.calibrate_probs(test_logits)
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        # Initialize as raw float; ensure positivity via softplus at call time
        # so optimization is unconstrained.
        self._raw_t = nn.Parameter(torch.tensor(float(init_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        # Softplus + tiny eps keeps T strictly positive
        return torch.nn.functional.softplus(self._raw_t) + 1e-6

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def calibrate_probs(self, logits: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        scaled = self.forward(logits.float())
        return torch.softmax(scaled, dim=1).cpu().numpy()

    def fit(
        self,
        logits: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        max_iter: int = 200,
        lr: float = 0.01,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        """Fit temperature on (logits, labels) by minimizing cross-entropy.

        Returns a dict with pre/post NLL for diagnostics.
        """
        device = device or torch.device("cpu")
        self.to(device)

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        logits = logits.float().to(device)
        labels = labels.long().to(device)

        loss_fn = nn.CrossEntropyLoss()
        pre_nll = loss_fn(logits, labels).item()

        optimizer = torch.optim.LBFGS([self._raw_t], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = loss_fn(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        post_nll = loss_fn(self.forward(logits), labels).item()
        return {
            "temperature": float(self.temperature.item()),
            "nll_before": float(pre_nll),
            "nll_after": float(post_nll),
        }


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Helper: get (logits, labels, id_codes) for a deterministic loader."""
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["id_code"])
    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_ids,
    )
