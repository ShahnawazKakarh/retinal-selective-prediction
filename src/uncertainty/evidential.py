"""Evidential Deep Learning (Sensoy, Kaplan & Kandemir, NeurIPS 2018).

The network outputs **evidence** e_k ≥ 0 per class via softplus on raw logits.
A Dirichlet prior is induced: α_k = e_k + 1, total strength S = Σ α_k.

The predicted probability per class is the Dirichlet mean: p_k = α_k / S.
The total uncertainty (vacuity) is u = K / S, with K = number of classes.

Loss (Eq. 5 of the paper) decomposes into an expected-MSE term plus a KL
regularizer that drives evidence towards 0 for the *wrong* classes, scaled
by an annealing coefficient λ_t that grows over epochs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EvidentialHead(nn.Module):
    """Replace the standard linear head with a softplus-evidence head.

    Drop-in replacement: same input dim, same num_classes. The model returns
    raw evidence (>= 0); convert to α with `evidence_to_alpha(evidence)`.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softplus keeps evidence non-negative and differentiable.
        return torch.nn.functional.softplus(self.linear(x))


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def edl_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    epoch: int,
    annealing_step: int = 10,
) -> dict[str, torch.Tensor]:
    """Sensoy et al. Eq. 5 — MSE + annealed KL regularizer.

    Args:
        evidence: (B, C) softplus output, non-negative.
        targets: (B,) integer class labels.
        num_classes: K.
        epoch: 0-indexed training epoch.
        annealing_step: How quickly λ ramps from 0 → 1.

    Returns:
        Dict with `total`, `mse`, `kl` losses (each scalar Tensor).
    """
    alpha = evidence_to_alpha(evidence)
    S = alpha.sum(dim=1, keepdim=True)
    y_onehot = torch.nn.functional.one_hot(targets, num_classes).float()

    # Expected MSE (Eq. 5)
    err = (y_onehot - alpha / S) ** 2
    var = alpha * (S - alpha) / (S * S * (S + 1))
    mse = (err + var).sum(dim=1).mean()

    # KL regularizer: KL(Dir(alpha_tilde) || Dir(1)), where alpha_tilde is the
    # Dirichlet parameter vector with the true-class evidence removed.
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    K = float(num_classes)

    kl = (
        torch.lgamma(S_tilde).squeeze(1)
        - torch.lgamma(torch.tensor(K, device=evidence.device))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + (
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
        ).sum(dim=1)
    ).mean()

    lam = min(1.0, float(epoch) / float(max(1, annealing_step)))
    total = mse + lam * kl
    return {"total": total, "mse": mse.detach(), "kl": kl.detach(), "lambda": lam}


@torch.no_grad()
def evidential_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, np.ndarray]:
    """Inference: extract Dirichlet mean probs + vacuity uncertainty."""
    model.eval()
    all_probs, all_alpha, all_unc = [], [], []
    all_labels, all_ids = [], []
    for batch in tqdm(loader, desc="Evidential inference"):
        x = batch["image"].to(device, non_blocking=True)
        evidence = model(x)
        alpha = evidence_to_alpha(evidence)
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        # Vacuity (total uncertainty): K / S
        vacuity = num_classes / S.squeeze(1)

        all_probs.append(probs.cpu().numpy())
        all_alpha.append(alpha.cpu().numpy())
        all_unc.append(vacuity.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["id_code"])

    return {
        "probs_mean": np.concatenate(all_probs).astype(np.float32),
        "alpha": np.concatenate(all_alpha).astype(np.float32),
        "vacuity": np.concatenate(all_unc).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "id_codes": all_ids,
    }


class EvidentialClassifier(nn.Module):
    """Backbone + EvidentialHead — same shape as RetinalClassifier."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)
        self.head = EvidentialHead(feat_dim, num_classes)
        self.num_classes = num_classes
        self._backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.head(feats)
