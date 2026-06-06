"""Classification backbone with an explicit, MC-Dropout-friendly head.

We deliberately attach our own dropout + linear head instead of using timm's
built-in classifier, because:
  1. We need dropout active at inference time for MC Dropout later.
  2. We want a single point to swap heads (e.g., for Evidential outputs).
"""
from __future__ import annotations

import timm
import torch
from torch import nn


class RetinalClassifier(nn.Module):
    """Backbone + dropout + linear head.

    Args:
        backbone: timm model name (e.g., "efficientnet_b0", "convnext_tiny").
        num_classes: Output classes.
        pretrained: Load ImageNet weights.
        dropout: Dropout prob applied before the final linear layer.
            Kept active at inference for MC Dropout sampling.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # num_classes=0 → timm returns pooled features instead of logits
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(feat_dim, num_classes)
        self._backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.head(feats)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled features (no dropout, no head) — useful for diagnostics."""
        return self.backbone(x)

    def enable_mc_dropout(self) -> None:
        """Force dropout active even during model.eval() — for MC sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @property
    def name(self) -> str:
        return self._backbone_name
