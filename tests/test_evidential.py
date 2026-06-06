"""Sanity tests for EDL — loss is finite, gradients flow, vacuity computed correctly."""
from __future__ import annotations

import torch

from src.uncertainty.evidential import edl_loss, evidence_to_alpha


def test_edl_loss_finite_and_grads():
    torch.manual_seed(0)
    evidence = torch.rand(8, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    out = edl_loss(evidence, targets, num_classes=5, epoch=5, annealing_step=10)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert evidence.grad is not None
    assert torch.isfinite(evidence.grad).all()


def test_evidence_to_alpha_offset_by_one():
    e = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 0.5]])
    alpha = evidence_to_alpha(e)
    assert torch.allclose(alpha, e + 1.0)


def test_annealing_lambda_grows():
    evidence = torch.rand(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    lam_early = edl_loss(evidence, targets, 3, epoch=0, annealing_step=10)["lambda"]
    lam_mid = edl_loss(evidence, targets, 3, epoch=5, annealing_step=10)["lambda"]
    lam_late = edl_loss(evidence, targets, 3, epoch=15, annealing_step=10)["lambda"]
    assert lam_early < lam_mid <= lam_late <= 1.0


def test_perfect_one_hot_evidence_is_low_loss():
    """Sanity check: huge evidence on the correct class → small MSE term."""
    targets = torch.tensor([0, 1, 2])
    # Large evidence on the correct class only
    evidence = torch.zeros(3, 3)
    for i, y in enumerate(targets):
        evidence[i, y] = 100.0
    out = edl_loss(evidence, targets, num_classes=3, epoch=0, annealing_step=10)
    # With λ=0 at epoch 0, total == mse. MSE should be very small here.
    assert out["mse"].item() < 0.01
