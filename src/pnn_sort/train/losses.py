"""Training objectives for list sorting (MSE, Huber, Monotonicity hinge, Moment/W1 matching)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sorting_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str = "mse",
    mono_weight: float = 0.0,
    w1_weight: float = 0.0,
) -> torch.Tensor:
    """Compute value-sorting loss with optional monotonicity hinge and multiset W1 terms."""
    if loss_name == "mse":
        base = F.mse_loss(pred, target)
    elif loss_name == "huber":
        base = F.huber_loss(pred, target, delta=0.05)
    elif loss_name == "l1":
        base = F.l1_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    total = base
    if mono_weight > 0.0:
        # Penalize adjacent order violations: pred[:, k] > pred[:, k+1]
        inversions = F.relu(pred[:, :-1] - pred[:, 1:])
        total = total + mono_weight * inversions.mean()
    if w1_weight > 0.0:
        # Multiset Wasserstein-1 loss (sort predicted values and match target multiset)
        pred_sorted = torch.sort(pred, dim=-1).values
        total = total + w1_weight * F.l1_loss(pred_sorted, target)
    return total
