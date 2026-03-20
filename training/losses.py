"""Loss functions for classification and detection tasks."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in marine species datasets.

    From Lin et al. "Focal Loss for Dense Object Detection" (2017).
    Down-weights well-classified examples to focus on hard negatives.

    Args:
        alpha: Weighting factor for each class. If float, applied uniformly.
        gamma: Focusing parameter (higher = more focus on hard examples).
        reduction: 'mean', 'sum', or 'none'.
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model predictions (logits) [B, C].
            targets: Ground truth labels [B].

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_function(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory function for loss functions.

    Args:
        loss_type: 'cross_entropy' or 'focal'.
        label_smoothing: Label smoothing factor.
        focal_gamma: Gamma for focal loss.
        class_weights: Optional per-class weights for imbalanced datasets.

    Returns:
        Loss function module.
    """
    if loss_type == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Using CrossEntropyLoss (label_smoothing=%.2f, weighted=%s)",
            label_smoothing,
            class_weights is not None,
        )
    elif loss_type == "focal":
        loss_fn = FocalLoss(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        logger.info("Using FocalLoss (gamma=%.1f)", focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn
