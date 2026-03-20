"""Classification heads for marine species recognition.

Provides linear and MLP classification heads that attach to
foundation model feature extractors.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Classification head for feature-based classification.

    Supports both simple linear projection and multi-layer MLP heads.

    Args:
        in_features: Input feature dimension from backbone.
        num_classes: Number of output classes.
        head_type: 'linear' for single linear layer, 'mlp' for 2-layer MLP.
        hidden_dim: Hidden dimension for MLP head. Defaults to in_features.
        dropout: Dropout probability for MLP head.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        head_type: str = "linear",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.head_type = head_type

        if head_type == "linear":
            self.head = nn.Linear(in_features, num_classes)
        elif head_type == "mlp":
            hidden = hidden_dim or in_features
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Use 'linear' or 'mlp'.")

        self._init_weights()

        n_params = sum(p.numel() for p in self.head.parameters())
        logger.info(
            "Classification head (%s): %d -> %d, params=%s",
            head_type,
            in_features,
            num_classes,
            f"{n_params:,}",
        )

    def _init_weights(self) -> None:
        """Initialize head weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, in_features].

        Returns:
            Class logits [B, num_classes].
        """
        return self.head(x)
