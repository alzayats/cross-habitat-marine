"""Baseline CNN models (ResNet50, EfficientNet-B4) for comparison.

These serve as non-foundation-model baselines for the cross-habitat
transfer experiments.
"""

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn
import torchvision.models as tv_models
from omegaconf import DictConfig

from models.classification_head import ClassificationHead

logger = logging.getLogger(__name__)


class BaselineCNN(nn.Module):
    """Baseline CNN model with feature extraction and classification head.

    Wraps torchvision or timm CNN models with a consistent interface
    matching FoundationModelWrapper.

    Args:
        model_cfg: Model configuration.
        adaptation_cfg: Adaptation configuration.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        adaptation_cfg: DictConfig,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.adaptation_cfg = adaptation_cfg
        self.num_classes = num_classes
        self.feature_dim = model_cfg.feature_dim

        self.backbone = self._create_backbone()
        self._apply_adaptation()

        head_type = getattr(adaptation_cfg, "head_type", "linear")
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            head_type=head_type,
        )

        self._log_param_counts()

    def _create_backbone(self) -> nn.Module:
        """Create the CNN backbone."""
        arch = self.model_cfg.architecture
        source = self.model_cfg.source
        pretrained = self.model_cfg.pretrained

        if source == "torchvision" and arch == "resnet50":
            return self._create_resnet50(pretrained)
        elif source == "timm":
            return self._create_timm_model(pretrained)
        else:
            raise ValueError(f"Unsupported CNN: source={source}, arch={arch}")

    def _create_resnet50(self, pretrained: bool) -> nn.Module:
        """Create ResNet50 backbone."""
        if pretrained:
            weights = self.model_cfg.get("pretrained_weights", "IMAGENET1K_V2")
            logger.info("Loading ResNet50 with weights: %s", weights)
            model = tv_models.resnet50(weights=weights)
        else:
            logger.info("Loading ResNet50 with random initialization")
            model = tv_models.resnet50(weights=None)

        # Remove the final FC layer; keep everything up to avgpool
        modules = list(model.children())[:-1]  # Remove fc
        backbone = nn.Sequential(*modules)
        # Add flatten
        backbone = nn.Sequential(backbone, nn.Flatten())
        return backbone

    def _create_timm_model(self, pretrained: bool) -> nn.Module:
        """Create a timm model backbone."""
        model_name = self.model_cfg.get("timm_model_name", self.model_cfg.architecture)
        logger.info("Loading timm model: %s (pretrained=%s)", model_name, pretrained)

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
        )
        return model

    def _apply_adaptation(self) -> None:
        """Apply adaptation strategy (freeze/unfreeze)."""
        adapt_type = self.adaptation_cfg.type

        if adapt_type in ("linear_probe",):
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("CNN backbone frozen for linear probing")
        elif adapt_type == "full_finetune":
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("CNN backbone unfrozen for full fine-tuning")
        elif adapt_type in ("lora", "vpt"):
            # LoRA/VPT not applicable to CNNs; fall back to full fine-tune
            logger.warning(
                "%s not applicable to CNNs. Using full fine-tuning instead.",
                adapt_type,
            )
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown adaptation type: {adapt_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classification head.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        features = self.get_features(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Feature tensor [B, feature_dim].
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def _log_param_counts(self) -> None:
        """Log parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        logger.info(
            "Baseline CNN params: total=%s, trainable=%s (%.2f%%), frozen=%s",
            f"{total:,}",
            f"{trainable:,}",
            100.0 * trainable / total if total > 0 else 0,
            f"{frozen:,}",
        )

        self._param_counts = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
        }

    @property
    def param_counts(self):
        return self._param_counts


def create_baseline_model(
    model_cfg: DictConfig,
    adaptation_cfg: DictConfig,
    num_classes: int,
) -> BaselineCNN:
    """Factory function for baseline CNN models.

    Args:
        model_cfg: Model configuration.
        adaptation_cfg: Adaptation configuration.
        num_classes: Number of classes.

    Returns:
        Configured BaselineCNN instance.
    """
    return BaselineCNN(model_cfg, adaptation_cfg, num_classes)
