"""Unified model creation factory from YAML configuration.

Creates any model + adaptation combination from config files,
providing a single entry point for all experiments.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from models.baseline_models import BaselineCNN
from models.foundation_wrapper import FoundationModelWrapper

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating models from YAML configurations.

    Loads model and adaptation configs and creates the appropriate
    model instance (foundation model wrapper or baseline CNN).

    Args:
        config_dir: Path to the configs/ directory.
    """

    FOUNDATION_FAMILIES = {"dinov2", "dinov3", "clip", "mae"}
    CNN_FAMILIES = {"cnn"}

    def __init__(self, config_dir: str = "./configs") -> None:
        self.config_dir = Path(config_dir)

    def load_model_config(self, model_name: str) -> DictConfig:
        """Load a model configuration YAML.

        Args:
            model_name: Model config filename (without .yaml extension).

        Returns:
            Model configuration as DictConfig.
        """
        config_path = self.config_dir / "models" / f"{model_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        return cfg.model

    def load_adaptation_config(self, adaptation_name: str) -> DictConfig:
        """Load an adaptation strategy configuration YAML.

        Args:
            adaptation_name: Adaptation config filename (without .yaml).

        Returns:
            Adaptation configuration as DictConfig.
        """
        config_path = self.config_dir / "adaptation" / f"{adaptation_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Adaptation config not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        return cfg.adaptation

    def create(
        self,
        model_name: str,
        adaptation_name: str,
        num_classes: int,
        model_cfg: Optional[DictConfig] = None,
        adaptation_cfg: Optional[DictConfig] = None,
    ) -> nn.Module:
        """Create a model with the specified adaptation strategy.

        Args:
            model_name: Name of the model config.
            adaptation_name: Name of the adaptation config.
            num_classes: Number of output classes.
            model_cfg: Optional pre-loaded model config (overrides file load).
            adaptation_cfg: Optional pre-loaded adaptation config.

        Returns:
            Configured model ready for training.
        """
        if model_cfg is None:
            model_cfg = self.load_model_config(model_name)
        if adaptation_cfg is None:
            adaptation_cfg = self.load_adaptation_config(adaptation_name)

        family = model_cfg.family

        logger.info(
            "Creating model: %s + %s (family=%s, classes=%d)",
            model_name,
            adaptation_name,
            family,
            num_classes,
        )

        if family in self.FOUNDATION_FAMILIES:
            model = FoundationModelWrapper(
                model_cfg=model_cfg,
                adaptation_cfg=adaptation_cfg,
                num_classes=num_classes,
            )
        elif family in self.CNN_FAMILIES:
            model = BaselineCNN(
                model_cfg=model_cfg,
                adaptation_cfg=adaptation_cfg,
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown model family: {family}. "
                f"Supported: {self.FOUNDATION_FAMILIES | self.CNN_FAMILIES}"
            )

        return model

    def create_from_experiment_config(
        self,
        experiment_cfg: DictConfig,
        model_name: str,
        adaptation_name: str,
        num_classes: int,
    ) -> nn.Module:
        """Create a model using an experiment configuration.

        Args:
            experiment_cfg: Experiment configuration.
            model_name: Model name from experiment config models list.
            adaptation_name: Adaptation name from experiment config.
            num_classes: Number of output classes.

        Returns:
            Configured model.
        """
        return self.create(model_name, adaptation_name, num_classes)


def create_model(
    model_name: str,
    adaptation_name: str,
    num_classes: int,
    config_dir: str = "./configs",
) -> nn.Module:
    """Convenience function to create a model.

    Args:
        model_name: Model config name (e.g., 'dinov2_base').
        adaptation_name: Adaptation config name (e.g., 'lora_r4').
        num_classes: Number of output classes.
        config_dir: Path to configs directory.

    Returns:
        Configured model ready for training.
    """
    factory = ModelFactory(config_dir)
    return factory.create(model_name, adaptation_name, num_classes)
