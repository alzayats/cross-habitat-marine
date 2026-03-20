"""Optimizer and scheduler factories with layer-wise learning rate decay."""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    layer_decay: Optional[float] = None,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Create an optimizer with optional layer-wise learning rate decay.

    Args:
        model: The model to optimize.
        optimizer_type: 'adamw' or 'sgd'.
        learning_rate: Base learning rate.
        weight_decay: Weight decay factor.
        layer_decay: Layer-wise LR decay factor for ViTs. If None, uniform LR.
        momentum: Momentum for SGD.

    Returns:
        Configured optimizer.
    """
    if layer_decay is not None and layer_decay < 1.0:
        param_groups = _get_layer_decay_params(
            model, learning_rate, weight_decay, layer_decay
        )
    else:
        # Standard parameter groups: decay and no-decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm", "layernorm", "norm", "bn"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    if optimizer_type == "adamw":
        optimizer = AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.999))
    elif optimizer_type == "sgd":
        optimizer = SGD(
            param_groups, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    trainable = sum(p.numel() for group in param_groups for p in group["params"])
    logger.info(
        "Optimizer: %s, lr=%.2e, weight_decay=%.4f, trainable_params=%s",
        optimizer_type,
        learning_rate,
        weight_decay,
        f"{trainable:,}",
    )

    return optimizer


def _get_layer_decay_params(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> List[Dict[str, Any]]:
    """Create parameter groups with layer-wise learning rate decay.

    Deeper layers get higher learning rates. The classification head
    gets the base learning rate.

    Args:
        model: Model with named parameters.
        base_lr: Base learning rate for the top layer.
        weight_decay: Weight decay.
        layer_decay: Multiplicative decay factor per layer.

    Returns:
        List of parameter group dicts.
    """
    param_groups: Dict[str, Dict[str, Any]] = {}

    # Determine number of layers
    num_layers = 12  # Default for ViT-Base
    for name, _ in model.named_parameters():
        if "blocks." in name or "layer." in name or "layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in ("blocks", "layer", "layers") and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        num_layers = max(num_layers, layer_idx + 1)
                    except ValueError:
                        pass

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer index
        layer_idx = _get_layer_index(name, num_layers)

        # Compute layer-specific learning rate
        lr_scale = layer_decay ** (num_layers - layer_idx)
        lr = base_lr * lr_scale

        # Determine weight decay
        wd = 0.0 if any(
            nd in name for nd in ["bias", "LayerNorm", "layernorm", "norm", "bn"]
        ) else weight_decay

        group_key = f"layer_{layer_idx}_wd_{wd}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": lr,
                "weight_decay": wd,
            }
        param_groups[group_key]["params"].append(param)

    groups = list(param_groups.values())
    logger.info(
        "Layer-wise LR decay: %d layers, decay=%.2f, %d param groups",
        num_layers,
        layer_decay,
        len(groups),
    )
    return groups


def _get_layer_index(name: str, num_layers: int) -> int:
    """Determine the layer index for a named parameter.

    Args:
        name: Parameter name.
        num_layers: Total number of layers.

    Returns:
        Layer index (0 = embedding, num_layers = head).
    """
    if any(kw in name for kw in ["head", "classifier", "fc"]):
        return num_layers  # Top layer

    if any(kw in name for kw in ["patch_embed", "cls_token", "pos_embed", "embed"]):
        return 0  # Embedding layer

    # Extract block/layer index
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in ("blocks", "layer", "layers") and i + 1 < len(parts):
            try:
                return int(parts[i + 1]) + 1
            except ValueError:
                pass

    return num_layers // 2  # Default: middle layer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 30,
    warmup_fraction: float = 0.05,
    steps_per_epoch: int = 100,
    min_lr: float = 1e-7,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create a learning rate scheduler with optional warmup.

    Args:
        optimizer: The optimizer.
        scheduler_type: 'cosine' or 'linear'.
        num_epochs: Total number of training epochs.
        warmup_fraction: Fraction of total steps for warmup.
        steps_per_epoch: Number of optimizer steps per epoch.
        min_lr: Minimum learning rate.

    Returns:
        Learning rate scheduler.
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_fraction)

    if warmup_steps > 0:
        # Warmup scheduler
        def warmup_fn(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)

        # Main scheduler
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr,
            )
        else:
            def linear_fn(step: int) -> float:
                remaining = total_steps - warmup_steps
                return max(0.0, 1.0 - step / remaining)

            main_scheduler = LambdaLR(optimizer, lr_lambda=linear_fn)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    else:
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=min_lr
            )
        else:
            def linear_fn(step: int) -> float:
                return max(0.0, 1.0 - step / total_steps)

            scheduler = LambdaLR(optimizer, lr_lambda=linear_fn)

    logger.info(
        "Scheduler: %s, total_steps=%d, warmup_steps=%d",
        scheduler_type,
        total_steps,
        warmup_steps,
    )

    return scheduler
