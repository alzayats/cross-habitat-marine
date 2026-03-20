"""LoRA (Low-Rank Adaptation) injection for Vision Transformers.

Uses the HuggingFace PEFT library to inject LoRA adapters into
attention Q, V (or fused QKV) matrices. Supports DINOv2, DINOv3,
CLIP, and MAE architectures with correct target module identification.
"""

import logging
from typing import List, Optional

import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

# Known target module patterns for different model families
TARGET_MODULE_PATTERNS = {
    "dinov2": {
        "qkv": ["attn.qkv"],           # Fused QKV in DINOv2 torch.hub
        "q_v": ["attn.proj"],           # Alternative
        "hf": ["query", "value"],       # HuggingFace DINOv2
    },
    "dinov3": {
        "qkv": ["attn.qkv"],
        "hf": ["query", "value"],
    },
    "clip": {
        "q_v": ["q_proj", "v_proj"],    # CLIP uses separate Q, V projections
    },
    "mae": {
        "q_v": ["query", "value"],      # HuggingFace MAE
    },
}


def _find_target_modules(
    model: nn.Module,
    target_modules: List[str],
    model_family: str = "dinov2",
) -> List[str]:
    """Find the correct target module names by inspecting the model.

    Args:
        model: The model to inspect.
        target_modules: Candidate target module names from config.
        model_family: Model family name for fallback patterns.

    Returns:
        List of valid target module name patterns.
    """
    # Collect all named modules
    all_module_names = [name for name, _ in model.named_modules()]

    # Check if any of the specified target modules exist
    valid_targets = []
    for target in target_modules:
        matches = [name for name in all_module_names if target in name]
        if matches:
            valid_targets.append(target)
            logger.info(
                "Found target module pattern '%s': %d matches", target, len(matches)
            )

    if valid_targets:
        return valid_targets

    # Fallback: try known patterns for this family
    logger.warning(
        "Specified target modules %s not found. Trying known patterns for %s.",
        target_modules,
        model_family,
    )

    patterns = TARGET_MODULE_PATTERNS.get(model_family, {})
    for pattern_name, candidates in patterns.items():
        for candidate in candidates:
            matches = [name for name in all_module_names if candidate in name]
            if matches:
                logger.info(
                    "Using fallback pattern '%s' (%s): %d matches",
                    candidate,
                    pattern_name,
                    len(matches),
                )
                return [candidate]

    # Last resort: find any Linear layers in attention blocks
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            kw in name.lower() for kw in ["attn", "attention", "query", "key", "value"]
        ):
            # Extract the module attribute name (last part)
            parts = name.split(".")
            if parts:
                linear_modules.append(parts[-1])

    if linear_modules:
        unique_names = list(set(linear_modules))
        logger.info("Auto-detected attention Linear modules: %s", unique_names)
        return unique_names

    raise ValueError(
        f"Could not find valid LoRA target modules for {model_family}. "
        f"Available modules: {all_module_names[:20]}... "
        f"Please inspect model architecture and update config."
    )


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 4,
    alpha: int = 8,
    dropout: float = 0.1,
    model_family: str = "dinov2",
    task_type: Optional[str] = None,
) -> PeftModel:
    """Inject LoRA adapters into a model.

    Args:
        model: The base model to adapt.
        target_modules: List of module name patterns to target.
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling factor (typically 2*r).
        dropout: Dropout probability for LoRA layers.
        model_family: Model family for target module resolution.
        task_type: PEFT task type (auto-detected if None).

    Returns:
        PeftModel with LoRA adapters injected.
    """
    # Find valid target modules
    valid_targets = _find_target_modules(model, target_modules, model_family)

    logger.info(
        "Injecting LoRA: rank=%d, alpha=%d, dropout=%.2f, targets=%s",
        rank,
        alpha,
        dropout,
        valid_targets,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=valid_targets,
        bias="none",
        modules_to_save=None,
    )

    peft_model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA injected: trainable=%s / total=%s (%.4f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / total_params,
    )

    # Print summary
    peft_model.print_trainable_parameters()

    return peft_model


def merge_lora(model: PeftModel) -> nn.Module:
    """Merge LoRA weights back into the base model for efficient inference.

    After merging, the model behaves as a standard model without LoRA
    overhead, but with the adapted weights.

    Args:
        model: PeftModel with trained LoRA adapters.

    Returns:
        Base model with merged weights.
    """
    logger.info("Merging LoRA weights into base model...")

    if isinstance(model, PeftModel):
        merged_model = model.merge_and_unload()
        logger.info("LoRA weights merged successfully")
        return merged_model
    else:
        logger.warning("Model is not a PeftModel, returning as-is")
        return model


def get_lora_state_dict(model: PeftModel) -> dict:
    """Extract only the LoRA adapter weights.

    Useful for saving just the small adapter weights instead of
    the full model checkpoint.

    Args:
        model: PeftModel with LoRA adapters.

    Returns:
        State dict containing only LoRA parameters.
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.clone()

    logger.info(
        "Extracted LoRA state dict: %d parameters, %.2f MB",
        len(lora_state),
        sum(p.numel() * 4 for p in lora_state.values()) / 1024 / 1024,
    )
    return lora_state
