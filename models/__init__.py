"""Model architectures and adaptation strategies for cross-habitat marine recognition."""

from models.model_factory import create_model, ModelFactory
from models.foundation_wrapper import FoundationModelWrapper
from models.lora_adapter import inject_lora, merge_lora
from models.vpt_adapter import VPTDeep
from models.classification_head import ClassificationHead
from models.baseline_models import create_baseline_model

__all__ = [
    "create_model",
    "ModelFactory",
    "FoundationModelWrapper",
    "inject_lora",
    "merge_lora",
    "VPTDeep",
    "ClassificationHead",
    "create_baseline_model",
]
