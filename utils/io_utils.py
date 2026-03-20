"""I/O utilities for saving/loading checkpoints and results."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_results(
    results: Any,
    output_path: str,
) -> None:
    """Save experiment results to JSON.

    Args:
        results: Results data (dict or list).
        output_path: Output file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info("Results saved: %s", path)


def load_results(input_path: str) -> Any:
    """Load experiment results from JSON.

    Args:
        input_path: Path to JSON file.

    Returns:
        Loaded data.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    logger.info("Results loaded: %s", input_path)
    return data


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optional optimizer state.
        epoch: Current epoch.
        output_path: Checkpoint file path.
        metadata: Optional metadata dict.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.
        strict: Whether to strictly enforce state dict matching.

    Returns:
        Checkpoint metadata dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metadata = checkpoint.get("metadata", {})

    logger.info("Checkpoint loaded: %s (epoch %d)", checkpoint_path, epoch)

    return {"epoch": epoch, "metadata": metadata}


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
