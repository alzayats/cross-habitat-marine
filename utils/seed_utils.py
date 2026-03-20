"""Reproducibility utilities: seed everything for deterministic experiments."""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across all libraries.

    Sets seeds for: torch, CUDA, numpy, random, and Python hash.
    Also configures CuDNN for deterministic behavior.

    Args:
        seed: Random seed value.
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Random seed set to %d (deterministic mode)", seed)


def get_random_state() -> dict:
    """Capture current random state for checkpointing.

    Returns:
        Dictionary of random states for all libraries.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state: dict) -> None:
    """Restore random state from checkpoint.

    Args:
        state: Random state dictionary from get_random_state().
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

    logger.info("Random state restored")
