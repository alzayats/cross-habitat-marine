"""Early stopping with patience for training loops."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping monitor based on validation loss.

    Tracks validation loss and signals when to stop training if no
    improvement is seen for a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' (for loss) or 'max' (for accuracy/F1).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value (e.g., validation loss).
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = False
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at epoch %d. "
                    "Best %s=%.6f at epoch %d.",
                    epoch,
                    "score" if self.mode == "max" else "loss",
                    self.best_score,
                    self.best_epoch,
                )
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
