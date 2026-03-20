"""Main training loop with mixed precision, logging, and checkpointing.

Supports classification and detection tasks with configurable
optimizer, scheduler, and early stopping.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.early_stopping import EarlyStopping
from training.evaluator import Evaluator
from training.losses import get_loss_function
from training.optimizers import create_optimizer, create_scheduler
from utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop manager with full experiment lifecycle.

    Handles training, validation, checkpointing, early stopping,
    mixed precision, gradient clipping, and logging (WandB + CSV).

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration dict or DictConfig.
        adaptation_config: Adaptation strategy configuration.
        output_dir: Directory for checkpoints and logs.
        experiment_name: Name for this experiment run.
        device: Training device.
        wandb_run: Optional WandB run object.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        adaptation_config: Any,
        output_dir: str = "./outputs",
        experiment_name: str = "experiment",
        device: str = "cuda",
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.adaptation_config = adaptation_config
        self.experiment_name = experiment_name
        self.wandb_run = wandb_run

        # Output directory
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Multi-GPU
        multi_gpu = getattr(config, "multi_gpu", False) if hasattr(config, "multi_gpu") else config.get("multi_gpu", False)
        if multi_gpu and torch.cuda.device_count() > 1:
            logger.info("Using DataParallel with %d GPUs", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # Training components
        adapt = adaptation_config
        lr = getattr(adapt, "learning_rate", 1e-4)
        wd = getattr(adapt, "weight_decay", 0.05)
        opt_type = getattr(adapt, "optimizer", "adamw")
        layer_decay = getattr(adapt, "layer_decay", None)
        self.epochs = getattr(adapt, "epochs", 30)
        # Allow base config to cap epochs (useful for smoke tests)
        max_epochs_override = getattr(config, "max_epochs", None)
        if max_epochs_override is not None:
            self.epochs = min(self.epochs, max_epochs_override)
        scheduler_type = getattr(adapt, "scheduler", "cosine")
        warmup_frac = getattr(adapt, "warmup_fraction", 0.05)
        patience = getattr(adapt, "early_stopping_patience", 10)
        label_smoothing = getattr(adapt, "label_smoothing", 0.0)

        self.optimizer = create_optimizer(
            self.model, opt_type, lr, wd, layer_decay
        )

        steps_per_epoch = len(train_loader)
        self.scheduler = create_scheduler(
            self.optimizer, scheduler_type, self.epochs, warmup_frac, steps_per_epoch
        )

        self.criterion = get_loss_function(
            loss_type="cross_entropy", label_smoothing=label_smoothing
        )

        self.early_stopping = EarlyStopping(patience=patience, mode="min")

        # Mixed precision
        mixed_precision = getattr(config, "mixed_precision", True) if hasattr(config, "mixed_precision") else config.get("mixed_precision", True)
        self.use_amp = mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip = getattr(config, "gradient_clip_max_norm", 1.0) if hasattr(config, "gradient_clip_max_norm") else config.get("gradient_clip_max_norm", 1.0)

        # Gradient accumulation
        self.grad_accum_steps = getattr(config, "gradient_accumulation_steps", 1) if hasattr(config, "gradient_accumulation_steps") else config.get("gradient_accumulation_steps", 1)

        # Training history
        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.start_epoch = 0

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            resume_from: Optional checkpoint path to resume from.

        Returns:
            Training results dictionary.
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(
            "Starting training: epochs=%d, device=%s, amp=%s",
            self.epochs,
            self.device,
            self.use_amp,
        )

        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            # Training epoch
            train_metrics = self._train_epoch(epoch)

            # Validation epoch
            val_metrics = self._validate_epoch(epoch)

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["macro_f1"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_metrics)

            # Logging
            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, "
                "val_acc=%.4f, val_f1=%.4f, lr=%.2e",
                epoch + 1,
                self.epochs,
                epoch_metrics["train_loss"],
                epoch_metrics["val_loss"],
                epoch_metrics["val_accuracy"],
                epoch_metrics["val_f1"],
                epoch_metrics["learning_rate"],
            )

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_metrics)

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)

            # Early stopping
            if self.early_stopping(val_metrics["loss"], epoch):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        total_time = time.time() - start_time

        # Save final checkpoint
        self._save_checkpoint(epoch, is_best=False)

        # Save training history
        self._save_history()

        results = {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch + 1,
            "total_time_seconds": total_time,
            "history": self.history,
        }

        logger.info(
            "Training complete: best_epoch=%d, best_val_loss=%.4f, time=%.1fs",
            self.best_epoch,
            self.best_val_loss,
            total_time,
        )

        return results

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Training metrics for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}/{self.epochs}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics.

        Args:
            epoch: Current epoch number.

        Returns:
            Validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        for batch in tqdm(
            self.val_loader,
            desc=f"Val Epoch {epoch + 1}",
            leave=False,
        ):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        from sklearn.metrics import accuracy_score, f1_score

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        return {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a model checkpoint.

        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
        """
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / "last_model.pt"

        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved: %s", path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint.get("history", [])

        logger.info(
            "Resumed from epoch %d (best_val_loss=%.4f)",
            self.start_epoch,
            self.best_val_loss,
        )

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", history_path)

    def load_best_model(self) -> None:
        """Load the best checkpoint back into the model."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            self._load_checkpoint(str(best_path))
            logger.info("Best model loaded from epoch %d", self.best_epoch)
        else:
            logger.warning("No best model checkpoint found at %s", best_path)
