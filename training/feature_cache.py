"""Feature extraction and caching for linear probe acceleration.

Extracts backbone features once per model×dataset pair and caches them
to disk. Subsequent linear probe training uses cached features directly,
avoiding repeated forward passes through the frozen backbone.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _cache_key(model_name: str, dataset_name: str, split: str, image_size: int) -> str:
    """Generate a deterministic cache filename."""
    raw = f"{model_name}_{dataset_name}_{split}_{image_size}"
    return raw


@torch.no_grad()
def extract_and_cache_features(
    model: nn.Module,
    dataset: Any,
    model_name: str,
    dataset_name: str,
    split: str,
    image_size: int,
    cache_dir: str = "./outputs/features",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from a frozen backbone and cache to disk.

    Args:
        model: Model with get_features() method (backbone only, no head).
        dataset: Dataset to extract features from.
        model_name: Model config name (for cache key).
        dataset_name: Dataset name (for cache key).
        split: Dataset split (for cache key).
        image_size: Image size used (for cache key).
        cache_dir: Directory to store cached features.
        device: Device for extraction.
        batch_size: Batch size for extraction.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (features [N, D], labels [N]) as tensors.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    key = _cache_key(model_name, dataset_name, split, image_size)
    feat_file = cache_path / f"{key}_features.pt"
    label_file = cache_path / f"{key}_labels.pt"

    # Return cached if exists
    if feat_file.exists() and label_file.exists():
        logger.info("Loading cached features: %s", key)
        features = torch.load(feat_file, map_location="cpu").float()
        labels = torch.load(label_file, map_location="cpu")
        logger.info("Cached features loaded: %s, shape=%s", key, features.shape)
        return features, labels

    # Extract features
    logger.info("Extracting features: %s (%d samples)", key, len(dataset))
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_features = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        with autocast(enabled=torch.cuda.is_available()):
            feats = model.get_features(images)

        all_features.append(feats.cpu())
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0).float()
    labels = torch.cat(all_labels, dim=0)

    # Save to disk
    torch.save(features, feat_file)
    torch.save(labels, label_file)
    logger.info("Features cached: %s, shape=%s", key, features.shape)

    return features, labels


class CachedFeatureDataset(torch.utils.data.Dataset):
    """Dataset backed by pre-extracted features (no images).

    Args:
        features: Feature tensor [N, D].
        labels: Label tensor [N].
        indices: Optional subset indices.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            self.features = features[indices]
            self.labels = labels[indices]
        else:
            self.features = features
            self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "feature": self.features[idx],
            "label": self.labels[idx],
        }

    def get_labels(self) -> np.ndarray:
        return self.labels.numpy()


class LinearProbeTrainer:
    """Fast linear probe training on cached features.

    Trains a simple linear head on pre-extracted features.
    Much faster than full Trainer since no backbone forward passes.

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of output classes.
        lr: Learning rate.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        device: Training device.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 10,
        device: str = "cuda",
        weight_decay: float = 0.0,
    ):
        self.device = device
        self.epochs = epochs
        self.patience = patience

        self.head = nn.Linear(feature_dim, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=lr, weight_decay=weight_decay,
        )

    def train(
        self,
        train_ds: CachedFeatureDataset,
        val_ds: CachedFeatureDataset,
    ) -> Dict[str, Any]:
        """Train the linear head on cached features.

        Returns:
            Dict with best_epoch, best_val_loss, history.
        """
        batch_size = min(256, len(train_ds))
        if batch_size < 1:
            batch_size = 1

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(512, len(val_ds)),
            shuffle=False, num_workers=0, pin_memory=False,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self.head.train()
            for batch in train_loader:
                feats = batch["feature"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.head(feats)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validate
            self.head.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch["feature"].to(self.device)
                    labels = batch["label"].to(self.device)
                    logits = self.head(feats)
                    val_loss += self.criterion(logits, labels).item()
                    n += 1
            val_loss /= max(n, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.head.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best
        if best_state is not None:
            self.head.load_state_dict(best_state)

        return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}

    @torch.no_grad()
    def evaluate(
        self,
        test_ds: CachedFeatureDataset,
        num_eval_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate the trained linear head.

        Args:
            test_ds: Test dataset of cached features.
            num_eval_classes: If set, restrict logits to the first N classes
                before argmax (used when head has source+target classes but
                evaluation is target-only).

        Returns:
            Dict with top1_accuracy, macro_f1, balanced_accuracy.
        """
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

        loader = DataLoader(
            test_ds, batch_size=512, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        self.head.eval()
        all_preds = []
        all_labels = []

        for batch in loader:
            feats = batch["feature"].to(self.device)
            labels = batch["label"]
            logits = self.head(feats)
            if num_eval_classes is not None:
                logits = logits[:, :num_eval_classes]
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        return {
            "top1_accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
