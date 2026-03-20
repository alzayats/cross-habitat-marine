"""DeepFish dataset loader with habitat-based splits for cross-habitat transfer.

DeepFish contains images from 20 distinct underwater habitats in tropical Australia
(Great Barrier Reef). Each habitat directory contains 'empty' (no fish) and 'valid'
(fish present) subdirectories for binary classification.

Habitat splits are done by physical location (directory name), NOT randomly,
to test genuine cross-habitat transfer.

Reference: Saleh et al., "A Realistic Fish-Habitat Dataset to Evaluate Algorithms
for Underwater Visual Analysis", Scientific Reports, 2020.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from data.base_dataset import BaseMarineDataset

logger = logging.getLogger(__name__)


class DeepFishDataset(BaseMarineDataset):
    """DeepFish dataset with habitat-aware splits.

    Supports binary classification (fish/no-fish) across 20 habitats.
    Habitats are identified by numeric IDs corresponding to physical
    reef locations in the Great Barrier Reef.

    Args:
        root_dir: Path to DeepFish root (containing Classification/).
        split: Data split ('train', 'val', 'test'). Used for standard splits.
        transform: Image transform pipeline.
        mode: 'classification' or 'detection'.
        image_size: Target image size.
        habitats: List of habitat IDs to include. If None, include all.
        task: 'binary' (fish/no-fish).
    """

    DATASET_NAME = "DeepFish"

    # All 20 habitat IDs (discovered from real directory structure)
    HABITAT_NAMES: List[str] = []  # Populated dynamically

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
        habitats: Optional[List[str]] = None,
        task: str = "binary",
    ) -> None:
        self.task = task
        self._classification_dir = Path(root_dir) / "Classification"

        # Discover all habitat directories
        all_habitats = self._discover_habitats()
        if not DeepFishDataset.HABITAT_NAMES:
            DeepFishDataset.HABITAT_NAMES = all_habitats

        self.habitats = habitats if habitats is not None else all_habitats
        self.habitat_descriptions = self._load_habitat_descriptions(Path(root_dir))

        super().__init__(root_dir, split, transform, mode, image_size)

    def _discover_habitats(self) -> List[str]:
        """Discover all habitat directories from the Classification folder."""
        if not self._classification_dir.exists():
            return []
        habitats = sorted([
            d.name for d in self._classification_dir.iterdir()
            if d.is_dir()
        ])
        return habitats

    def _load_habitat_descriptions(self, root: Path) -> Dict[str, str]:
        """Load habitat descriptions from the summary CSV if available."""
        descriptions: Dict[str, str] = {}
        csv_path = root / "SummeryOFdatset_190915.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            descriptions[row[0].strip()] = row[1].strip()
            except Exception as e:
                logger.debug("Could not parse habitat CSV: %s", e)
        return descriptions

    def _load_annotations(self) -> None:
        """Load DeepFish annotations from directory structure.

        Each habitat directory contains 'empty' and 'valid' subdirs.
        'empty' = no fish (label: 'empty'), 'valid' = fish present (label: 'valid').
        """
        if not self._classification_dir.exists():
            raise FileNotFoundError(
                f"Classification directory not found: {self._classification_dir}. "
                f"Expected: {self.root_dir}/Classification/"
            )

        for habitat_id in self.habitats:
            habitat_dir = self._classification_dir / habitat_id

            if not habitat_dir.exists():
                logger.warning(
                    "Habitat directory not found: %s. Skipping.", habitat_dir
                )
                continue

            self._load_habitat_images(habitat_dir, habitat_id)

        if not self.samples:
            available = [d.name for d in self._classification_dir.iterdir() if d.is_dir()]
            logger.warning(
                "No samples loaded. Requested habitats: %s. Available: %s",
                self.habitats, available,
            )

    def _load_habitat_images(self, habitat_dir: Path, habitat_id: str) -> None:
        """Load images from a single habitat directory.

        Args:
            habitat_dir: Path to the habitat directory (e.g., Classification/7482/).
            habitat_id: Numeric habitat identifier.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # DeepFish uses 'empty' and 'valid' subdirectories
        for label_dir_name in ["empty", "valid"]:
            label_dir = habitat_dir / label_dir_name
            if not label_dir.exists():
                continue

            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() in image_extensions:
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "label": label_dir_name,
                            "habitat": habitat_id,
                        }
                    )

    @staticmethod
    def get_all_habitats(root_dir: str) -> List[str]:
        """Get all habitat IDs from the dataset directory.

        Args:
            root_dir: Path to DeepFish root.

        Returns:
            Sorted list of habitat ID strings.
        """
        cls_dir = Path(root_dir) / "Classification"
        if not cls_dir.exists():
            return []
        return sorted([d.name for d in cls_dir.iterdir() if d.is_dir()])

    @staticmethod
    def get_leave_one_out_splits(
        habitats: Optional[List[str]] = None,
        root_dir: Optional[str] = None,
    ) -> List[Tuple[List[str], str]]:
        """Get leave-one-habitat-out splits (train on N-1, test on 1).

        Args:
            habitats: List of habitat IDs. If None, discovers from root_dir.
            root_dir: Path to DeepFish root (used if habitats is None).

        Returns:
            List of (source_habitats, target_habitat) tuples.
        """
        if habitats is None and root_dir is not None:
            habitats = DeepFishDataset.get_all_habitats(root_dir)
        elif habitats is None:
            raise ValueError("Must provide either habitats or root_dir")

        splits: List[Tuple[List[str], str]] = []
        for target in habitats:
            sources = [h for h in habitats if h != target]
            splits.append((sources, target))
        return splits

    @staticmethod
    def get_multi_source_splits(
        habitats: Optional[List[str]] = None,
        root_dir: Optional[str] = None,
        n_train: int = 15,
        n_test: int = 5,
        n_configs: int = 5,
        seed: int = 42,
    ) -> List[Tuple[List[str], List[str]]]:
        """Get random multi-source splits (train on n_train, test on n_test).

        Args:
            habitats: List of habitat IDs.
            root_dir: Path to DeepFish root.
            n_train: Number of training habitats.
            n_test: Number of test habitats.
            n_configs: Number of random configurations.
            seed: Random seed.

        Returns:
            List of (train_habitats, test_habitats) tuples.
        """
        if habitats is None and root_dir is not None:
            habitats = DeepFishDataset.get_all_habitats(root_dir)
        elif habitats is None:
            raise ValueError("Must provide either habitats or root_dir")

        rng = np.random.RandomState(seed)
        splits = []
        for _ in range(n_configs):
            perm = rng.permutation(len(habitats))
            train_habs = [habitats[i] for i in perm[:n_train]]
            test_habs = [habitats[i] for i in perm[n_train:n_train + n_test]]
            splits.append((train_habs, test_habs))
        return splits

    @classmethod
    def get_habitat_split(
        cls,
        root_dir: str,
        source_habitats: List[str],
        target_habitat: str,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
        task: str = "binary",
        val_split: float = 0.1,
        seed: int = 42,
    ) -> Tuple["DeepFishDataset", Any, "DeepFishDataset"]:
        """Create train/val/test datasets with habitat-based splits.

        Args:
            root_dir: Path to DeepFish root.
            source_habitats: List of habitat IDs for training.
            target_habitat: Habitat ID for testing.
            train_transform: Transform for training images.
            test_transform: Transform for test images.
            mode: 'classification' or 'detection'.
            image_size: Target image size.
            task: 'binary'.
            val_split: Fraction of source data for validation.
            seed: Random seed for val split.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        # Load source data
        source_dataset = cls(
            root_dir=root_dir,
            split="train",
            transform=train_transform,
            mode=mode,
            image_size=image_size,
            habitats=source_habitats,
            task=task,
        )

        # Split source into train/val
        rng = np.random.RandomState(seed)
        n_samples = len(source_dataset)
        indices = rng.permutation(n_samples)
        n_val = int(n_samples * val_split)

        val_indices = indices[:n_val].tolist()
        train_indices = indices[n_val:].tolist()

        train_dataset = source_dataset.get_subset(train_indices)
        val_dataset = source_dataset.get_subset(val_indices)

        # Load target data
        test_dataset = cls(
            root_dir=root_dir,
            split="test",
            transform=test_transform,
            mode=mode,
            image_size=image_size,
            habitats=[target_habitat],
            task=task,
        )

        logger.info(
            "Habitat split: source=%s, target=%s | train=%d, val=%d, test=%d",
            source_habitats,
            target_habitat,
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )

        return train_dataset, val_dataset, test_dataset

    @classmethod
    def create_dataloaders(
        cls,
        root_dir: str,
        source_habitats: List[str],
        target_habitat: str,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for a habitat split.

        Args:
            root_dir: Path to DeepFish root.
            source_habitats: Source habitat IDs.
            target_habitat: Target habitat ID.
            train_transform: Training transform.
            test_transform: Test transform.
            batch_size: Batch size.
            num_workers: Number of data workers.
            pin_memory: Pin memory for GPU transfer.
            **kwargs: Additional arguments passed to get_habitat_split.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_ds, val_ds, test_ds = cls.get_habitat_split(
            root_dir=root_dir,
            source_habitats=source_habitats,
            target_habitat=target_habitat,
            train_transform=train_transform,
            test_transform=test_transform,
            **kwargs,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader
