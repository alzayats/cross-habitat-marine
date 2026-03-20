"""Base dataset class for marine species recognition datasets."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Allow loading of truncated images (common in large marine datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class BaseMarineDataset(Dataset, ABC):
    """Abstract base class for marine species recognition datasets.

    Provides common functionality for loading underwater images with
    classification and optional detection annotations. Subclasses must
    implement dataset-specific loading logic.

    Args:
        root_dir: Path to the dataset root directory.
        split: Data split ('train', 'val', 'test').
        transform: Optional image transform pipeline.
        mode: Task mode ('classification' or 'detection').
        image_size: Target image size for resizing.
    """

    # Subclasses should define these
    DATASET_NAME: str = "base"
    SPECIES_NAMES: List[str] = []
    HABITAT_NAMES: List[str] = []

    # Taxonomic family mapping for cross-dataset transfer
    SPECIES_TO_FAMILY: Dict[str, str] = {}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.mode = mode
        self.image_size = image_size

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root_dir}. "
                f"Run data/download_datasets.sh first."
            )

        self.samples: List[Dict[str, Any]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_annotations()
        self._build_class_mapping()

        logger.info(
            "%s dataset loaded: split=%s, mode=%s, samples=%d, classes=%d",
            self.DATASET_NAME,
            split,
            mode,
            len(self.samples),
            len(self.class_to_idx),
        )

    @abstractmethod
    def _load_annotations(self) -> None:
        """Load dataset annotations and populate self.samples.

        Each sample should be a dict with at minimum:
            - 'image_path': str or Path to the image file
            - 'label': str class label
            - 'habitat': str habitat/location identifier (if available)
        For detection mode, also include:
            - 'boxes': List of [x1, y1, x2, y2] bounding boxes
            - 'box_labels': List of str class labels for each box
        """
        raise NotImplementedError

    def _build_class_mapping(self) -> None:
        """Build class-to-index and index-to-class mappings."""
        if self.mode == "classification":
            unique_labels = sorted(set(s["label"] for s in self.samples))
        else:
            unique_labels = sorted(
                set(
                    lbl
                    for s in self.samples
                    for lbl in s.get("box_labels", [s.get("label", "unknown")])
                )
            )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_labels)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_to_idx)

    @property
    def class_names(self) -> List[str]:
        """Return sorted list of class names."""
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

    def get_class_distribution(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        dist: Dict[str, int] = {}
        for sample in self.samples:
            label = sample["label"]
            dist[label] = dist.get(label, 0) + 1
        return dict(sorted(dist.items()))

    def get_habitat_ids(self) -> List[str]:
        """Return list of habitat identifiers for all samples."""
        return [s.get("habitat", "unknown") for s in self.samples]

    def get_samples_by_habitat(self, habitat: str) -> List[int]:
        """Return indices of samples from a specific habitat."""
        return [
            i for i, s in enumerate(self.samples) if s.get("habitat") == habitat
        ]

    def get_labels(self) -> np.ndarray:
        """Return array of integer labels for all samples."""
        return np.array([self.class_to_idx[s["label"]] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset sample.

        Returns:
            Dictionary with keys:
                - 'image': Transformed image tensor
                - 'label': Integer class label
                - 'label_name': String class name
                - 'habitat': Habitat identifier
                - 'image_path': Path to source image
                - 'boxes' (detection mode): Bounding boxes tensor
                - 'box_labels' (detection mode): Box class labels tensor
        """
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        result = {
            "image": image,
            "label": self.class_to_idx[sample["label"]],
            "label_name": sample["label"],
            "habitat": sample.get("habitat", "unknown"),
            "image_path": str(sample["image_path"]),
        }

        if self.mode == "detection" and "boxes" in sample:
            boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
            box_labels = torch.tensor(
                [self.class_to_idx[lbl] for lbl in sample["box_labels"]],
                dtype=torch.long,
            )
            result["boxes"] = boxes
            result["box_labels"] = box_labels

        return result

    def get_subset(self, indices: List[int]) -> "SubsetMarineDataset":
        """Create a subset dataset from given indices."""
        return SubsetMarineDataset(self, indices)

    def get_family_labels(self) -> Optional[np.ndarray]:
        """Return family-level labels for taxonomic grouping."""
        if not self.SPECIES_TO_FAMILY:
            return None
        families = sorted(set(self.SPECIES_TO_FAMILY.values()))
        family_to_idx = {f: i for i, f in enumerate(families)}
        labels = []
        for s in self.samples:
            family = self.SPECIES_TO_FAMILY.get(s["label"], "unknown")
            labels.append(family_to_idx.get(family, -1))
        return np.array(labels)


class SubsetMarineDataset(Dataset):
    """A subset of a BaseMarineDataset, preserving metadata access.

    Args:
        dataset: The parent dataset.
        indices: List of indices to include.
    """

    def __init__(self, dataset: BaseMarineDataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def class_names(self) -> List[str]:
        return self.dataset.class_names

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self.dataset.class_to_idx

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return self.dataset.idx_to_class

    def get_labels(self) -> np.ndarray:
        all_labels = self.dataset.get_labels()
        return all_labels[self.indices]

    def get_subset(self, indices: List[int]) -> "SubsetMarineDataset":
        """Create a further subset from this subset."""
        new_indices = [self.indices[i] for i in indices]
        return SubsetMarineDataset(self.dataset, new_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.indices[idx]]
