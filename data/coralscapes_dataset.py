"""Coralscapes dataset loader for benthic classification.

Dense pixel-level semantic segmentation dataset from the Red Sea (5 countries)
with 39 benthic classes. For classification experiments, each image is assigned
a label based on the dominant class in its segmentation mask.

Structure:
    Coralscapes_dataset/
    ├── train/
    │   ├── images/    (1517 JPGs)
    │   └── labels/    (1517 PNGs, 8-bit grayscale masks)
    ├── validation/
    │   ├── images/    (166 JPGs)
    │   └── labels/    (166 PNGs)
    └── test/
        ├── images/    (392 JPGs)
        └── labels/    (392 PNGs)

Reference: Coralscapes (EPFL-ECEO), benchmarks DINOv2+LoRA.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from data.base_dataset import BaseMarineDataset

logger = logging.getLogger(__name__)

# Coralscapes class names (39 benthic classes, index 0-38)
# Based on the EPFL-ECEO/coralscapes HuggingFace dataset
CORALSCAPES_CLASSES = [
    "Background",
    "Hard_Coral_Branching",
    "Hard_Coral_Encrusting",
    "Hard_Coral_Foliose",
    "Hard_Coral_Massive",
    "Hard_Coral_Mushroom",
    "Hard_Coral_Submassive",
    "Hard_Coral_Table",
    "Soft_Coral",
    "Soft_Coral_Leather",
    "Soft_Coral_Xenia",
    "Sponge",
    "Sponge_Barrel",
    "Algae_Encrusting",
    "Algae_Fleshy",
    "Algae_Halimeda",
    "Algae_Turf",
    "Seagrass",
    "Millepora",
    "Clam",
    "Anemone",
    "Zoanthid",
    "Crinoid",
    "Sea_Urchin",
    "Sea_Cucumber",
    "Crown_of_Thorns",
    "Fish",
    "Rubble",
    "Rock",
    "Sand",
    "Silt",
    "Pavement",
    "Trash",
    "Unknown_Hard_Coral",
    "Unknown_Soft_Coral",
    "Unknown_Organism",
    "Shadow",
    "Water",
    "Human_Equipment",
]


class CoralscapesDataset(BaseMarineDataset):
    """Coralscapes semantic segmentation dataset adapted for classification.

    For classification, assigns each image a label based on the dominant
    class in its segmentation mask (excluding Background/Shadow/Water/Unknown).

    Args:
        root_dir: Path to Coralscapes_dataset root.
        split: 'train', 'val'/'validation', or 'test'.
        transform: Image transform pipeline.
        mode: 'classification'.
        image_size: Target image size.
        exclude_classes: Class names to exclude from dominant class selection.
    """

    DATASET_NAME = "Coralscapes"
    SPECIES_NAMES = CORALSCAPES_CLASSES
    HABITAT_NAMES = ["red_sea"]

    # Classes to exclude when computing dominant label
    _DEFAULT_EXCLUDE = {
        "Background", "Shadow", "Water", "Human_Equipment",
        "Unknown_Hard_Coral", "Unknown_Soft_Coral", "Unknown_Organism",
        "Trash",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
        exclude_classes: Optional[List[str]] = None,
    ) -> None:
        if mode != "classification":
            logger.warning(
                "Coralscapes classification mode used. Ignoring mode=%s.", mode
            )
            mode = "classification"
        self.exclude_classes = set(exclude_classes) if exclude_classes else self._DEFAULT_EXCLUDE
        self._exclude_indices = {
            i for i, name in enumerate(CORALSCAPES_CLASSES)
            if name in self.exclude_classes
        }
        super().__init__(root_dir, split, transform, mode, image_size)

    def _get_dominant_class_from_mask(self, mask_path: Path) -> Optional[str]:
        """Get the most frequent non-excluded class from a segmentation mask.

        Args:
            mask_path: Path to the label PNG file.

        Returns:
            Class name of the dominant class, or None if no valid class.
        """
        mask = np.array(Image.open(mask_path))
        unique, counts = np.unique(mask, return_counts=True)

        best_class = None
        best_count = 0

        for idx, count in zip(unique, counts):
            if idx >= len(CORALSCAPES_CLASSES):
                continue
            if idx in self._exclude_indices:
                continue
            if count > best_count:
                best_count = count
                best_class = CORALSCAPES_CLASSES[idx]

        return best_class

    def _load_annotations(self) -> None:
        """Load Coralscapes images and compute dominant class labels."""
        # Map split names to directory names
        split_map = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test",
        }

        split_dir_name = split_map.get(self.split, self.split)
        split_dir = self.root_dir / split_dir_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            logger.warning("Images directory not found: %s", images_dir)
            return

        if not labels_dir.exists():
            logger.warning("Labels directory not found: %s", labels_dir)
            return

        for img_path in sorted(images_dir.glob("*.jpg")):
            # Find corresponding label mask
            stem = img_path.stem
            mask_path = labels_dir / f"{stem}.png"

            if not mask_path.exists():
                continue

            dominant_class = self._get_dominant_class_from_mask(mask_path)
            if dominant_class is None:
                continue

            self.samples.append(
                {
                    "image_path": img_path,
                    "label": dominant_class,
                    "habitat": "red_sea",
                    "mask_path": str(mask_path),
                }
            )

        if not self.samples:
            logger.warning(
                "No samples loaded for Coralscapes split=%s. Check: %s",
                self.split,
                split_dir,
            )
