"""AQUA20 dataset loader for marine species recognition.

AQUA20 is a benchmark dataset for underwater image classification
with 20 aquatic species classes. Uses ImageFolder structure under
train_images/ and test_images/ directories.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from data.base_dataset import BaseMarineDataset

logger = logging.getLogger(__name__)


class AQUA20Dataset(BaseMarineDataset):
    """AQUA20 dataset loader using ImageFolder-style directory structure.

    Real structure:
        AQUA20_dataset/
        ├── train_images/
        │   ├── coral/
        │   ├── crab/
        │   ├── ...
        │   └── turtle/
        ├── test_images/
        │   ├── coral/
        │   ├── crab/
        │   ├── ...
        │   └── turtle/
        └── labels.txt

    Note: The train directory has a typo 'seaCucumbe' which should be
    'seaCucumber'. This loader handles the inconsistency.

    Args:
        root_dir: Path to AQUA20 root directory.
        split: 'train' or 'test'.
        transform: Image transform pipeline.
        mode: 'classification' (only mode supported).
        image_size: Target image size.
    """

    DATASET_NAME = "AQUA20"
    HABITAT_NAMES = ["aquatic"]

    SPECIES_NAMES = [
        "coral",
        "crab",
        "diver",
        "eel",
        "fish",
        "fishInGroups",
        "flatworm",
        "jellyfish",
        "marine_dolphin",
        "octopus",
        "rayfish",
        "seaAnemone",
        "seaCucumber",
        "seaSlug",
        "seaUrchin",
        "shark",
        "shrimp",
        "squid",
        "starfish",
        "turtle",
    ]

    # Mapping from directory name typos to canonical names
    _NAME_FIXES = {
        "seaCucumbe": "seaCucumber",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
    ) -> None:
        if mode != "classification":
            logger.warning(
                "AQUA20 only supports classification mode. Ignoring mode=%s.", mode
            )
            mode = "classification"
        super().__init__(root_dir, split, transform, mode, image_size)

    def _load_annotations(self) -> None:
        """Load AQUA20 annotations from ImageFolder structure."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # Map split names to actual directory names
        split_dir_map = {
            "train": "train_images",
            "test": "test_images",
            "val": "test_images",  # No separate val; use test
        }

        dir_name = split_dir_map.get(self.split, f"{self.split}_images")
        split_dir = self.root_dir / dir_name

        if not split_dir.exists():
            # Fallback: try split name directly
            split_dir = self.root_dir / self.split
            if not split_dir.exists():
                logger.warning(
                    "Split directory not found: %s or %s",
                    self.root_dir / dir_name,
                    split_dir,
                )
                return

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            # Fix known typos in directory names
            class_name = class_dir.name
            class_name = self._NAME_FIXES.get(class_name, class_name)

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in image_extensions:
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "label": class_name,
                            "habitat": "aquatic",
                        }
                    )

        if not self.samples:
            logger.warning(
                "No samples found in %s. Check directory structure.", split_dir
            )
