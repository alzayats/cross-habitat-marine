"""Moorea Labeled Corals dataset loader.

Point-annotation dataset from the MCR LTER site in Moorea, French Polynesia.
Each image has a companion .jpg.txt file with point annotations in the format:
    # Row; Col; Label
    671; 217; Sand
    ...

Contains ~2,050 images across 3 years (2008-2010) with ~400K annotation points.
Labels include benthic categories (CCA, Sand, Turf, Porit, Macro, etc.) with
case inconsistencies that are normalized by this loader.

For classification experiments, each image is assigned a label based on the
dominant (most frequent) class among its annotation points.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from data.base_dataset import BaseMarineDataset

logger = logging.getLogger(__name__)


# Canonical label mapping: normalize case variants to uppercase
_LABEL_CANONICAL = {
    "cca": "CCA",
    "Cca": "CCA",
    "sand": "SAND",
    "Sand": "SAND",
    "turf": "TURF",
    "Turf": "TURF",
    "porit": "PORIT",
    "Porit": "PORIT",
    "macro": "MACRO",
    "Macro": "MACRO",
    "pocill": "POCILL",
    "Pocill": "POCILL",
    "off": "OFF",
    "Off": "OFF",
    "monti": "MONTI",
    "Monti": "MONTI",
    "pavon": "PAVON",
    "Pavon": "PAVON",
    "acrop": "ACROP",
    "Acrop": "ACROP",
    "mille": "MILLE",
    "Mille": "MILLE",
    "monta": "MONTA",
    "Monta": "MONTA",
    "lepta": "LEPTA",
    "Lepta": "LEPTA",
    "soft": "SOFT",
    "Soft": "SOFT",
    "fung": "FUNG",
    "Fung": "FUNG",
    "gardin": "GARDIN",
    "Gardin": "GARDIN",
    "herpo": "HERPO",
    "Herpo": "HERPO",
    "lobo": "LOBO",
    "Lobo": "LOBO",
    "favia": "FAVIA",
    "Favia": "FAVIA",
    "astreo": "ASTREO",
    "Astreo": "ASTREO",
    "cypha": "CYPHA",
    "Cypha": "CYPHA",
    "acan": "ACAN",
    "Acan": "ACAN",
    "stylo": "STYLO",
    "Stylo": "STYLO",
    "psam": "PSAM",
    "Psam": "PSAM",
    "sando": "SANDO",
    "Sando": "SANDO",
    "tuba": "TUBA",
    "Tuba": "TUBA",
    "lepto": "LEPTO",
    "Lepto": "LEPTO",
    "p. rus": "P_RUS",
    "P. Rus": "P_RUS",
    "P. rus": "P_RUS",
    "p mass": "P_MASS",
    "P mass": "P_MASS",
    "P. Irr": "P_IRR",
    "p. irr": "P_IRR",
    "P. irr": "P_IRR",
}

# Major benthic categories for simplified classification
MAJOR_CLASSES = [
    "CCA", "SAND", "TURF", "PORIT", "MACRO", "POCILL",
    "OFF", "MONTI", "PAVON",
]


class MooreaCoralsDataset(BaseMarineDataset):
    """Moorea Labeled Corals dataset.

    For classification, assigns each image a label based on the dominant
    benthic class among its annotation points (excluding 'OFF' by default).

    Args:
        root_dir: Path to Moorea_Labeled_Corals root.
        split: 'train', 'val', or 'test'. Split by year: 2008+2009=train, 2010=test.
        transform: Image transform pipeline.
        mode: 'classification'.
        image_size: Target image size.
        use_major_classes_only: If True, only use the 9 major benthic classes.
        exclude_off: If True, exclude 'OFF' (off-quadrat) from classification.
    """

    DATASET_NAME = "Moorea"
    SPECIES_NAMES = MAJOR_CLASSES
    HABITAT_NAMES = ["moorea"]

    # Year-based data directories
    _YEAR_DIRS = {
        "2008": "MCR_LTER_ComputerVision_LabeledCorals_2008_20120521/2008",
        "2009": "MCR_LTER_ComputerVision_LabeledCorals_2009_20120523/2009",
        "2010": "MCR_LTER_ComputerVision_LabeledCorals_2010_20120521/2010",
    }

    # Default split: 2008+2009 for train, 2010 for test
    _SPLIT_YEARS = {
        "train": ["2008", "2009"],
        "val": ["2010"],  # Use part of 2010 or same as test
        "test": ["2010"],
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
        use_major_classes_only: bool = True,
        exclude_off: bool = True,
    ) -> None:
        if mode != "classification":
            logger.warning(
                "Moorea dataset in classification mode only. Ignoring mode=%s.", mode
            )
            mode = "classification"
        self.use_major_classes_only = use_major_classes_only
        self.exclude_off = exclude_off
        super().__init__(root_dir, split, transform, mode, image_size)

    def _normalize_label(self, raw_label: str) -> str:
        """Normalize a raw label to canonical uppercase form."""
        label = raw_label.strip()
        if label in _LABEL_CANONICAL:
            return _LABEL_CANONICAL[label]
        return label.upper()

    def _parse_annotation_file(self, ann_path: Path) -> List[Dict[str, Any]]:
        """Parse a .jpg.txt annotation file.

        Returns list of {'row': int, 'col': int, 'label': str} dicts.
        """
        points = []
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(";")
                if len(parts) >= 3:
                    try:
                        row = int(parts[0].strip())
                        col = int(parts[1].strip())
                        label = self._normalize_label(parts[2])
                        points.append({"row": row, "col": col, "label": label})
                    except (ValueError, IndexError):
                        continue
        return points

    def _get_dominant_class(self, points: List[Dict[str, Any]]) -> Optional[str]:
        """Get the most frequent class from annotation points.

        Args:
            points: List of annotation point dicts.

        Returns:
            Most frequent class label, or None if no valid points.
        """
        labels = [p["label"] for p in points]

        if self.exclude_off:
            labels = [l for l in labels if l != "OFF"]

        if self.use_major_classes_only:
            labels = [l for l in labels if l in MAJOR_CLASSES]

        if not labels:
            return None

        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    def _load_annotations(self) -> None:
        """Load Moorea annotations, assigning dominant class per image."""
        base_dir = self.root_dir / "knb-lter-mcr.5006.3"
        if not base_dir.exists():
            # Try root directly
            base_dir = self.root_dir

        years = self._SPLIT_YEARS.get(self.split, ["2008", "2009", "2010"])

        for year in years:
            year_rel = self._YEAR_DIRS.get(year)
            if year_rel is None:
                continue

            year_dir = base_dir / year_rel
            if not year_dir.exists():
                logger.warning("Year directory not found: %s", year_dir)
                continue

            # Find all image files with annotation files
            for img_path in sorted(year_dir.glob("*.jpg")):
                ann_path = year_dir / f"{img_path.name}.txt"
                if not ann_path.exists():
                    continue

                points = self._parse_annotation_file(ann_path)
                dominant_class = self._get_dominant_class(points)

                if dominant_class is None:
                    continue

                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": dominant_class,
                        "habitat": "moorea",
                        "year": year,
                        "n_points": len(points),
                    }
                )

        if not self.samples:
            logger.warning(
                "No samples loaded for Moorea split=%s. Check directory: %s",
                self.split,
                base_dir,
            )
