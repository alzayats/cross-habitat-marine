"""Brackish Underwater dataset loader.

Video-based underwater detection dataset from temperate Danish waters.
Contains 89 videos with ~14K extracted frames and 6 object classes.
Annotations are in COCO format.

The dataset includes pre-defined train/valid/test splits via text files
listing frame filenames. Frames must be extracted from AVI videos.

Structure:
    Brackish_dataset/
    ├── dataset/videos/
    │   ├── crab/             (AVI video files)
    │   ├── fish-big/
    │   ├── fish-school/
    │   ├── fish-small-shrimp/
    │   └── jellyfish/
    ├── annotations/
    │   ├── annotations_COCO/
    │   │   ├── train_groundtruth.json
    │   │   ├── valid_groundtruth.json
    │   │   └── test_groundtruth.json
    │   ├── annotations_YOLO/
    │   └── annotations_AAU/
    ├── train.txt
    ├── valid.txt
    └── test.txt

Categories (from COCO annotations):
    1: fish, 2: small_fish, 3: crab, 4: shrimp, 5: jellyfish, 6: starfish
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from data.base_dataset import BaseMarineDataset

logger = logging.getLogger(__name__)


class BrackishDataset(BaseMarineDataset):
    """Brackish underwater dataset for classification and detection.

    For classification, images are labeled by the dominant object class
    present in their COCO annotations. Images with no annotations are
    labeled as 'background'.

    Uses the official train/valid/test splits from text files. Frame images
    are referenced from the COCO annotation image entries.

    Args:
        root_dir: Path to Brackish_dataset root.
        split: 'train', 'val'/'valid', or 'test'.
        transform: Image transform pipeline.
        mode: 'classification' or 'detection'.
        image_size: Target image size.
        frames_dir: Directory containing extracted frames (if None, searches
            common locations).
    """

    DATASET_NAME = "Brackish"
    SPECIES_NAMES = ["fish", "small_fish", "crab", "shrimp", "jellyfish", "starfish"]
    HABITAT_NAMES = ["brackish"]

    # Map category ID to name
    _CATEGORY_MAP = {
        1: "fish",
        2: "small_fish",
        3: "crab",
        4: "shrimp",
        5: "jellyfish",
        6: "starfish",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
        frames_dir: Optional[str] = None,
    ) -> None:
        self.frames_dir = Path(frames_dir) if frames_dir else None
        super().__init__(root_dir, split, transform, mode, image_size)

    def _find_frames_dir(self) -> Optional[Path]:
        """Find directory containing extracted video frames."""
        if self.frames_dir and self.frames_dir.exists():
            return self.frames_dir

        # Common locations for extracted frames
        candidates = [
            self.root_dir / "frames",
            self.root_dir / "images",
            self.root_dir / "extracted_frames",
        ]
        for cand in candidates:
            if cand.exists() and any(cand.iterdir()):
                return cand

        return None

    def _get_coco_json_path(self) -> Path:
        """Get the path to the COCO annotation JSON for this split."""
        split_map = {
            "train": "train_groundtruth.json",
            "val": "valid_groundtruth.json",
            "valid": "valid_groundtruth.json",
            "test": "test_groundtruth.json",
        }
        filename = split_map.get(self.split, f"{self.split}_groundtruth.json")
        return self.root_dir / "annotations" / "annotations_COCO" / filename

    def _extract_video_id(self, filename: str) -> str:
        """Extract the video session identifier from a frame filename.

        Filenames look like: 2019-02-21_06-36-31to2019-02-21_06-36-39_1-0082.png
        The video ID is everything before the last '-NNNN.ext'.
        """
        stem = Path(filename).stem
        parts = stem.rsplit("-", 1)
        if len(parts) == 2:
            return parts[0]
        return stem

    def _load_annotations(self) -> None:
        """Load Brackish annotations from COCO JSON format."""
        coco_path = self._get_coco_json_path()

        if not coco_path.exists():
            logger.warning("COCO annotation file not found: %s", coco_path)
            return

        with open(coco_path, "r") as f:
            coco_data = json.load(f)

        # Build category map from annotation data
        cat_map = {}
        for cat in coco_data.get("categories", []):
            cat_map[cat["id"]] = cat["name"]

        if not cat_map:
            cat_map = self._CATEGORY_MAP

        # Build image ID to annotations mapping
        image_annotations: Dict[int, List[Dict]] = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # Find frames directory
        frames_dir = self._find_frames_dir()

        # Process each image entry
        for img_info in coco_data.get("images", []):
            img_id = img_info["id"]
            filename = img_info["file_name"]

            # Try to find the actual image file
            img_path = None
            if frames_dir:
                img_path = frames_dir / filename
                if not img_path.exists():
                    img_path = None

            if img_path is None:
                # Try root directory
                img_path = self.root_dir / filename
                if not img_path.exists():
                    img_path = self.root_dir / "frames" / filename
                    if not img_path.exists():
                        # Skip samples with missing image files
                        continue

            video_id = self._extract_video_id(filename)
            anns = image_annotations.get(img_id, [])

            if self.mode == "detection":
                # Detection mode: include bounding boxes
                boxes = []
                box_labels = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    box_labels.append(cat_map.get(ann["category_id"], "unknown"))

                label = box_labels[0] if box_labels else "background"

                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": label,
                        "habitat": "brackish",
                        "video_id": video_id,
                        "boxes": boxes,
                        "box_labels": box_labels,
                    }
                )
            else:
                # Classification mode: dominant class
                if anns:
                    from collections import Counter
                    class_counts = Counter(
                        cat_map.get(a["category_id"], "unknown") for a in anns
                    )
                    label = class_counts.most_common(1)[0][0]
                else:
                    label = "background"

                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": label,
                        "habitat": "brackish",
                        "video_id": video_id,
                    }
                )

        if not self.samples:
            logger.warning(
                "No samples loaded for Brackish split=%s. Check: %s",
                self.split,
                coco_path,
            )
