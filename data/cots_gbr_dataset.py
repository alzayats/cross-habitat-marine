"""COTS GBR (Crown-of-Thorns Starfish) dataset loader.

Full Kaggle competition dataset with 23,501 video frames across 3 videos.
Supports both classification (COTS present/absent) and detection (bounding boxes).

Structure:
    Cots_GBR_dataset/
    ├── train.csv              (23,501 rows with JSON bounding box annotations)
    └── train_images/
        ├── video_0/           (6,708 frames)
        ├── video_1/           (8,232 frames)
        └── video_2/           (8,561 frames)

Annotations in train.csv are JSON strings:
    [{'x': 559, 'y': 213, 'width': 50, 'height': 32}, ...] or []

Split by video_id to prevent temporal data leakage.
"""

import ast
import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from data.base_dataset import BaseMarineDataset

logger = logging.getLogger(__name__)


class COTSGBRDataset(BaseMarineDataset):
    """COTS GBR dataset from the Kaggle TensorFlow COTS competition.

    Supports two modes:
    - classification: binary labels "cots" (annotations present) / "no_cots" (empty)
    - detection: bounding boxes [x1, y1, x2, y2] with single class "cots"

    Splits by video_id to prevent temporal data leakage between train/test.

    Args:
        root_dir: Path to Cots_GBR_dataset root.
        split: 'train', 'val', or 'test'.
        transform: Image transform pipeline.
        mode: 'classification' or 'detection'.
        image_size: Target image size.
        train_videos: Video IDs for training (default: [0, 1]).
        test_videos: Video IDs for testing (default: [2]).
        val_fraction: Fraction of train videos to hold out for validation.
        val_seed: Random seed for train/val split within train videos.
    """

    DATASET_NAME = "COTS_GBR"
    SPECIES_NAMES = ["cots", "no_cots"]
    HABITAT_NAMES = ["gbr"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "classification",
        image_size: int = 224,
        train_videos: Optional[List[int]] = None,
        test_videos: Optional[List[int]] = None,
        val_fraction: float = 0.1,
        val_seed: int = 42,
    ) -> None:
        self.train_videos = train_videos if train_videos is not None else [0, 1]
        self.test_videos = test_videos if test_videos is not None else [2]
        self.val_fraction = val_fraction
        self.val_seed = val_seed
        super().__init__(root_dir, split, transform, mode, image_size)

    def _parse_annotations(self, ann_str: str) -> List[Dict[str, int]]:
        """Parse JSON annotation string into list of box dicts."""
        ann_str = ann_str.strip()
        if not ann_str or ann_str == "[]":
            return []
        try:
            return ast.literal_eval(ann_str)
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse annotation: %s", ann_str[:80])
            return []

    def _load_annotations(self) -> None:
        """Load annotations from train.csv and split by video_id."""
        csv_path = self.root_dir / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"train.csv not found at {csv_path}")

        # Read all rows
        all_rows: List[Dict[str, Any]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = int(row["video_id"])
                image_id = row["image_id"]
                # image_id format is "{video_id}-{frame}", frame number is the filename
                frame_num = image_id.split("-", 1)[1]
                image_path = (
                    self.root_dir
                    / "train_images"
                    / f"video_{video_id}"
                    / f"{frame_num}.jpg"
                )
                annotations = self._parse_annotations(row["annotations"])

                all_rows.append(
                    {
                        "video_id": video_id,
                        "sequence": int(row["sequence"]),
                        "image_id": image_id,
                        "image_path": image_path,
                        "annotations": annotations,
                    }
                )

        if not all_rows:
            logger.warning("No rows found in %s", csv_path)
            return

        # Split by video_id
        if self.split == "test":
            split_rows = [r for r in all_rows if r["video_id"] in self.test_videos]
        else:
            # train and val come from train_videos
            train_pool = [r for r in all_rows if r["video_id"] in self.train_videos]

            if self.val_fraction > 0 and self.split in ("train", "val"):
                # Split train_pool by sequence to keep temporal coherence
                import numpy as np

                sequences = sorted(
                    set((r["video_id"], r["sequence"]) for r in train_pool)
                )
                rng = np.random.RandomState(self.val_seed)
                perm = rng.permutation(len(sequences))
                n_val = max(1, int(len(sequences) * self.val_fraction))

                if self.split == "val":
                    val_seqs = set(sequences[i] for i in perm[:n_val])
                    split_rows = [
                        r
                        for r in train_pool
                        if (r["video_id"], r["sequence"]) in val_seqs
                    ]
                else:
                    train_seqs = set(sequences[i] for i in perm[n_val:])
                    split_rows = [
                        r
                        for r in train_pool
                        if (r["video_id"], r["sequence"]) in train_seqs
                    ]
            else:
                split_rows = train_pool

        # Build samples based on mode
        for row in split_rows:
            annotations = row["annotations"]
            has_cots = len(annotations) > 0

            sample: Dict[str, Any] = {
                "image_path": row["image_path"],
                "habitat": "gbr",
                "video_id": row["video_id"],
                "sequence": row["sequence"],
                "image_id": row["image_id"],
            }

            if self.mode == "classification":
                sample["label"] = "cots" if has_cots else "no_cots"
            else:
                # Detection mode
                sample["label"] = "cots" if has_cots else "no_cots"
                if has_cots:
                    boxes = [
                        [a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]]
                        for a in annotations
                    ]
                    sample["boxes"] = boxes
                    sample["box_labels"] = ["cots"] * len(boxes)
                else:
                    sample["boxes"] = []
                    sample["box_labels"] = []

            self.samples.append(sample)

        n_cots = sum(1 for s in self.samples if s["label"] == "cots")
        n_no = len(self.samples) - n_cots
        logger.info(
            "COTS GBR split=%s mode=%s: %d samples (cots=%d, no_cots=%d)",
            self.split,
            self.mode,
            len(self.samples),
            n_cots,
            n_no,
        )
