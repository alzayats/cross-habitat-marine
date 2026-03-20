"""Extract features from trained models for visualization and analysis.

Saves features as .npz files with features, labels, habitat IDs,
and image paths for downstream analysis (t-SNE, CKA, per-species).

Usage:
    python -m analysis.extract_features --model dinov2_base --adaptation lora_r4 \
        --dataset deepfish --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_test_transforms
from data.deepfish_dataset import DeepFishDataset
from data.aqua20_dataset import AQUA20Dataset
from data.cots_gbr_dataset import COTSGBRDataset
from data.moorea_dataset import MooreaCoralsDataset
from data.coralscapes_dataset import CoralscapesDataset
from data.brackish_dataset import BrackishDataset
from models.model_factory import ModelFactory
from utils.seed_utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_MAP = {
    "deepfish": (DeepFishDataset, "DeepFish"),
    "aqua20": (AQUA20Dataset, "AQUA20_dataset"),
    "cots_gbr": (COTSGBRDataset, "Cots_GBR_dataset"),
    "moorea": (MooreaCoralsDataset, "Moorea_Labeled_Corals"),
    "coralscapes": (CoralscapesDataset, "Coralscapes_dataset"),
    "brackish": (BrackishDataset, "Brackish_dataset"),
}


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract penultimate-layer features from a model.

    Args:
        model: Model with get_features() method.
        dataloader: Data loader.
        device: Device.

    Returns:
        Dict with 'features', 'labels', 'habitats', 'image_paths'.
    """
    model.eval()
    model = model.to(device)

    all_features = []
    all_labels = []
    all_habitats = []
    all_paths = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch["image"].to(device)

        if hasattr(model, "get_features"):
            features = model.get_features(images)
        else:
            features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.extend(
            batch["label"].numpy().tolist()
            if isinstance(batch["label"], torch.Tensor)
            else batch["label"]
        )
        all_habitats.extend(batch.get("habitat", ["unknown"] * len(images)))
        all_paths.extend(batch.get("image_path", [""] * len(images)))

    return {
        "features": np.concatenate(all_features, axis=0),
        "labels": np.array(all_labels),
        "habitats": np.array(all_habitats),
        "image_paths": np.array(all_paths),
    }


def save_features(
    data: Dict[str, np.ndarray],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save extracted features to .npz file.

    Args:
        data: Feature data dict.
        output_path: Output file path.
        metadata: Optional metadata to save alongside.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {k: v for k, v in data.items()}
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(str(v))

    np.savez_compressed(str(path), **save_dict)
    logger.info(
        "Features saved: %s (shape=%s, %d samples)",
        path,
        data["features"].shape,
        len(data["labels"]),
    )


def main() -> None:
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features for analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adaptation", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", default="./outputs/features")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    base_config = OmegaConf.load(args.config)
    set_seed(42)

    factory = ModelFactory()
    model_cfg = factory.load_model_config(args.model)
    image_size = model_cfg.image_size

    test_transform = get_test_transforms(image_size=image_size)

    ds_cls, ds_dir = DATASET_MAP[args.dataset]
    data_root = str(Path(base_config.data.root_dir) / ds_dir)

    kwargs: Dict[str, Any] = {
        "root_dir": data_root,
        "split": "test",
        "transform": test_transform,
        "mode": "classification",
        "image_size": image_size,
    }
    dataset = ds_cls(**kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=base_config.data.num_workers,
        pin_memory=True,
    )

    # Create and load model
    model = factory.create(args.model, args.adaptation, dataset.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    device = base_config.hardware.device
    data = extract_features(model, dataloader, device)

    output_name = f"{args.model}_{args.adaptation}_{args.dataset}_features.npz"
    output_path = str(Path(args.output_dir) / output_name)

    save_features(
        data,
        output_path,
        metadata={
            "model": args.model,
            "adaptation": args.adaptation,
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
        },
    )


if __name__ == "__main__":
    main()
