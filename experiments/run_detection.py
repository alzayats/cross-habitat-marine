"""Detection experiments on COTS GBR and Brackish datasets.

Evaluates foundation model features for marine object detection using
a lightweight detection head.

Usage:
    python -m experiments.run_detection --model dinov2_base --adaptation linear_probe \
        --dataset cots_gbr
    python -m experiments.run_detection --model dinov2_base --adaptation lora_r4 \
        --dataset brackish
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.augmentations import get_test_transforms, get_train_transforms
from data.cots_gbr_dataset import COTSGBRDataset
from data.brackish_dataset import BrackishDataset
from models.model_factory import ModelFactory
from training.evaluator import Evaluator
from utils.io_utils import save_results
from utils.seed_utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DETECTION_DATASETS = {
    "cots_gbr": {
        "class": COTSGBRDataset,
        "dir": "Cots_GBR_dataset",
        "mode": "detection",
    },
    "brackish": {
        "class": BrackishDataset,
        "dir": "Brackish_dataset",
        "mode": "detection",
    },
}


def run_detection_experiment(
    model_name: str,
    adaptation_name: str,
    dataset_name: str,
    seed: int,
    base_config: Any,
    output_dir: str = "./outputs/detection",
) -> Dict[str, Any]:
    """Run a detection experiment.

    Args:
        model_name: Model config name.
        adaptation_name: Adaptation strategy.
        dataset_name: 'cots_gbr' or 'brackish'.
        seed: Random seed.
        base_config: Base configuration.
        output_dir: Output directory.

    Returns:
        Detection experiment results.
    """
    set_seed(seed)

    ds_info = DETECTION_DATASETS.get(dataset_name)
    if ds_info is None:
        raise ValueError(f"Unknown detection dataset: {dataset_name}")

    exp_name = f"detection_{dataset_name}_{model_name}_{adaptation_name}_seed{seed}"
    logger.info("Detection experiment: %s", exp_name)

    factory = ModelFactory()
    model_cfg = factory.load_model_config(model_name)
    image_size = model_cfg.image_size

    train_transform = get_train_transforms(image_size=image_size, underwater_augment=True)
    test_transform = get_test_transforms(image_size=image_size)

    data_root = base_config.data.root_dir
    dataset_root = str(Path(data_root) / ds_info["dir"])
    ds_cls = ds_info["class"]
    mode = ds_info["mode"]

    train_ds = ds_cls(
        root_dir=dataset_root,
        split="train",
        transform=train_transform,
        mode=mode,
        image_size=image_size,
    )
    test_ds = ds_cls(
        root_dir=dataset_root,
        split="test",
        transform=test_transform,
        mode=mode,
        image_size=image_size,
    )

    num_classes = train_ds.num_classes
    batch_size = base_config.data.batch_size_detection

    model = factory.create(model_name, adaptation_name, num_classes)

    if mode == "detection":
        from models.detection_head import SimpleDetectionHead

        det_head = SimpleDetectionHead(
            in_features=model_cfg.feature_dim,
            num_classes=num_classes,
            patch_size=model_cfg.patch_size,
        )
        model.detection_head = det_head

    model = model.to(base_config.hardware.device)

    logger.info(
        "Detection model created: %s + %s on %s, classes=%d",
        model_name, adaptation_name, dataset_name, num_classes,
    )

    results = {
        "experiment": exp_name,
        "dataset": dataset_name,
        "model": model_name,
        "adaptation": adaptation_name,
        "seed": seed,
        "num_classes": num_classes,
        "status": "setup_complete",
    }

    results_path = Path(output_dir) / exp_name / "results.json"
    save_results(results, str(results_path))

    return results


def main() -> None:
    """Main entry point for detection experiments."""
    parser = argparse.ArgumentParser(description="Detection on COTS GBR / Brackish")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adaptation", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, default="cots_gbr",
        choices=["cots_gbr", "brackish"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./outputs/detection")
    parser.add_argument("--config", default="configs/base_config.yaml")

    args = parser.parse_args()
    base_config = OmegaConf.load(args.config)

    run_detection_experiment(
        args.model, args.adaptation, args.dataset,
        args.seed, base_config, args.output_dir,
    )


if __name__ == "__main__":
    main()
