"""Protocol B: Cross-dataset geographic transfer experiments.

Evaluates transfer across different marine datasets along a difficulty gradient:
  EASY:    DeepFish → AQUA20 (tropical fish → mixed marine species)
  MEDIUM:  DeepFish → COTS GBR (fish → starfish pest detection, same ocean)
  HARD:    DeepFish → Moorea (fish → coral cover, cross-Pacific)
  HARDER:  AQUA20 → Coralscapes (tropical → Red Sea benthic, cross-ocean)
  HARDEST: DeepFish → Brackish (tropical reef → cold temperate)

Usage:
    python -m experiments.run_cross_dataset --model dinov2_base --adaptation lora_r4 \
        --source deepfish --target aqua20 --strategy open_set
    python -m experiments.run_cross_dataset --all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.augmentations import get_test_transforms, get_train_transforms
from data.base_dataset import BaseMarineDataset
from data.deepfish_dataset import DeepFishDataset
from data.aqua20_dataset import AQUA20Dataset
from data.cots_gbr_dataset import COTSGBRDataset
from data.moorea_dataset import MooreaCoralsDataset
from data.coralscapes_dataset import CoralscapesDataset
from data.brackish_dataset import BrackishDataset
from models.model_factory import ModelFactory
from training.evaluator import Evaluator
from training.feature_cache import (
    extract_and_cache_features,
    CachedFeatureDataset,
    LinearProbeTrainer,
)
from training.trainer import Trainer
from utils.io_utils import save_results
from utils.seed_utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_CLASSES = {
    "deepfish": DeepFishDataset,
    "aqua20": AQUA20Dataset,
    "cots_gbr": COTSGBRDataset,
    "moorea": MooreaCoralsDataset,
    "coralscapes": CoralscapesDataset,
    "brackish": BrackishDataset,
}

DATASET_DIRS = {
    "deepfish": "DeepFish",
    "aqua20": "AQUA20_dataset",
    "cots_gbr": "Cots_GBR_dataset",
    "moorea": "Moorea_Labeled_Corals",
    "coralscapes": "Coralscapes_dataset",
    "brackish": "Brackish_dataset",
}


def _get_dataset(
    name: str,
    root_dir: str,
    split: str,
    transform: Any,
    image_size: int = 224,
) -> BaseMarineDataset:
    """Load a dataset by name.

    Args:
        name: Dataset name.
        root_dir: Base data root directory.
        split: Data split.
        transform: Image transform.
        image_size: Target image size.

    Returns:
        Dataset instance.
    """
    cls = DATASET_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CLASSES.keys())}")

    dataset_root = Path(root_dir) / DATASET_DIRS[name]

    kwargs: Dict[str, Any] = {
        "root_dir": str(dataset_root),
        "split": split,
        "transform": transform,
        "mode": "classification",
        "image_size": image_size,
    }

    return cls(**kwargs)


def _get_species_intersection(
    source_ds: BaseMarineDataset,
    target_ds: BaseMarineDataset,
) -> Set[str]:
    """Find overlapping species between source and target datasets."""
    source_species = set(source_ds.class_to_idx.keys())
    target_species = set(target_ds.class_to_idx.keys())
    intersection = source_species & target_species
    logger.info(
        "Species intersection: source=%d, target=%d, shared=%d",
        len(source_species),
        len(target_species),
        len(intersection),
    )
    return intersection


def _filter_dataset_by_species(
    dataset: BaseMarineDataset,
    keep_species: Set[str],
) -> List[int]:
    """Get indices of samples belonging to specified species."""
    return [
        i for i, sample in enumerate(dataset.samples)
        if sample["label"] in keep_species
    ]


def align_label_spaces(*datasets: BaseMarineDataset) -> Dict[str, int]:
    """Force several datasets onto one shared class-name -> index mapping.

    ``BaseMarineDataset._build_class_mapping`` derives ``class_to_idx`` from the
    classes present in a single split. Open-set evaluation builds prototypes from the
    target *train* split and scores them against the target *test* split, so whenever
    those two splits contain different class sets the same integer denotes different
    classes on each side and the comparison is meaningless.

    This is not hypothetical: Moorea has 8 train / 7 test classes (POCILL is absent
    from test) and Coralscapes has 21 / 15, leaving only one aligned index. AQUA20 and
    Brackish are unaffected because their splits share identical class sets.

    Labels are resolved through ``class_to_idx`` inside ``__getitem__``, so rewriting
    the mapping is sufficient -- no need to touch ``samples``.

    Args:
        *datasets: Datasets to place on a common label space.

    Returns:
        The shared class-name -> index mapping.
    """
    union = sorted(set().union(*(set(d.class_to_idx) for d in datasets)))
    shared = {cls: idx for idx, cls in enumerate(union)}
    for dataset in datasets:
        dataset.class_to_idx = dict(shared)
        dataset.idx_to_class = {idx: cls for cls, idx in shared.items()}
    logger.info(
        "Aligned label spaces across %d datasets: %d shared classes",
        len(datasets), len(shared),
    )
    return shared


def run_single_cross_dataset(
    model_name: str,
    adaptation_name: str,
    source_name: str,
    target_name: str,
    strategy: str,
    seed: int,
    base_config: Any,
    output_dir: str = "./outputs/cross_dataset",
) -> Dict[str, Any]:
    """Run a single cross-dataset transfer experiment."""
    set_seed(seed)

    exp_name = (
        f"{model_name}_{adaptation_name}_{source_name}_to_{target_name}"
        f"_{strategy}_seed{seed}"
    )

    # Skip if results already exist
    import json
    exp_dir = Path(output_dir) / exp_name
    results_path = exp_dir / "results.json"
    if results_path.exists():
        logger.info("SKIP (results exist): %s", exp_name)
        with open(results_path) as f:
            return json.load(f)

    # Check for crashed run: checkpoint exists but no results
    best_ckpt_path = exp_dir / "checkpoints" / "best_model.pt"
    history_path = exp_dir / "training_history.json"
    crashed = best_ckpt_path.exists() and not results_path.exists()

    if crashed:
        logger.info("=" * 60)
        logger.info("RESUME (crash recovery — eval only): %s", exp_name)
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Cross-dataset experiment: %s", exp_name)
        logger.info("=" * 60)

    factory = ModelFactory()
    model_cfg = factory.load_model_config(model_name)
    image_size = model_cfg.image_size

    train_transform = get_train_transforms(image_size=image_size)
    test_transform = get_test_transforms(image_size=image_size)

    data_root = base_config.data.root_dir

    # Load datasets
    source_ds = _get_dataset(source_name, data_root, "train", train_transform, image_size)
    target_ds = _get_dataset(target_name, data_root, "test", test_transform, image_size)

    # Apply species mapping strategy
    if strategy == "intersection":
        shared = _get_species_intersection(source_ds, target_ds)
        if len(shared) < 2:
            logger.warning("Too few shared species (%d). Skipping.", len(shared))
            return {"status": "skipped", "reason": "insufficient_overlap"}

        source_indices = _filter_dataset_by_species(source_ds, shared)
        target_indices = _filter_dataset_by_species(target_ds, shared)
        source_ds = source_ds.get_subset(source_indices)
        target_ds = target_ds.get_subset(target_indices)
        num_classes = len(shared)
        target_train_ds = None

    elif strategy == "open_set":
        # Open-set: train on source species, evaluate using target prototypes.
        # Prototypes are built from TARGET train split (one per target class).
        num_classes = source_ds.num_classes
        target_train_ds = _get_dataset(
            target_name, data_root, "train", test_transform, image_size,
        )
        # Prototypes come from the target train split and are scored against the
        # target test split, so both must share one label space (see
        # align_label_spaces). Without this, targets whose splits hold different
        # class sets are evaluated against misaligned integer labels.
        align_label_spaces(target_train_ds, target_ds)
        logger.info(
            "Open-set: source classes=%d, target classes=%d, target train=%d",
            source_ds.num_classes, target_train_ds.num_classes, len(target_train_ds),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create data loaders
    batch_size = base_config.data.batch_size
    # Reduce batch size for full fine-tune with large image sizes to avoid OOM
    adaptation_cfg_check = factory.load_adaptation_config(adaptation_name)
    if adaptation_cfg_check.type == "full_finetune" and image_size >= 380:
        batch_size = max(batch_size // 4, 4)
        logger.info("Reduced batch_size to %d for full_finetune at image_size=%d",
                     batch_size, image_size)
    elif adaptation_cfg_check.type == "full_finetune":
        batch_size = max(batch_size // 2, 8)
        logger.info("Reduced batch_size to %d for full_finetune", batch_size)
    eval_batch_size = max(batch_size // 2, 1)
    num_workers = base_config.data.num_workers
    device = base_config.hardware.device

    is_linear_probe = adaptation_cfg_check.type == "linear_probe"
    features_dir = getattr(base_config.project, "features_dir", "./outputs/features")

    # === LINEAR PROBE FAST PATH: feature caching ===
    # For LP, the backbone is frozen — extract features once, train head on tensors.
    # Open-set eval also uses backbone features, so cached features work end-to-end.
    if is_linear_probe and not crashed:
        logger.info("LINEAR PROBE FAST PATH: using cached backbone features")

        # Create model for feature extraction only
        feat_model = factory.create(model_name, adaptation_name, num_classes)

        # Cache source features
        src_feats, src_labels = extract_and_cache_features(
            model=feat_model, dataset=source_ds,
            model_name=model_name, dataset_name=source_name,
            split="train", image_size=image_size,
            cache_dir=features_dir, device=device,
            batch_size=batch_size, num_workers=num_workers,
        )

        # Cache target test features
        tgt_feats, tgt_labels = extract_and_cache_features(
            model=feat_model, dataset=target_ds,
            model_name=model_name, dataset_name=target_name,
            split="test", image_size=image_size,
            cache_dir=features_dir, device=device,
            batch_size=batch_size, num_workers=num_workers,
        )

        # Cache target train features (for open-set prototypes)
        if target_train_ds is not None:
            tgt_train_feats, tgt_train_labels = extract_and_cache_features(
                model=feat_model, dataset=target_train_ds,
                model_name=model_name, dataset_name=target_name,
                split="train", image_size=image_size,
                cache_dir=features_dir, device=device,
                batch_size=batch_size, num_workers=num_workers,
            )

        del feat_model
        torch.cuda.empty_cache()

        feature_dim = src_feats.shape[1]

        # Split source into train/val
        rng = np.random.RandomState(seed)
        n = len(source_ds)
        indices = rng.permutation(n)
        n_val = int(n * base_config.data.val_split)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        src_train_feats = CachedFeatureDataset(src_feats, src_labels, indices=train_idx)
        src_val_feats = CachedFeatureDataset(src_feats, src_labels, indices=val_idx)

        # Train linear head on cached features
        adaptation_cfg = factory.load_adaptation_config(adaptation_name)
        lp = LinearProbeTrainer(
            feature_dim=feature_dim,
            num_classes=num_classes,
            lr=getattr(adaptation_cfg, "learning_rate", 1e-3),
            epochs=getattr(adaptation_cfg, "epochs", 50),
            patience=getattr(adaptation_cfg, "early_stopping_patience", 10),
            device=device,
        )
        train_results = lp.train(src_train_feats, src_val_feats)

        # Evaluation
        if strategy == "open_set":
            tgt_proto_ds = CachedFeatureDataset(tgt_train_feats, tgt_train_labels)
            test_metrics = _evaluate_open_set_cached(
                tgt_proto_ds, tgt_feats, tgt_labels,
            )
        else:
            tgt_ds_feats = CachedFeatureDataset(tgt_feats, tgt_labels)
            test_metrics = lp.evaluate(tgt_ds_feats)

        results = {
            "experiment": exp_name,
            "model": model_name,
            "adaptation": adaptation_name,
            "source": source_name,
            "target": target_name,
            "strategy": strategy,
            "seed": seed,
            "num_classes": num_classes,
            "train_results": train_results,
            "test_metrics": test_metrics,
        }

        save_results(results, str(results_path))
        return results

    # === STANDARD PATH: full training (non-LP or crash recovery) ===
    if crashed:
        # --- Crash recovery: load best model directly ---
        logger.info("Loading best checkpoint from %s", best_ckpt_path)
        model = factory.create(model_name, adaptation_name, num_classes)
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        train_results = {
            "best_epoch": checkpoint.get("best_epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", 0),
            "total_epochs": checkpoint.get("epoch", 0) + 1,
            "total_time_seconds": 0,
            "history": checkpoint.get("history", []),
        }
        if history_path.exists():
            with open(history_path) as f:
                train_results["history"] = json.load(f)

        del checkpoint
        torch.cuda.empty_cache()
    else:
        # --- Normal path: full training ---
        # Split source into train/val
        rng = np.random.RandomState(seed)
        n = len(source_ds)
        indices = rng.permutation(n)
        n_val = int(n * base_config.data.val_split)

        from data.base_dataset import SubsetMarineDataset

        if hasattr(source_ds, "dataset"):
            train_subset = SubsetMarineDataset(source_ds.dataset, indices[n_val:].tolist())
            val_subset = SubsetMarineDataset(source_ds.dataset, indices[:n_val].tolist())
        else:
            train_subset = source_ds.get_subset(indices[n_val:].tolist())
            val_subset = source_ds.get_subset(indices[:n_val].tolist())

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        model = factory.create(model_name, adaptation_name, num_classes)
        adaptation_cfg = factory.load_adaptation_config(adaptation_name)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=base_config.training,
            adaptation_config=adaptation_cfg,
            output_dir=output_dir,
            experiment_name=exp_name,
            device=device,
        )

        train_results = trainer.train()
        trainer.load_best_model()
        model = trainer.model

        # Free trainer memory before eval
        del trainer.optimizer, trainer.scheduler, trainer.scaler
        del trainer
        torch.cuda.empty_cache()

    # --- Evaluation ---
    test_loader = DataLoader(
        target_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    if strategy == "open_set":
        # For open_set, build prototypes from TARGET train data
        proto_loader = DataLoader(
            target_train_ds, batch_size=eval_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        test_metrics = _evaluate_open_set(model, test_loader, proto_loader, device)
    else:
        evaluator = Evaluator(num_classes=num_classes, device=device)
        test_metrics = evaluator.evaluate(model, test_loader, use_amp=True)

    results = {
        "experiment": exp_name,
        "model": model_name,
        "adaptation": adaptation_name,
        "source": source_name,
        "target": target_name,
        "strategy": strategy,
        "seed": seed,
        "num_classes": num_classes,
        "train_results": train_results,
        "test_metrics": test_metrics,
    }

    save_results(results, str(results_path))

    del model
    torch.cuda.empty_cache()

    return results


def _evaluate_open_set_cached(
    proto_feats: CachedFeatureDataset,
    tgt_feats: torch.Tensor,
    tgt_labels: torch.Tensor,
) -> Dict[str, Any]:
    """Open-set evaluation using cached features (no model needed).

    Computes prototypes from target train features (one per class), then
    classifies target test samples by nearest-prototype cosine similarity.
    """
    from collections import defaultdict
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        cohen_kappa_score,
    )

    # Build prototypes from target train features (one per target class)
    class_features = defaultdict(list)
    for i in range(len(proto_feats)):
        item = proto_feats[i]
        class_features[item["label"].item()].append(item["feature"])

    prototypes = {}
    for cls, feats in class_features.items():
        prototypes[cls] = torch.stack(feats).mean(dim=0)

    sorted_classes = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[k] for k in sorted_classes])
    proto_matrix = proto_matrix / proto_matrix.norm(dim=1, keepdim=True)

    logger.info(
        "Open-set prototypes: %d classes, ids=%s",
        len(sorted_classes), sorted_classes,
    )

    # Classify target features
    target_features = tgt_feats / tgt_feats.norm(dim=1, keepdim=True)
    similarities = target_features @ proto_matrix.T

    # Map argmax indices back to actual class IDs
    pred_indices = similarities.argmax(dim=1).numpy()
    idx_to_class = np.array(sorted_classes)
    y_pred = idx_to_class[pred_indices]
    y_true = tgt_labels.numpy()

    metrics = {
        "strategy": "open_set",
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mean_similarity": float(similarities.max(dim=1).values.mean()),
        "n_samples": len(y_true),
    }

    logger.info(
        "Open-set eval (cached): acc=%.4f, balanced_acc=%.4f, macro_f1=%.4f",
        metrics["top1_accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    )

    return metrics


@torch.no_grad()
def _evaluate_open_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    train_loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Open-set evaluation using feature-based nearest neighbor.

    Computes full classification metrics (F1, balanced accuracy, etc.)
    in addition to similarity-based metrics.
    """
    from collections import defaultdict
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        cohen_kappa_score, precision_recall_fscore_support,
        confusion_matrix,
    )

    model.eval()
    use_amp = torch.cuda.is_available()

    class_features = defaultdict(list)

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        with torch.cuda.amp.autocast(enabled=use_amp):
            features = model.get_features(images)
        features = features.float().cpu()
        for feat, lbl in zip(features, labels):
            class_features[lbl.item()].append(feat)
        del images, features
        torch.cuda.empty_cache()

    prototypes = {}
    for cls, feats in class_features.items():
        prototypes[cls] = torch.stack(feats).mean(dim=0)

    sorted_classes = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[k] for k in sorted_classes])
    proto_matrix = proto_matrix / proto_matrix.norm(dim=1, keepdim=True)

    logger.info(
        "Open-set prototypes: %d classes, ids=%s",
        len(sorted_classes), sorted_classes,
    )

    all_sims = []
    all_labels = []

    for batch in test_loader:
        images = batch["image"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            features = model.get_features(images)
        features = features.float().cpu()
        features = features / features.norm(dim=1, keepdim=True)

        similarities = features @ proto_matrix.T
        all_sims.append(similarities)
        all_labels.extend(batch["label"].numpy().tolist())
        del images, features
        torch.cuda.empty_cache()

    all_sims = torch.cat(all_sims)
    # Map argmax indices back to actual class IDs
    pred_indices = all_sims.argmax(dim=1).numpy()
    idx_to_class = np.array(sorted_classes)
    y_pred = idx_to_class[pred_indices]
    y_true = np.array(all_labels)

    # Compute full metrics
    metrics = {
        "strategy": "open_set",
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mean_similarity": float(all_sims.max(dim=1).values.mean()),
        "n_samples": len(y_true),
    }

    logger.info(
        "Open-set eval: acc=%.4f, balanced_acc=%.4f, macro_f1=%.4f",
        metrics["top1_accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    )

    return metrics


def main() -> None:
    """Main entry point for Protocol B experiments."""
    parser = argparse.ArgumentParser(
        description="Protocol B: Cross-dataset geographic transfer"
    )
    parser.add_argument("--model", type=str, help="Model config name")
    parser.add_argument("--adaptation", type=str, help="Adaptation strategy")
    parser.add_argument("--source", type=str, help="Source dataset")
    parser.add_argument("--target", type=str, help="Target dataset")
    parser.add_argument(
        "--strategy",
        type=str,
        default="open_set",
        choices=["intersection", "open_set"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", default="./outputs/cross_dataset")
    parser.add_argument("--config", default="configs/base_config.yaml")

    args = parser.parse_args()
    base_config = OmegaConf.load(args.config)

    if args.all:
        exp_config = OmegaConf.load("configs/experiments/cross_dataset.yaml")
        exp = exp_config.experiment

        for pair in exp.transfer_pairs:
            for strategy in exp.strategies:
                for model_name in exp.models:
                    for adaptation_name in exp.adaptations:
                        for seed in base_config.seeds:
                            try:
                                run_single_cross_dataset(
                                    model_name, adaptation_name,
                                    pair.source, pair.target,
                                    strategy, seed, base_config,
                                    args.output_dir,
                                )
                            except Exception as e:
                                logger.error("Failed: %s", e)
    elif args.model and args.adaptation and args.source and args.target:
        run_single_cross_dataset(
            args.model, args.adaptation,
            args.source, args.target,
            args.strategy, args.seed, base_config,
            args.output_dir,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
