"""Protocol C: Few-shot adaptation curve experiments (KEY EXPERIMENT).

Measures how many labeled target samples are needed when using foundation models.
Runs k-shot experiments with k = {0, 1, 2, 5, 10, 20, 50, 100} across 5 trials.

Two fine-tuning strategies:
1. Combined: Source + k-shot target (train together)
2. Sequential: Source pretrained -> k-shot target fine-tune

Speed optimizations:
- Feature caching: linear_probe extracts backbone features once, trains heads instantly
- k=0 model caching: source-only model trained once, evaluated on all trial query sets
- Adaptive epochs/patience: scales training effort with k (small k = fewer epochs)
- Reduced trials for high k: k>=50 uses 3 trials (low variance), k<50 uses full n_trials

Usage:
    python -m experiments.run_fewshot_curve --model dinov2_base --adaptation lora_r4 \
        --source deepfish --target aqua20 --strategy combined
    python -m experiments.run_fewshot_curve --all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from data.augmentations import get_test_transforms, get_train_transforms
from data.base_dataset import BaseMarineDataset, SubsetMarineDataset
from torch.utils.data import Dataset as _Dataset
from data.deepfish_dataset import DeepFishDataset
from data.aqua20_dataset import AQUA20Dataset
from data.cots_gbr_dataset import COTSGBRDataset
from data.moorea_dataset import MooreaCoralsDataset
from data.coralscapes_dataset import CoralscapesDataset
from data.brackish_dataset import BrackishDataset
from data.fewshot_sampler import FewShotSampler
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

class _LabelOffsetDataset(_Dataset):
    """Wraps a dataset and adds a fixed offset to all labels.

    Used to remap source labels into a non-overlapping range when combining
    source and target datasets with different label spaces.
    """

    def __init__(self, dataset, offset: int):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["label"] = item["label"] + self.offset
        return item


def _adaptive_training_params(k: int, adapt_cfg: Any) -> Dict[str, int]:
    """Scale training epochs and patience based on k-shot count.

    Small k converges fast; no need for 50 epochs with patience 10.

    Returns:
        Dict with 'max_epochs' and 'patience' overrides.
    """
    base_epochs = getattr(adapt_cfg, "epochs", 30)
    base_patience = getattr(adapt_cfg, "early_stopping_patience", 10)

    if k <= 2:
        return {"max_epochs": min(10, base_epochs), "patience": 3}
    elif k <= 5:
        return {"max_epochs": min(15, base_epochs), "patience": 3}
    elif k <= 20:
        return {"max_epochs": min(20, base_epochs), "patience": 5}
    else:
        return {"max_epochs": base_epochs, "patience": base_patience}


def _effective_n_trials(k: int, n_trials: int) -> int:
    """Reduce trial count for high k where variance is low.

    k >= 50: 3 trials (low variance across samples).
    k < 50: full n_trials.
    """
    if k >= 50:
        return min(3, n_trials)
    return n_trials


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


def _load_dataset(
    name: str,
    data_root: str,
    split: str,
    transform: Any,
    image_size: int,
) -> BaseMarineDataset:
    """Load a dataset by name."""
    cls = DATASET_CLASSES[name]
    root = str(Path(data_root) / DATASET_DIRS[name])
    kwargs: Dict[str, Any] = {
        "root_dir": root,
        "split": split,
        "transform": transform,
        "mode": "classification",
        "image_size": image_size,
    }
    return cls(**kwargs)


def run_fewshot_experiment(
    model_name: str,
    adaptation_name: str,
    source_name: str,
    target_name: str,
    finetuning_strategy: str,
    seed: int,
    base_config: Any,
    output_dir: str = "./outputs/fewshot_curve",
) -> Dict[str, Any]:
    """Run few-shot adaptation curve for one model/adaptation combination.

    Speed optimizations applied:
    1. Feature caching: linear_probe uses pre-extracted backbone features
    2. k=0 caching: source-only model trained once, eval on all trial query sets
    3. Adaptive epochs/patience: scaled by k value
    4. Reduced trials: k>=50 uses min(3, n_trials) trials
    """
    set_seed(seed)

    exp_name = (
        f"fewshot_{model_name}_{adaptation_name}_{source_name}_to_{target_name}"
        f"_{finetuning_strategy}_seed{seed}"
    )
    logger.info("=" * 60)
    logger.info("Few-shot experiment: %s", exp_name)
    logger.info("=" * 60)

    factory = ModelFactory()
    model_cfg = factory.load_model_config(model_name)
    adaptation_cfg = factory.load_adaptation_config(adaptation_name)
    image_size = model_cfg.image_size
    device = base_config.hardware.device
    is_linear_probe = adaptation_cfg.type == "linear_probe"

    train_transform = get_train_transforms(image_size=image_size)
    test_transform = get_test_transforms(image_size=image_size)

    data_root = base_config.data.root_dir
    batch_size = base_config.data.batch_size
    num_workers = base_config.data.num_workers
    features_dir = getattr(base_config.project, "features_dir", "./outputs/features")

    # Load source dataset
    source_ds = _load_dataset(
        source_name, data_root, "train", train_transform, image_size
    )

    # Load target dataset (test transform for eval, train transform for support)
    target_ds = _load_dataset(
        target_name, data_root, "test", test_transform, image_size
    )
    target_ds_train = _load_dataset(
        target_name, data_root, "test", train_transform, image_size
    )

    target_labels = target_ds.get_labels()

    # Smoke-test subsampling: use only N images per dataset
    max_samples = getattr(base_config, "smoketest_max_samples", None)
    if max_samples is not None:
        logger.info("SMOKETEST: subsampling datasets to %d samples", max_samples)
        rng_sub = np.random.RandomState(seed)
        src_idx = rng_sub.choice(len(source_ds), min(max_samples, len(source_ds)), replace=False).tolist()
        source_ds = source_ds.get_subset(src_idx)
        tgt_idx = rng_sub.choice(len(target_ds), min(max_samples, len(target_ds)), replace=False).tolist()
        target_ds = target_ds.get_subset(tgt_idx)
        target_ds_train = target_ds_train.get_subset(tgt_idx)
        target_labels = target_ds.get_labels()

    # Label space: target labels occupy 0..target_nc-1,
    # source labels are offset to target_nc..target_nc+source_nc-1
    # so they never collide during combined training.
    target_nc = target_ds.num_classes if hasattr(target_ds, "num_classes") else int(target_labels.max()) + 1
    source_nc = source_ds.num_classes if hasattr(source_ds, "num_classes") else 2
    source_label_offset = target_nc
    num_classes = target_nc + source_nc

    # Few-shot sampler
    k_values = list(base_config.fewshot_k_values)
    n_trials = base_config.fewshot_trials
    sampler = FewShotSampler(k_values=k_values, n_trials=n_trials, base_seed=seed)

    # Source train/val split
    rng = np.random.RandomState(seed)
    n_source = len(source_ds)
    source_indices = rng.permutation(n_source)
    n_val = max(1, int(n_source * base_config.data.val_split))
    source_train_idx = source_indices[n_val:].tolist()
    source_val_idx = source_indices[:n_val].tolist()

    source_train = _LabelOffsetDataset(source_ds.get_subset(source_train_idx), source_label_offset)
    source_val = _LabelOffsetDataset(source_ds.get_subset(source_val_idx), source_label_offset)

    # === OPTIMIZATION 1: Feature caching for linear probe ===
    # Extract backbone features once; all LP training uses cached tensors.
    cached_source_feats = None
    cached_source_labels = None
    cached_target_feats = None
    cached_target_labels = None

    if is_linear_probe:
        logger.info("LINEAR PROBE: extracting and caching backbone features")
        # Create a temporary model just for feature extraction
        feat_model = factory.create(model_name, adaptation_name, num_classes)

        # Source features (with test transform for consistency)
        source_ds_eval = _load_dataset(
            source_name, data_root, "train", test_transform, image_size
        )
        if max_samples is not None:
            source_ds_eval = source_ds_eval.get_subset(
                rng_sub.choice(len(source_ds_eval), min(max_samples, len(source_ds_eval)), replace=False).tolist()
            )
        smoke_tag = f"_smoke{max_samples}" if max_samples is not None else ""
        cached_source_feats, cached_source_labels = extract_and_cache_features(
            model=feat_model,
            dataset=source_ds_eval,
            model_name=model_name,
            dataset_name=source_name,
            split=f"train{smoke_tag}",
            image_size=image_size,
            cache_dir=features_dir,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Target features
        cached_target_feats, cached_target_labels = extract_and_cache_features(
            model=feat_model,
            dataset=target_ds,
            model_name=model_name,
            dataset_name=target_name,
            split=f"test{smoke_tag}",
            image_size=image_size,
            cache_dir=features_dir,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        feature_dim = cached_source_feats.shape[1]

        # Offset source labels for combined training
        cached_source_labels_offset = cached_source_labels + source_label_offset

        # Source train/val split on cached features
        src_train_feats = CachedFeatureDataset(
            cached_source_feats, cached_source_labels_offset,
            indices=np.array(source_train_idx),
        )
        src_val_feats = CachedFeatureDataset(
            cached_source_feats, cached_source_labels_offset,
            indices=np.array(source_val_idx),
        )

        del feat_model
        torch.cuda.empty_cache()
        logger.info("Feature extraction complete: source=%s, target=%s",
                     cached_source_feats.shape, cached_target_feats.shape)

    # === OPTIMIZATION 2: k=0 source model caching ===
    # For k=0 non-CLIP: train once on source, eval on each trial's query set.
    k0_cached_model = None

    curve_results: Dict[int, List[Dict[str, Any]]] = {}

    for k in k_values:
        curve_results[k] = []
        # OPTIMIZATION 4: Reduce trials for high k
        eff_trials = _effective_n_trials(k, n_trials)
        trials = sampler.sample_all_trials(target_labels, k)

        for trial_idx, (support_indices, query_indices) in enumerate(trials):
            if trial_idx >= eff_trials:
                break

            trial_name = f"k{k}_trial{trial_idx}"
            logger.info("Running %s: k=%d, trial=%d/%d", exp_name, k, trial_idx, eff_trials)

            set_seed(seed + trial_idx * 100 + k)

            if k == 0:
                if model_cfg.family == "clip":
                    model = factory.create(model_name, adaptation_name, num_classes)
                    trial_metrics = _clip_zero_shot_eval(
                        model, target_ds, query_indices, device,
                        model_cfg, batch_size, num_workers,
                    )
                    del model
                    torch.cuda.empty_cache()
                elif is_linear_probe:
                    # LP k=0: train linear head on cached source features once
                    if k0_cached_model is None:
                        logger.info("k=0 LP: training linear head on cached source features")
                        lp_trainer = LinearProbeTrainer(
                            feature_dim=feature_dim,
                            num_classes=num_classes,
                            lr=getattr(adaptation_cfg, "learning_rate", 1e-3),
                            epochs=getattr(adaptation_cfg, "epochs", 50),
                            patience=getattr(adaptation_cfg, "early_stopping_patience", 10),
                            device=device,
                        )
                        lp_trainer.train(src_train_feats, src_val_feats)
                        k0_cached_model = lp_trainer
                    # Eval on this trial's query set (restrict to target classes)
                    query_feats = CachedFeatureDataset(
                        cached_target_feats, cached_target_labels,
                        indices=query_indices,
                    )
                    trial_metrics = k0_cached_model.evaluate(
                        query_feats, num_eval_classes=target_nc,
                    )
                else:
                    # Non-LP, non-CLIP k=0: train once, cache model, eval all trials
                    if k0_cached_model is None:
                        logger.info("k=0: training source model (will reuse for all trials)")
                        model = factory.create(model_name, adaptation_name, num_classes)
                        _train_and_eval(
                            model=model,
                            train_ds=source_train,
                            val_ds=source_val,
                            test_ds=None,
                            adaptation_cfg=adaptation_cfg,
                            config=base_config,
                            output_dir=output_dir,
                            exp_name=f"{exp_name}/k0_source_model",
                            device=device,
                            batch_size=batch_size,
                            num_workers=num_workers,
                        )
                        k0_cached_model = model
                    # Eval on this trial's query set
                    query_set = target_ds.get_subset(query_indices.tolist())
                    test_loader = DataLoader(
                        query_set, batch_size=max(batch_size // 2, 1),
                        shuffle=False, num_workers=num_workers, pin_memory=True,
                    )
                    evaluator = Evaluator(num_classes=target_nc, device=device)
                    trial_metrics = evaluator.evaluate(
                        k0_cached_model, test_loader, save_predictions=False,
                        use_amp=True, num_eval_classes=target_nc,
                    )
            elif is_linear_probe:
                # LP k>0: train linear head on cached source+support features
                # OPTIMIZATION 3: adaptive epochs/patience
                tp = _adaptive_training_params(k, adaptation_cfg)
                support_feats = CachedFeatureDataset(
                    cached_target_feats, cached_target_labels,
                    indices=support_indices,
                )
                query_feats = CachedFeatureDataset(
                    cached_target_feats, cached_target_labels,
                    indices=query_indices,
                )

                if finetuning_strategy == "combined":
                    # Combine source + support features
                    combined_feats = torch.cat([src_train_feats.features, support_feats.features], dim=0)
                    combined_labels = torch.cat([src_train_feats.labels, support_feats.labels], dim=0)
                    combined_ds = CachedFeatureDataset(combined_feats, combined_labels)

                    lp = LinearProbeTrainer(
                        feature_dim=feature_dim,
                        num_classes=num_classes,
                        lr=getattr(adaptation_cfg, "learning_rate", 1e-3),
                        epochs=tp["max_epochs"],
                        patience=tp["patience"],
                        device=device,
                    )
                    lp.train(combined_ds, src_val_feats)
                    trial_metrics = lp.evaluate(
                        query_feats, num_eval_classes=target_nc,
                    )
                else:
                    # Sequential: train on source, then on support
                    lp = LinearProbeTrainer(
                        feature_dim=feature_dim,
                        num_classes=num_classes,
                        lr=getattr(adaptation_cfg, "learning_rate", 1e-3),
                        epochs=getattr(adaptation_cfg, "epochs", 50),
                        patience=getattr(adaptation_cfg, "early_stopping_patience", 10),
                        device=device,
                    )
                    lp.train(src_train_feats, src_val_feats)
                    # Fine-tune on support
                    lp2 = LinearProbeTrainer(
                        feature_dim=feature_dim,
                        num_classes=num_classes,
                        lr=getattr(adaptation_cfg, "learning_rate", 1e-3),
                        epochs=tp["max_epochs"],
                        patience=tp["patience"],
                        device=device,
                    )
                    lp2.head.load_state_dict(lp.head.state_dict())
                    lp2.train(support_feats, support_feats)
                    trial_metrics = lp2.evaluate(
                        query_feats, num_eval_classes=target_nc,
                    )
            else:
                # Non-LP: standard training with adaptive epochs/patience
                model = factory.create(model_name, adaptation_name, num_classes)
                support_set = target_ds_train.get_subset(support_indices.tolist())
                query_set = target_ds.get_subset(query_indices.tolist())

                # OPTIMIZATION 3: adaptive epochs/patience
                tp = _adaptive_training_params(k, adaptation_cfg)

                if finetuning_strategy == "combined":
                    combined_train = ConcatDataset([source_train, support_set])
                    trial_metrics = _train_and_eval(
                        model=model,
                        train_ds=combined_train,
                        val_ds=source_val,
                        test_ds=query_set,
                        adaptation_cfg=adaptation_cfg,
                        config=base_config,
                        output_dir=output_dir,
                        exp_name=f"{exp_name}/{trial_name}",
                        device=device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        max_epochs_override=tp["max_epochs"],
                        patience_override=tp["patience"],
                        num_eval_classes=target_nc,
                    )
                elif finetuning_strategy == "sequential":
                    _train_and_eval(
                        model=model,
                        train_ds=source_train,
                        val_ds=source_val,
                        test_ds=None,
                        adaptation_cfg=adaptation_cfg,
                        config=base_config,
                        output_dir=output_dir,
                        exp_name=f"{exp_name}/{trial_name}_source",
                        device=device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                    trial_metrics = _train_and_eval(
                        model=model,
                        train_ds=support_set,
                        val_ds=support_set,
                        test_ds=query_set,
                        adaptation_cfg=adaptation_cfg,
                        config=base_config,
                        output_dir=output_dir,
                        exp_name=f"{exp_name}/{trial_name}_target",
                        device=device,
                        batch_size=min(batch_size, len(support_set)),
                        num_workers=num_workers,
                        max_epochs_override=tp["max_epochs"],
                        patience_override=tp["patience"],
                        num_eval_classes=target_nc,
                    )
                else:
                    raise ValueError(f"Unknown strategy: {finetuning_strategy}")

                del model
                torch.cuda.empty_cache()

            trial_metrics["k"] = k
            trial_metrics["trial"] = trial_idx
            trial_metrics["n_support"] = len(support_indices)
            trial_metrics["n_query"] = len(query_indices)
            curve_results[k].append(trial_metrics)

    # Clean up k=0 cached model
    if k0_cached_model is not None and not isinstance(k0_cached_model, LinearProbeTrainer):
        del k0_cached_model
        torch.cuda.empty_cache()

    aggregated = _aggregate_curve_results(curve_results)

    results = {
        "experiment": exp_name,
        "model": model_name,
        "adaptation": adaptation_name,
        "source": source_name,
        "target": target_name,
        "strategy": finetuning_strategy,
        "seed": seed,
        "k_values": k_values,
        "n_trials": n_trials,
        "raw_results": {str(k): v for k, v in curve_results.items()},
        "aggregated": aggregated,
    }

    results_path = Path(output_dir) / exp_name / "fewshot_curve.json"
    save_results(results, str(results_path))

    return results


def _train_and_eval(
    model: torch.nn.Module,
    train_ds: Any,
    val_ds: Any,
    test_ds: Optional[Any],
    adaptation_cfg: Any,
    config: Any,
    output_dir: str,
    exp_name: str,
    device: str,
    batch_size: int,
    num_workers: int,
    max_epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
    num_eval_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a model and optionally evaluate.

    Supports crash recovery: if best_model.pt exists but results.json doesn't,
    skips training and jumps to evaluation.

    Args:
        max_epochs_override: If set, caps training epochs (adaptive by k).
        patience_override: If set, overrides early stopping patience.
    """
    import json

    exp_dir = Path(output_dir) / exp_name
    best_ckpt = exp_dir / "checkpoints" / "best_model.pt"
    results_file = exp_dir / "results.json"
    eval_batch_size = max(batch_size // 2, 1)

    # Skip if results already exist
    if results_file.exists():
        logger.info("SKIP (results exist): %s", exp_name)
        with open(results_file) as f:
            return json.load(f)

    # Crash recovery: checkpoint exists but no results
    crashed = best_ckpt.exists() and not results_file.exists()

    if crashed:
        logger.info("RESUME (crash recovery — eval only): %s", exp_name)
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        del checkpoint
        torch.cuda.empty_cache()
    else:
        effective_batch = min(batch_size, len(train_ds))
        if effective_batch < 1:
            effective_batch = 1

        train_loader = DataLoader(
            train_ds, batch_size=effective_batch, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=len(train_ds) > effective_batch,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            adaptation_config=adaptation_cfg,
            output_dir=output_dir,
            experiment_name=exp_name,
            device=device,
        )

        # Apply adaptive epoch/patience overrides
        if max_epochs_override is not None:
            trainer.epochs = min(trainer.epochs, max_epochs_override)
        if patience_override is not None:
            from training.early_stopping import EarlyStopping
            trainer.early_stopping = EarlyStopping(patience=patience_override, mode="min")

        trainer.train()
        trainer.load_best_model()
        model = trainer.model

        # Free trainer memory before eval
        del trainer.optimizer, trainer.scheduler, trainer.scaler
        del trainer
        torch.cuda.empty_cache()

    if test_ds is None:
        return {}

    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    eval_nc = num_eval_classes if num_eval_classes is not None else (
        model.num_classes if hasattr(model, "num_classes") else model.head.num_classes
    )
    evaluator = Evaluator(num_classes=eval_nc, device=device)
    metrics = evaluator.evaluate(
        model, test_loader, save_predictions=False, use_amp=True,
        num_eval_classes=num_eval_classes,
    )

    # Save results for crash recovery
    save_results(metrics, str(results_file))

    return metrics


@torch.no_grad()
def _clip_zero_shot_eval(
    model: torch.nn.Module,
    target_ds: BaseMarineDataset,
    query_indices: np.ndarray,
    device: str,
    model_cfg: Any,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    """CLIP zero-shot evaluation using text prompts."""
    model = model.to(device)
    model.eval()

    query_set = target_ds.get_subset(query_indices.tolist())
    loader = DataLoader(
        query_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = target_ds.class_names
    templates = getattr(model_cfg, "zero_shot_templates", None)

    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        logits = model.clip_zero_shot(images, class_names, templates)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "method": "clip_zero_shot",
    }


def _aggregate_curve_results(
    curve_results: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate few-shot curve results across trials."""
    aggregated: Dict[str, Any] = {}

    for k, trials in curve_results.items():
        metrics_keys = ["top1_accuracy", "macro_f1", "balanced_accuracy"]
        k_agg: Dict[str, Any] = {"k": k, "n_trials": len(trials)}

        for metric in metrics_keys:
            values = [t.get(metric, 0.0) for t in trials if metric in t]
            if values:
                k_agg[f"{metric}_mean"] = float(np.mean(values))
                k_agg[f"{metric}_std"] = float(np.std(values))

        aggregated[str(k)] = k_agg

    return aggregated


def main() -> None:
    """Main entry point for Protocol C experiments."""
    parser = argparse.ArgumentParser(
        description="Protocol C: Few-shot adaptation curve"
    )
    parser.add_argument("--model", type=str, help="Model config name")
    parser.add_argument("--adaptation", type=str, help="Adaptation strategy")
    parser.add_argument("--source", type=str, help="Source dataset")
    parser.add_argument("--target", type=str, help="Target dataset")
    parser.add_argument(
        "--strategy", type=str, default="combined",
        choices=["combined", "sequential"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", default="./outputs/fewshot_curve")
    parser.add_argument("--config", default="configs/base_config.yaml")

    args = parser.parse_args()
    base_config = OmegaConf.load(args.config)

    if args.all:
        exp_config = OmegaConf.load("configs/experiments/fewshot_curve.yaml")
        exp = exp_config.experiment

        for pair in exp.transfer_pairs:
            for ft_strategy in exp.finetuning_strategies:
                for model_name in exp.models:
                    for adaptation_name in exp.adaptations:
                        for seed in base_config.seeds:
                            try:
                                run_fewshot_experiment(
                                    model_name, adaptation_name,
                                    pair.source, pair.target,
                                    ft_strategy, seed, base_config,
                                    args.output_dir,
                                )
                            except Exception as e:
                                logger.error("Failed: %s", e)
    elif args.model and args.adaptation and args.source and args.target:
        run_fewshot_experiment(
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
