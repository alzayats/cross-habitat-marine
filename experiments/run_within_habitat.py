"""Protocol A: Within-dataset cross-habitat transfer experiments.

DeepFish has 20 habitats. This protocol evaluates:
1. Leave-one-habitat-out: train on 19, test on 1 (20 experiments)
2. Multi-source: train on 15, test on 5 (5 random configurations)

Usage:
    python -m experiments.run_within_habitat --model dinov2_base --adaptation lora_r4
    python -m experiments.run_within_habitat --all  # Run all combinations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

from data.augmentations import get_test_transforms, get_train_transforms
from data.deepfish_dataset import DeepFishDataset
from models.model_factory import ModelFactory
from training.evaluator import Evaluator
from training.trainer import Trainer
from utils.io_utils import save_results
from utils.seed_utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    model_name: str,
    adaptation_name: str,
    source_habitats: List[str],
    target_habitat: str,
    seed: int,
    base_config: Any,
    output_dir: str = "./outputs/within_habitat",
) -> Dict[str, Any]:
    """Run a single within-habitat transfer experiment."""
    set_seed(seed)

    source_str = "+".join(source_habitats[:3])
    if len(source_habitats) > 3:
        source_str += f"+{len(source_habitats)-3}more"
    exp_name = f"{model_name}_{adaptation_name}_{source_str}_to_{target_habitat}_seed{seed}"

    # Skip if results already exist
    exp_dir = Path(output_dir) / exp_name
    results_path = exp_dir / "results.json"
    if results_path.exists():
        logger.info("SKIP (results exist): %s", exp_name)
        with open(results_path) as f:
            return json.load(f)

    # Check for crashed run: training done (checkpoint exists) but eval failed
    best_ckpt_path = exp_dir / "checkpoints" / "best_model.pt"
    history_path = exp_dir / "training_history.json"
    crashed = best_ckpt_path.exists() and not results_path.exists()

    if crashed:
        logger.info("=" * 60)
        logger.info("RESUME (crash recovery — eval only): %s", exp_name)
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Experiment: %s", exp_name)
        logger.info("=" * 60)

    factory = ModelFactory()
    model_cfg = factory.load_model_config(model_name)
    image_size = model_cfg.image_size

    train_transform = get_train_transforms(image_size=image_size)
    test_transform = get_test_transforms(image_size=image_size)

    data_root = base_config.data.root_dir + "/DeepFish"
    train_ds, val_ds, test_ds = DeepFishDataset.get_habitat_split(
        root_dir=data_root,
        source_habitats=source_habitats,
        target_habitat=target_habitat,
        train_transform=train_transform,
        test_transform=test_transform,
        val_split=base_config.data.val_split,
        seed=seed,
    )

    num_classes = train_ds.dataset.num_classes if hasattr(train_ds, "dataset") else train_ds.num_classes

    batch_size = base_config.data.batch_size
    num_workers = base_config.data.num_workers
    # Use half batch size for eval to reduce GPU memory pressure
    eval_batch_size = max(batch_size // 2, 1)

    device = base_config.hardware.device

    if crashed:
        # --- Crash recovery: load best model directly, skip training ---
        logger.info("Loading best checkpoint from %s", best_ckpt_path)
        model = factory.create(model_name, adaptation_name, num_classes)
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Load training results from checkpoint + history file
        train_results = {
            "best_epoch": checkpoint.get("best_epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", 0),
            "total_epochs": checkpoint.get("epoch", 0) + 1,
            "total_time_seconds": 0,  # unknown for recovered runs
            "history": checkpoint.get("history", []),
        }
        # Prefer training_history.json if it exists (more complete)
        if history_path.exists():
            with open(history_path) as f:
                train_results["history"] = json.load(f)

        # Free checkpoint memory
        del checkpoint
        torch.cuda.empty_cache()
    else:
        # --- Normal path: full training ---
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
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

        # Free trainer memory (optimizer, scheduler, scaler) before eval
        del trainer.optimizer, trainer.scheduler, trainer.scaler
        del trainer
        torch.cuda.empty_cache()

    # --- Evaluation (shared path for normal + crash recovery) ---
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = test_ds.class_names if hasattr(test_ds, "class_names") else None
    evaluator = Evaluator(
        num_classes=num_classes,
        class_names=class_names,
        device=device,
    )
    test_metrics = evaluator.evaluate(model, test_loader, use_amp=True)

    results = {
        "experiment": exp_name,
        "model": model_name,
        "adaptation": adaptation_name,
        "source_habitats": source_habitats,
        "target_habitat": target_habitat,
        "seed": seed,
        "num_classes": num_classes,
        "train_results": train_results,
        "test_metrics": test_metrics,
        "param_counts": model.param_counts if hasattr(model, "param_counts") else {},
    }

    save_results(results, str(results_path))

    logger.info(
        "Results: acc=%.4f, f1=%.4f, balanced_acc=%.4f",
        test_metrics["top1_accuracy"],
        test_metrics["macro_f1"],
        test_metrics["balanced_accuracy"],
    )

    del model
    torch.cuda.empty_cache()

    return results


def run_all_experiments(
    base_config: Any,
    models: Optional[List[str]] = None,
    adaptations: Optional[List[str]] = None,
    output_dir: str = "./outputs/within_habitat",
    seed_override: Optional[int] = None,
    leave_one_out_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run all within-habitat experiment combinations."""
    exp_config = OmegaConf.load("configs/experiments/within_habitat.yaml")
    exp = exp_config.experiment

    if models is None:
        models = list(exp.models)
    if adaptations is None:
        adaptations = list(exp.adaptations)

    if seed_override is not None:
        seeds = [seed_override]
    else:
        seeds = list(base_config.seeds)
    data_root = base_config.data.root_dir + "/DeepFish"

    # Get all habitat IDs
    all_habitats = DeepFishDataset.get_all_habitats(data_root)
    logger.info("Found %d habitats: %s", len(all_habitats), all_habitats)

    # Build experiment splits
    habitat_splits = []

    # Leave-one-out splits (train on N-1, test on 1)
    if exp.get("leave_one_out", True):
        for target in all_habitats:
            sources = [h for h in all_habitats if h != target]
            habitat_splits.append((sources, target))

    # Multi-source random splits (skip if leave_one_out_only)
    multi_cfg = exp.get("multi_source", {})
    if not leave_one_out_only and multi_cfg.get("enabled", False):
        multi_splits = DeepFishDataset.get_multi_source_splits(
            habitats=all_habitats,
            n_train=multi_cfg.get("n_train", 15),
            n_test=multi_cfg.get("n_test", 5),
            n_configs=multi_cfg.get("n_configs", 5),
            seed=multi_cfg.get("seed", 42),
        )
        # For multi-source, test on each of the test habitats separately
        for train_habs, test_habs in multi_splits:
            for target in test_habs:
                habitat_splits.append((train_habs, target))

    all_results: List[Dict[str, Any]] = []

    total = len(models) * len(adaptations) * len(habitat_splits) * len(seeds)
    logger.info("Total experiments to run: %d", total)

    for model_name in models:
        for adaptation_name in adaptations:
            for source_habitats, target_habitat in habitat_splits:
                for seed in seeds:
                    try:
                        result = run_single_experiment(
                            model_name=model_name,
                            adaptation_name=adaptation_name,
                            source_habitats=source_habitats,
                            target_habitat=target_habitat,
                            seed=seed,
                            base_config=base_config,
                            output_dir=output_dir,
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            "Experiment failed: %s + %s, %s -> %s, seed=%d: %s",
                            model_name,
                            adaptation_name,
                            source_habitats[:3],
                            target_habitat,
                            seed,
                            e,
                        )

    agg_path = Path(output_dir) / "all_results.json"
    save_results(all_results, str(agg_path))
    logger.info("All results saved to %s", agg_path)

    return all_results


def main() -> None:
    """Main entry point for Protocol A experiments."""
    parser = argparse.ArgumentParser(
        description="Protocol A: Within-dataset cross-habitat transfer (20 habitats)"
    )
    parser.add_argument("--model", type=str, help="Model config name")
    parser.add_argument("--adaptation", type=str, help="Adaptation config name")
    parser.add_argument("--source", type=str, nargs="+", help="Source habitat ID(s)")
    parser.add_argument("--target", type=str, help="Target habitat ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--all", action="store_true", help="Run all combinations")
    parser.add_argument(
        "--leave-one-out-only", action="store_true",
        help="Skip multi-source splits, only run leave-one-out",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/within_habitat"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from base config")

    args = parser.parse_args()

    base_config = OmegaConf.load(args.config)
    if args.batch_size is not None:
        base_config.data.batch_size = args.batch_size

    if args.all:
        run_all_experiments(
            base_config,
            models=[args.model] if args.model else None,
            adaptations=[args.adaptation] if args.adaptation else None,
            output_dir=args.output_dir,
            seed_override=args.seed,
            leave_one_out_only=args.leave_one_out_only,
        )
    elif args.model and args.adaptation and args.source and args.target:
        run_single_experiment(
            model_name=args.model,
            adaptation_name=args.adaptation,
            source_habitats=args.source,
            target_habitat=args.target,
            seed=args.seed,
            base_config=base_config,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
