"""Re-evaluate Protocol B open-set runs with a corrected shared label space.

Background
----------
``BaseMarineDataset._build_class_mapping`` derives ``class_to_idx`` from the classes
present in a single split. Protocol B builds prototypes from the target *train* split
and scores them against the target *test* split, each carrying its own mapping, so
whenever the two splits hold different class sets the integer labels do not correspond.

    AQUA20       20 train / 20 test classes -> mappings identical, results valid
    Brackish      6 train /  6 test classes -> mappings identical, results valid
    Moorea        8 train /  7 test classes -> 4 of 8 indices misaligned
    Coralscapes  21 train / 15 test classes -> 20 of 21 indices misaligned

The fix is ``experiments.run_cross_dataset.align_label_spaces``. This script re-runs
the evaluation for already-trained runs without retraining: model weights are unchanged
by the bug, only the label bookkeeping at scoring time was wrong.

Linear-probe runs have no checkpoint (they use the cached-feature fast path) and are
re-scored directly from ``outputs/features``. Every other adaptation modifies the
backbone, so its ``best_model.pt`` is loaded and inference is re-run.

Results are written to ``results_fixed.json`` inside each run directory; the original
``results.json`` is left untouched so the two can be compared.

Usage:
    python -m analysis.reeval_protocol_b                      # affected targets only
    python -m analysis.reeval_protocol_b --targets brackish   # control: expect no change
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader

from data.augmentations import get_test_transforms
from experiments.run_cross_dataset import _get_dataset, align_label_spaces
from models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

# Targets whose train/test class sets differ. AQUA20 and Brackish are included only
# so they can be run as controls -- their numbers must not move.
AFFECTED_TARGETS = ["moorea", "coralscapes"]
CONTROL_TARGETS = ["aqua20", "brackish"]

FEATURE_SIZES = {
    "dinov2_base": 224,
    "clip_base": 224,
    "resnet50_imagenet": 224,
    "efficientnet_b4": 380,
}


def _score(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Score predictions over an explicit fixed label set.

    The label set is passed explicitly because sklearn's default macro average uses the
    union of classes appearing in y_true and y_pred, which makes the denominator vary
    between runs and between a model and a trivial baseline.
    """
    return {
        "strategy": "open_set",
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "n_labels_scored": int(len(labels)),
    }


def _prototypes_and_predictions(
    train_feats: torch.Tensor,
    train_labels: np.ndarray,
    test_feats: torch.Tensor,
    ) -> np.ndarray:
    """Nearest-prototype prediction, mirroring the original evaluation."""
    by_class: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for feat, lab in zip(train_feats, train_labels):
        by_class[int(lab)].append(feat)

    classes = sorted(by_class)
    protos = torch.stack([torch.stack(by_class[c]).mean(0) for c in classes])
    protos = protos / protos.norm(dim=1, keepdim=True)
    feats = test_feats / test_feats.norm(dim=1, keepdim=True)
    return np.array(classes)[(feats @ protos.T).argmax(1).numpy()]


@torch.no_grad()
def _extract(model: torch.nn.Module, dataset: Any, batch_size: int,
             num_workers: int, device: str) -> tuple[torch.Tensor, np.ndarray]:
    """Run the backbone over a dataset, returning features and integer labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    use_amp = torch.cuda.is_available()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model.get_features(images)
        feats.append(out.float().cpu())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats), np.array(labels)


def reeval_run(run_dir: Path, data_root: str, device: str, batch_size: int,
               num_workers: int, features_dir: Path) -> Optional[Dict[str, Any]]:
    """Re-evaluate one Protocol B run on a corrected shared label space."""
    original = json.loads((run_dir / "results.json").read_text())
    model_name = original["model"]
    adaptation = original["adaptation"]
    target = original["target"]

    factory = ModelFactory()
    image_size = factory.load_model_config(model_name).image_size
    transform = get_test_transforms(image_size=image_size)

    train_ds = _get_dataset(target, data_root, "train", transform, image_size)
    test_ds = _get_dataset(target, data_root, "test", transform, image_size)
    shared = align_label_spaces(train_ds, test_ds)

    ckpt_path = run_dir / "checkpoints" / "best_model.pt"

    if adaptation == "linear_probe" and not ckpt_path.exists():
        # Frozen backbone: the cached features are exactly those the run used. The
        # cached *labels*, however, were written under the old per-split mapping, so
        # they are recomputed from the aligned datasets instead of being trusted.
        size = FEATURE_SIZES[model_name]
        stem = f"{model_name}_{target}"
        try:
            train_feats = torch.load(features_dir / f"{stem}_train_{size}_features.pt").float()
            test_feats = torch.load(features_dir / f"{stem}_test_{size}_features.pt").float()
        except FileNotFoundError:
            logger.warning("  no cached features for %s/%s -- skipping", model_name, target)
            return None
        train_labels = train_ds.get_labels()
        y_true = test_ds.get_labels()
        source = "cached_features"
    else:
        if not ckpt_path.exists():
            logger.warning("  no checkpoint at %s -- skipping", ckpt_path)
            return None
        model = factory.create(model_name, adaptation, original["num_classes"])
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        train_feats, train_labels = _extract(model, train_ds, batch_size, num_workers, device)
        test_feats, y_true = _extract(model, test_ds, batch_size, num_workers, device)
        del model
        torch.cuda.empty_cache()
        source = "checkpoint_inference"

    y_pred = _prototypes_and_predictions(train_feats, train_labels, test_feats)

    labels_test_present = np.unique(y_true)
    labels_all = np.array(sorted(shared.values()))

    fixed = _score(y_true, y_pred, labels_test_present)
    fixed_all = _score(y_true, y_pred, labels_all)

    return {
        "experiment": original["experiment"],
        "model": model_name,
        "adaptation": adaptation,
        "source": original["source"],
        "target": target,
        "strategy": original["strategy"],
        "seed": original["seed"],
        "label_space": "shared (union of target train and test classes)",
        "n_shared_classes": len(shared),
        "recomputed_from": source,
        "test_metrics": fixed,
        "test_metrics_all_classes": fixed_all,
        "original_test_metrics": original["test_metrics"],
        "macro_f1_original": original["test_metrics"]["macro_f1"],
        "macro_f1_fixed": fixed["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--cross-dataset-dir", default="outputs/cross_dataset")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--targets", nargs="*", default=AFFECTED_TARGETS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--summary", default="outputs/protocol_b_reeval_summary.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("data.base_dataset", "experiments.run_cross_dataset", "models.model_factory"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = yaml.safe_load(Path(args.config).read_text())
    data_root = config["data"]["root_dir"]
    num_workers = config["data"].get("num_workers", 8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_dir = Path(args.cross_dataset_dir)

    summary = []
    for target in args.targets:
        runs = sorted(cross_dir.glob(f"*_deepfish_to_{target}_open_set_seed42"))
        logger.info("\n=== deepfish -> %s (%d runs) ===", target, len(runs))
        for run_dir in runs:
            if not (run_dir / "results.json").exists():
                continue
            result = reeval_run(
                run_dir, data_root, device, args.batch_size, num_workers,
                Path(args.features_dir),
            )
            if result is None:
                continue
            (run_dir / "results_fixed.json").write_text(json.dumps(result, indent=2))
            summary.append(result)
            before, after = result["macro_f1_original"], result["macro_f1_fixed"]
            if abs(after - before) < 1e-4:
                delta = "unchanged"          # within AMP re-inference noise
            elif before == 0.0:
                delta = "0 -> nonzero"
            else:
                delta = f"{(after - before) / before * 100:+.0f}%"
            logger.info(
                "  %-19s %-15s  %.4f -> %.4f  (%s)  [%s]",
                result["model"], result["adaptation"], before, after, delta,
                result["recomputed_from"],
            )

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info("\nWrote %s (%d runs)", args.summary, len(summary))


if __name__ == "__main__":
    main()
