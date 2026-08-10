"""Trivial-classifier baselines for Protocol B (cross-dataset transfer).

Reports uniform-random and majority-class baselines for every Protocol B transfer, so
that nearest-prototype results can be judged against what a trivial classifier would
achieve on the same label set.

The metric computation mirrors ``_evaluate_open_set_cached`` in
``experiments/run_cross_dataset.py``: the same sklearn call with ``average="macro"``
and ``zero_division=0``, on the same target test split.

IMPORTANT -- macro-F1 denominator. ``f1_score(average="macro")`` without an explicit
``labels`` argument averages over the *union of classes appearing in y_true and
y_pred*. A constant (majority-class) predictor emits one class, so it is averaged over
the classes present in the test split; a model that predicts classes present in the
target train split but absent from test is averaged over a strictly larger set. On
DeepFish -> Coralscapes that is 15 labels versus 21 -- comparing the two directly
penalises the model by ~40%. Every figure here is therefore computed with an explicit
fixed ``labels`` set, reported under both conventions:

    "test-present"  : classes present in the target test split
    "all-prototype" : every class with a prototype (i.e. present in target train)

The same issue affects the *reported* Protocol B results: because the original
evaluation used sklearn's default label set, the denominator varies from run to run,
so the published per-run macro-F1 values are not strictly comparable with one another.
Re-evaluating all runs with a fixed label set is recommended -- see ``--recompute``,
which does this exactly for the four linear-probe configurations reproducible from
cached features.

Usage:
    python -m analysis.protocol_b_baselines
    python -m analysis.protocol_b_baselines --output outputs/protocol_b_baselines.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from experiments.run_cross_dataset import _get_dataset

logger = logging.getLogger(__name__)

TARGETS = ["aqua20", "moorea", "brackish", "coralscapes"]

# Number of Monte Carlo draws for the uniform-random baseline.
N_RANDOM_DRAWS = 1000
RANDOM_SEED = 42


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None
) -> Dict[str, float]:
    """Score a prediction vector, optionally over an explicit fixed label set."""
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
    }


def _analytic_uniform_macro_f1(y_true: np.ndarray, pred_classes: np.ndarray) -> float:
    """Closed-form expected macro F1 for a uniform-random classifier.

    For a predictor independent of the truth that draws uniformly over K classes,
    class c has precision p_c (its prevalence) and recall 1/K, giving
    F1_c = 2 p_c (1/K) / (p_c + 1/K). Macro F1 averages over the label set sklearn
    would use, i.e. the union of classes present in y_true and in the predictions.
    """
    k = len(pred_classes)
    label_set = np.union1d(np.unique(y_true), pred_classes)
    n = len(y_true)

    f1s = []
    for c in label_set:
        p_c = float((y_true == c).sum()) / n
        recall = 1.0 / k if c in pred_classes else 0.0
        precision = p_c if c in pred_classes else 0.0
        f1s.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def compute_for_target(target: str, root_dir: str) -> Dict[str, Any]:
    """Compute both trivial baselines for one DeepFish -> target transfer."""
    test_ds = _get_dataset(target, root_dir, split="test", transform=None)
    train_ds = _get_dataset(target, root_dir, split="train", transform=None)

    y_true = test_ds.get_labels()

    # The prototype classifier can only ever emit a class that has a prototype,
    # i.e. a class present in the target train split. Trivial baselines are given
    # exactly the same prediction space so the comparison is fair.
    pred_classes = np.unique(train_ds.get_labels())

    # Both label-set conventions (see the module docstring).
    label_sets = {
        "test_present": np.unique(y_true),
        "all_prototype": pred_classes,
    }

    train_labels = train_ds.get_labels()
    majority_train = int(np.bincount(train_labels).argmax())
    majority_test = int(np.bincount(y_true).argmax())

    majority: Dict[str, Any] = {}
    majority_oracle: Dict[str, Any] = {}
    random_mean: Dict[str, Any] = {}
    random_sd: Dict[str, Any] = {}

    for conv, labels in label_sets.items():
        majority[conv] = _metrics(y_true, np.full_like(y_true, majority_train), labels)
        majority_oracle[conv] = _metrics(
            y_true, np.full_like(y_true, majority_test), labels
        )

        rng = np.random.default_rng(RANDOM_SEED)
        draws = [
            _metrics(y_true, rng.choice(pred_classes, size=len(y_true)), labels)
            for _ in range(N_RANDOM_DRAWS)
        ]
        random_mean[conv] = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}
        random_sd[conv] = {k: float(np.std([d[k] for d in draws])) for k in draws[0]}

    # Class distribution, for the prevalence discussion in the response letter.
    counts = np.bincount(y_true, minlength=int(max(pred_classes)) + 1)
    prevalences = counts / counts.sum()

    return {
        "target": target,
        "n_test_samples": int(len(y_true)),
        "n_classes_test": int(len(np.unique(y_true))),
        "n_classes_prototype": int(len(pred_classes)),
        "majority_class_prevalence": float(prevalences.max()),
        "class_counts": {
            test_ds.idx_to_class[i]: int(c) for i, c in enumerate(counts) if c > 0
        },
        "baseline_majority_class": majority,
        "baseline_majority_class_oracle": majority_oracle,
        "baseline_uniform_random_mean": random_mean,
        "baseline_uniform_random_sd": random_sd,
        "baseline_uniform_random_analytic_macro_f1": _analytic_uniform_macro_f1(
            y_true, pred_classes
        ),
    }


def recompute_linear_probes(target: str, root_dir: str, features_dir: Path) -> Dict[str, Any]:
    """Re-score the linear-probe Protocol B runs on a fixed label set.

    Only the linear-probe configurations can be reproduced from cached features: the
    backbone is frozen, so the cached features are exactly those the original run used.
    LoRA, VPT and full fine-tuning modify the backbone and would need re-inference from
    the saved checkpoints.
    """
    import torch

    sizes = {
        "dinov2_base": 224,
        "clip_base": 224,
        "resnet50_imagenet": 224,
        "efficientnet_b4": 380,
    }
    out: Dict[str, Any] = {}
    y_true = None
    for model, size in sizes.items():
        stem = f"{model}_{target}"
        try:
            tr_f = torch.load(features_dir / f"{stem}_train_{size}_features.pt").float()
            tr_l = torch.load(features_dir / f"{stem}_train_{size}_labels.pt")
            te_f = torch.load(features_dir / f"{stem}_test_{size}_features.pt").float()
            te_l = torch.load(features_dir / f"{stem}_test_{size}_labels.pt")
        except FileNotFoundError:
            continue

        from collections import defaultdict

        class_feats = defaultdict(list)
        for feat, lab in zip(tr_f, tr_l):
            class_feats[int(lab)].append(feat)
        classes = sorted(class_feats)
        protos = torch.stack([torch.stack(class_feats[c]).mean(0) for c in classes])
        protos = protos / protos.norm(dim=1, keepdim=True)
        feats = te_f / te_f.norm(dim=1, keepdim=True)
        y_pred = np.array(classes)[(feats @ protos.T).argmax(1).numpy()]
        y_true = te_l.numpy()

        out[model] = {
            "macro_f1_default_labels": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "macro_f1_test_present": float(
                f1_score(
                    y_true, y_pred, average="macro",
                    labels=np.unique(y_true), zero_division=0,
                )
            ),
            "macro_f1_all_prototype": float(
                f1_score(
                    y_true, y_pred, average="macro",
                    labels=np.array(classes), zero_division=0,
                )
            ),
            "n_classes_predicted": int(len(np.unique(y_pred))),
        }
    return out


def load_model_results(output_dir: Path, target: str) -> List[Dict[str, Any]]:
    """Collect the reported Protocol B macro F1 values for one target."""
    rows = []
    for run in sorted(output_dir.glob(f"*_deepfish_to_{target}_open_set_seed42")):
        results_file = run / "results.json"
        if not results_file.exists():
            continue
        d = json.loads(results_file.read_text())
        rows.append(
            {
                "model": d["model"],
                "adaptation": d["adaptation"],
                "macro_f1": d["test_metrics"]["macro_f1"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--cross-dataset-dir", default="outputs/cross_dataset")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--output", default="outputs/protocol_b_baselines.json")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-score the linear-probe runs on a fixed label set from cached features.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(Path(args.config).read_text())
    root_dir = config["data"]["root_dir"]
    cross_dir = Path(args.cross_dataset_dir)

    results = {}
    for target in TARGETS:
        logger.info("=== deepfish -> %s ===", target)
        res = compute_for_target(target, root_dir)
        model_rows = load_model_results(cross_dir, target)
        res["model_macro_f1"] = model_rows
        if model_rows:
            f1s = [r["macro_f1"] for r in model_rows]
            res["model_macro_f1_best"] = max(f1s)
            res["model_macro_f1_mean"] = float(np.mean(f1s))
        if args.recompute:
            res["linear_probe_fixed_labels"] = recompute_linear_probes(
                target, root_dir, Path(args.features_dir)
            )
        results[target] = res

        logger.info(
            "  n=%d, %d classes present in test (prototypes: %d), majority prevalence=%.3f",
            res["n_test_samples"],
            res["n_classes_test"],
            res["n_classes_prototype"],
            res["majority_class_prevalence"],
        )
        for conv in ("test_present", "all_prototype"):
            line = (
                f"  [{conv:<13}] macro F1 -- random: "
                f"{res['baseline_uniform_random_mean'][conv]['macro_f1']:.4f} "
                f"+/- {res['baseline_uniform_random_sd'][conv]['macro_f1']:.4f} | "
                f"majority: {res['baseline_majority_class'][conv]['macro_f1']:.4f}"
            )
            lp = res.get("linear_probe_fixed_labels")
            if lp:
                key = f"macro_f1_{conv}"
                vals = [v[key] for v in lp.values()]
                line += f" | LP mean: {np.mean(vals):.4f} | LP best: {max(vals):.4f}"
            logger.info(line)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
