"""Shared data loading utilities for all paper figures.

Loads all results.json files from within_habitat experiments into a tidy
pandas DataFrame suitable for plotting.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_results_df(
    results_dir: str = "./outputs/within_habitat",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Load all results.json files into a tidy DataFrame.

    Each row represents one experiment run (model × adaptation × habitat × seed).

    Args:
        results_dir: Directory containing experiment subdirectories.
        seed: If set, filter to this seed only (default: 42 to match paper).
              Set to None to include all seeds.

    Returns:
        DataFrame with columns: model, adaptation, target_habitat, seed,
        macro_f1, balanced_accuracy, cohens_kappa, empty_f1, valid_f1,
        n_samples, trainable_params, total_params, trainable_pct,
        training_time, best_epoch, total_epochs, history (list of dicts),
        model_display, adapt_display, combo.
    """
    from figures.plot_utils import MODEL_DISPLAY_NAMES, ADAPTATION_DISPLAY_NAMES

    rows = []
    results_path = Path(results_dir)

    for path in sorted(results_path.rglob("results.json")):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            logger.warning("Failed to load %s", path)
            continue

        test = r.get("test_metrics", {})
        params = r.get("param_counts", {})
        train = r.get("train_results", {})
        per_class = test.get("per_class", {})

        rows.append({
            "model": r.get("model", ""),
            "adaptation": r.get("adaptation", ""),
            "target_habitat": str(r.get("target_habitat", "")),
            "seed": r.get("seed", 42),
            "macro_f1": test.get("macro_f1", None),
            "balanced_accuracy": test.get("balanced_accuracy", None),
            "cohens_kappa": test.get("cohens_kappa", None),
            "empty_f1": per_class.get("empty", {}).get("f1", None),
            "valid_f1": per_class.get("valid", {}).get("f1", None),
            "empty_support": per_class.get("empty", {}).get("support", 0),
            "valid_support": per_class.get("valid", {}).get("support", 0),
            "n_samples": test.get("n_samples", 0),
            "trainable_params": params.get("trainable", None),
            "total_params": params.get("total", None),
            "trainable_pct": params.get("trainable_pct", None),
            "training_time": train.get("total_time_seconds", None),
            "best_epoch": train.get("best_epoch", None),
            "total_epochs": train.get("total_epochs", None),
            "history": train.get("history", []),
            "result_path": str(path),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("No results found in %s", results_dir)
        return df

    # Filter to requested seed (default: 42 to match paper Protocol A numbers)
    if seed is not None:
        pre_count = len(df)
        df = df[df["seed"] == seed].reset_index(drop=True)
        logger.info("Filtered to seed=%d: %d -> %d runs", seed, pre_count, len(df))

    # Fix CLIP param counts: text encoder (63M params) was incorrectly
    # included as trainable. Only the head/LoRA/VPT params are truly trainable.
    clip_mask = df["model"].str.startswith("clip_")
    if clip_mask.any():
        clip_text_params = 63_428_096  # CLIP-B text encoder params
        clip_rows = df.loc[clip_mask]
        df.loc[clip_mask, "trainable_params"] = (
            clip_rows["trainable_params"] - clip_text_params
        ).clip(lower=0)
        total = df.loc[clip_mask, "total_params"]
        trainable = df.loc[clip_mask, "trainable_params"]
        df.loc[clip_mask, "trainable_pct"] = (
            100.0 * trainable / total.replace(0, 1)
        )

    # Add display names and combo label
    df["model_display"] = df["model"].map(
        lambda m: MODEL_DISPLAY_NAMES.get(m, m)
    )
    df["adapt_display"] = df["adaptation"].map(
        lambda a: ADAPTATION_DISPLAY_NAMES.get(a, a)
    )
    df["combo"] = df["model_display"] + " + " + df["adapt_display"]

    # Class imbalance ratio (empty / total)
    df["empty_ratio"] = df["empty_support"] / df["n_samples"].replace(0, 1)

    logger.info(
        "Loaded %d runs: %d combos, %d habitats",
        len(df),
        df.groupby(["model", "adaptation"]).ngroups,
        df["target_habitat"].nunique(),
    )

    return df
