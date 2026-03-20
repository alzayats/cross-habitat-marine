"""Per-species transfer performance analysis.

Identifies which species transfer well across habitats and which don't,
enabling ecological insights about habitat-specific visual features.

Usage:
    python -m analysis.per_species_analysis --results-dir outputs/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all experiment result JSON files.

    Args:
        results_dir: Root directory containing result files.

    Returns:
        List of result dictionaries.
    """
    results = []
    for path in Path(results_dir).rglob("results.json"):
        try:
            with open(path) as f:
                results.append(json.load(f))
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
    logger.info("Loaded %d result files from %s", len(results), results_dir)
    return results


def extract_per_species_metrics(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Extract per-species metrics from all results into a DataFrame.

    Args:
        results: List of experiment results.

    Returns:
        DataFrame with columns: model, adaptation, source, target, species, f1, precision, recall.
    """
    rows = []
    for r in results:
        test_metrics = r.get("test_metrics", {})
        per_class = test_metrics.get("per_class", {})

        for species, metrics in per_class.items():
            rows.append(
                {
                    "model": r.get("model", "unknown"),
                    "adaptation": r.get("adaptation", "unknown"),
                    "source": r.get("source_habitats", r.get("source", "unknown")),
                    "target": r.get("target_habitat", r.get("target", "unknown")),
                    "seed": r.get("seed", 0),
                    "species": species,
                    "f1": metrics.get("f1", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "support": metrics.get("support", 0),
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Extracted per-species metrics: %d rows", len(df))
    return df


def compute_transfer_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-species transfer gap (in-domain vs cross-domain F1).

    Args:
        df: Per-species metrics DataFrame.

    Returns:
        DataFrame with transfer gap per species.
    """
    # Group by species and compute mean F1 across seeds
    grouped = df.groupby(["model", "adaptation", "species"]).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        n_experiments=("f1", "count"),
    ).reset_index()

    return grouped


def plot_species_transfer_heatmap(
    df: pd.DataFrame,
    model_name: str = "dinov2_base",
    adaptation_name: str = "lora_r4",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-species F1 heatmap across transfer scenarios.

    Args:
        df: Per-species metrics DataFrame.
        model_name: Model to plot.
        adaptation_name: Adaptation to plot.
        output_path: Output file path.

    Returns:
        matplotlib Figure.
    """
    subset = df[(df["model"] == model_name) & (df["adaptation"] == adaptation_name)]

    if subset.empty:
        logger.warning("No data for %s + %s", model_name, adaptation_name)
        return plt.figure()

    # Pivot: species vs transfer scenario
    subset = subset.copy()
    subset["scenario"] = subset["source"].astype(str) + " -> " + subset["target"].astype(str)
    pivot = subset.pivot_table(
        values="f1", index="species", columns="scenario", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.4)))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        annot_kws={"size": 7},
    )

    ax.set_title(
        f"Per-Species F1: {model_name} + {adaptation_name}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Species", fontsize=9)
    ax.set_xlabel("Transfer Scenario", fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output_path)

    return fig


def identify_hard_species(
    df: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Identify species that are difficult to transfer across habitats.

    Args:
        df: Per-species metrics DataFrame.
        threshold: F1 threshold below which species is "hard".

    Returns:
        DataFrame of hard-to-transfer species.
    """
    mean_f1 = df.groupby("species")["f1"].mean().reset_index()
    hard = mean_f1[mean_f1["f1"] < threshold].sort_values("f1")

    if not hard.empty:
        logger.info("Hard-to-transfer species (F1 < %.2f):", threshold)
        for _, row in hard.iterrows():
            logger.info("  %s: F1 = %.3f", row["species"], row["f1"])

    return hard


def statistical_comparison(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test comparing two models per-species.

    Args:
        df: Per-species metrics DataFrame.
        model_a: First model name.
        model_b: Second model name.

    Returns:
        Statistical test results.
    """
    df_a = df[df["model"] == model_a].groupby("species")["f1"].mean()
    df_b = df[df["model"] == model_b].groupby("species")["f1"].mean()

    shared_species = set(df_a.index) & set(df_b.index)
    if len(shared_species) < 5:
        return {"error": "Insufficient shared species for comparison"}

    vals_a = [df_a[s] for s in sorted(shared_species)]
    vals_b = [df_b[s] for s in sorted(shared_species)]

    stat, p_value = stats.wilcoxon(vals_a, vals_b)

    result = {
        "model_a": model_a,
        "model_b": model_b,
        "n_species": len(shared_species),
        "mean_f1_a": float(np.mean(vals_a)),
        "mean_f1_b": float(np.mean(vals_b)),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }

    logger.info(
        "Wilcoxon %s vs %s: p=%.4f (%s)",
        model_a,
        model_b,
        p_value,
        "significant" if p_value < 0.05 else "not significant",
    )

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Per-species transfer analysis")
    parser.add_argument("--results-dir", default="./outputs")
    parser.add_argument("--output-dir", default="./outputs/figures")

    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    df = extract_per_species_metrics(results)

    if df.empty:
        logger.error("No per-species metrics found")
        return

    # Generate heatmaps for key models
    for model in df["model"].unique():
        for adapt in df["adaptation"].unique():
            output = str(
                Path(args.output_dir) / f"species_heatmap_{model}_{adapt}.png"
            )
            plot_species_transfer_heatmap(df, model, adapt, output)

    # Identify hard species
    identify_hard_species(df)

    # Statistical comparisons
    models = df["model"].unique().tolist()
    if len(models) >= 2:
        statistical_comparison(df, models[0], models[1])

    plt.close("all")


if __name__ == "__main__":
    main()
