"""Habitat Transfer Gap metric computation.

Defines and computes the Habitat Transfer Gap (HTG): the difference
between in-habitat performance and cross-habitat performance, normalized
by in-habitat performance. Lower HTG = better cross-habitat generalization.

HTG = (in_habitat_F1 - cross_habitat_F1) / in_habitat_F1

Usage:
    python -m analysis.habitat_gap_metric --results-dir outputs/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_habitat_transfer_gap(
    in_habitat_f1: float,
    cross_habitat_f1: float,
) -> float:
    """Compute the Habitat Transfer Gap.

    HTG = (in_habitat_F1 - cross_habitat_F1) / in_habitat_F1

    Args:
        in_habitat_f1: F1 score when evaluated on same habitat as training.
        cross_habitat_f1: F1 score when evaluated on different habitat.

    Returns:
        HTG value. 0 = perfect transfer, 1 = complete failure.
    """
    if in_habitat_f1 <= 0:
        return 1.0
    gap = (in_habitat_f1 - cross_habitat_f1) / in_habitat_f1
    return float(np.clip(gap, 0, 1))


def compute_relative_transfer_efficiency(
    foundation_htg: float,
    baseline_htg: float,
) -> float:
    """Compute Relative Transfer Efficiency (RTE).

    RTE = (baseline_HTG - foundation_HTG) / baseline_HTG
    Positive RTE means foundation model transfers better.

    Args:
        foundation_htg: HTG for foundation model.
        baseline_htg: HTG for baseline (e.g., ResNet50).

    Returns:
        RTE value.
    """
    if baseline_htg <= 0:
        return 0.0
    return float((baseline_htg - foundation_htg) / baseline_htg)


def load_and_compute_gaps(
    results_dir: str,
) -> pd.DataFrame:
    """Load results and compute HTG for all experiments.

    Args:
        results_dir: Directory with result JSON files.

    Returns:
        DataFrame with HTG values.
    """
    results = []
    for path in Path(results_dir).rglob("results.json"):
        try:
            with open(path) as f:
                r = json.load(f)
            results.append(r)
        except Exception:
            continue

    if not results:
        logger.warning("No results found in %s", results_dir)
        return pd.DataFrame()

    # Separate in-habitat and cross-habitat results
    rows = []
    for r in results:
        test_metrics = r.get("test_metrics", {})
        macro_f1 = test_metrics.get("macro_f1", 0.0)

        row = {
            "model": r.get("model", "unknown"),
            "adaptation": r.get("adaptation", "unknown"),
            "source": str(r.get("source_habitats", r.get("source", "unknown"))),
            "target": str(r.get("target_habitat", r.get("target", "unknown"))),
            "seed": r.get("seed", 0),
            "macro_f1": macro_f1,
            "top1_accuracy": test_metrics.get("top1_accuracy", 0.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute HTG: need to match in-domain vs cross-domain results
    # In-domain: source == target (or same habitat)
    # Cross-domain: source != target

    # Group and compute gaps
    gap_rows = []
    for (model, adapt), group in df.groupby(["model", "adaptation"]):
        # Find pairs where we have both in-domain and cross-domain results
        for target in group["target"].unique():
            in_domain = group[group["source"].str.contains(str(target), na=False)]
            cross_domain = group[
                (group["target"] == target) &
                (~group["source"].str.contains(str(target), na=False))
            ]

            if in_domain.empty or cross_domain.empty:
                continue

            in_f1 = in_domain["macro_f1"].mean()
            cross_f1 = cross_domain["macro_f1"].mean()
            htg = compute_habitat_transfer_gap(in_f1, cross_f1)

            gap_rows.append(
                {
                    "model": model,
                    "adaptation": adapt,
                    "target": target,
                    "in_habitat_f1": in_f1,
                    "cross_habitat_f1": cross_f1,
                    "htg": htg,
                }
            )

    gap_df = pd.DataFrame(gap_rows)
    if not gap_df.empty:
        logger.info("Computed HTG for %d configurations", len(gap_df))
    return gap_df


def plot_htg_comparison(
    gap_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot HTG comparison across models.

    Args:
        gap_df: DataFrame with HTG values.
        output_path: Output file path.

    Returns:
        matplotlib Figure.
    """
    if gap_df.empty:
        return plt.figure()

    # Average HTG per model
    model_htg = gap_df.groupby("model")["htg"].agg(["mean", "std"]).reset_index()
    model_htg = model_htg.sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(
        model_htg["model"],
        model_htg["mean"],
        xerr=model_htg["std"],
        capsize=3,
        color=plt.cm.RdYlGn_r(model_htg["mean"] / model_htg["mean"].max()),
        edgecolor="gray",
        linewidth=0.5,
    )

    ax.set_xlabel("Habitat Transfer Gap (lower = better)", fontsize=10)
    ax.set_title("Habitat Transfer Gap by Model", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=9)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output_path)

    return fig


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Habitat Transfer Gap analysis")
    parser.add_argument("--results-dir", default="./outputs")
    parser.add_argument("--output-dir", default="./outputs/figures")

    args = parser.parse_args()

    gap_df = load_and_compute_gaps(args.results_dir)

    if not gap_df.empty:
        output = str(Path(args.output_dir) / "habitat_transfer_gap.png")
        plot_htg_comparison(gap_df, output)

        # Print summary
        summary = gap_df.groupby("model")["htg"].mean().sort_values()
        logger.info("HTG Summary (lower = better transfer):")
        for model, htg in summary.items():
            logger.info("  %s: %.4f", model, htg)
    else:
        logger.warning("No HTG data available")


if __name__ == "__main__":
    main()
