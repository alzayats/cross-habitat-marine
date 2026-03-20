"""Centered Kernel Alignment (CKA) analysis.

Computes linear CKA between feature representations to measure:
- Habitat invariance: same model, different habitats
- Model similarity: different models, same habitat

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.

Usage:
    python -m analysis.cka_analysis --features-dir outputs/features/
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def linear_cka(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: Feature matrix [N, D1].
        Y: Feature matrix [N, D2].

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center the features
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute HSIC
    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return float(cka)


def compute_cka_matrix(
    feature_sets: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise CKA matrix for multiple feature sets.

    Args:
        feature_sets: Dict mapping name -> feature matrix [N, D].

    Returns:
        Tuple of (CKA matrix, list of names).
    """
    names = list(feature_sets.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                # Need same number of samples; subsample if needed
                X = feature_sets[names[i]]
                Y = feature_sets[names[j]]
                min_n = min(len(X), len(Y))
                X = X[:min_n]
                Y = Y[:min_n]
                cka = linear_cka(X, Y)
                matrix[i, j] = cka
                matrix[j, i] = cka

    return matrix, names


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    names: List[str],
    title: str = "CKA Similarity",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 7),
) -> plt.Figure:
    """Plot CKA similarity matrix as a heatmap.

    Args:
        cka_matrix: CKA matrix [N, N].
        names: List of feature set names.
        title: Plot title.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cka_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
        logger.info("CKA heatmap saved: %s", output_path)

    return fig


def analyze_habitat_invariance(
    features_dir: str,
    output_dir: str = "./outputs/figures",
) -> Dict[str, float]:
    """Compute CKA between same model across different habitats.

    Args:
        features_dir: Directory containing .npz feature files.
        output_dir: Output directory for figures.

    Returns:
        Dict of CKA scores.
    """
    features_path = Path(features_dir)
    feature_files = sorted(features_path.glob("*.npz"))

    if not feature_files:
        logger.warning("No feature files found in %s", features_dir)
        return {}

    # Group by model
    model_habitats: Dict[str, Dict[str, np.ndarray]] = {}

    for fpath in feature_files:
        data = np.load(fpath, allow_pickle=True)
        features = data["features"]
        habitats = data["habitats"]

        model_name = fpath.stem.replace("_features", "")

        for hab in np.unique(habitats):
            mask = habitats == hab
            key = f"{model_name}_{hab}"

            if model_name not in model_habitats:
                model_habitats[model_name] = {}
            model_habitats[model_name][str(hab)] = features[mask]

    # Compute within-model cross-habitat CKA
    results: Dict[str, float] = {}
    for model_name, hab_features in model_habitats.items():
        habs = list(hab_features.keys())
        for i in range(len(habs)):
            for j in range(i + 1, len(habs)):
                cka = linear_cka(hab_features[habs[i]], hab_features[habs[j]])
                key = f"{model_name}:{habs[i]}_vs_{habs[j]}"
                results[key] = cka
                logger.info("CKA %s = %.4f", key, cka)

    return results


def analyze_model_similarity(
    features_dir: str,
    output_dir: str = "./outputs/figures",
) -> None:
    """Compute CKA between different models on the same data.

    Args:
        features_dir: Directory with feature .npz files.
        output_dir: Output directory.
    """
    features_path = Path(features_dir)
    feature_files = sorted(features_path.glob("*.npz"))

    feature_sets: Dict[str, np.ndarray] = {}
    for fpath in feature_files:
        data = np.load(fpath, allow_pickle=True)
        name = fpath.stem.replace("_features", "")
        feature_sets[name] = data["features"]

    if len(feature_sets) < 2:
        logger.warning("Need at least 2 feature sets for comparison")
        return

    matrix, names = compute_cka_matrix(feature_sets)

    output = str(Path(output_dir) / "cka_model_similarity.png")
    plot_cka_heatmap(
        matrix, names,
        title="CKA: Model Similarity",
        output_path=output,
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CKA analysis")
    parser.add_argument("--features-dir", default="./outputs/features")
    parser.add_argument("--output-dir", default="./outputs/figures")
    parser.add_argument(
        "--mode",
        choices=["habitat", "model", "both"],
        default="both",
    )

    args = parser.parse_args()

    if args.mode in ("habitat", "both"):
        analyze_habitat_invariance(args.features_dir, args.output_dir)

    if args.mode in ("model", "both"):
        analyze_model_similarity(args.features_dir, args.output_dir)


if __name__ == "__main__":
    main()
