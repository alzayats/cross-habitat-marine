"""t-SNE and UMAP visualization of feature spaces.

Creates panels colored by species and by habitat to show how well
foundation models separate species across different underwater habitats.

Usage:
    python -m analysis.tsne_visualization --features-dir outputs/features/
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features(feature_path: str) -> Dict[str, np.ndarray]:
    """Load features from .npz file.

    Args:
        feature_path: Path to .npz feature file.

    Returns:
        Dict with 'features', 'labels', 'habitats'.
    """
    data = np.load(feature_path, allow_pickle=True)
    return {
        "features": data["features"],
        "labels": data["labels"],
        "habitats": data["habitats"],
    }


def compute_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE embedding.

    Args:
        features: Feature matrix [N, D].
        perplexity: t-SNE perplexity parameter.
        n_components: Output dimensions.
        random_state: Random seed.

    Returns:
        t-SNE embedding [N, n_components].
    """
    logger.info(
        "Computing t-SNE: N=%d, D=%d, perplexity=%.1f",
        features.shape[0],
        features.shape[1],
        perplexity,
    )
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        init="pca",
    )
    return tsne.fit_transform(features)


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP embedding.

    Args:
        features: Feature matrix [N, D].
        n_neighbors: Number of nearest neighbors.
        min_dist: Minimum distance in embedding.
        n_components: Output dimensions.
        random_state: Random seed.

    Returns:
        UMAP embedding [N, n_components].
    """
    import umap

    logger.info(
        "Computing UMAP: N=%d, D=%d, n_neighbors=%d",
        features.shape[0],
        features.shape[1],
        n_neighbors,
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def plot_embedding_panels(
    embedding: np.ndarray,
    labels: np.ndarray,
    habitats: np.ndarray,
    title: str = "",
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Create side-by-side panels: (a) colored by species, (b) colored by habitat.

    Args:
        embedding: 2D embedding [N, 2].
        labels: Integer class labels.
        habitats: Habitat identifiers.
        title: Figure title.
        class_names: Optional class name labels.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel (a): Colored by species
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        ax1.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[cmap(i)],
            label=name,
            s=8,
            alpha=0.6,
        )

    ax1.set_title("(a) Colored by Species", fontsize=10)
    ax1.set_xlabel("Dimension 1", fontsize=8)
    ax1.set_ylabel("Dimension 2", fontsize=8)
    ax1.tick_params(labelsize=7)
    if len(unique_labels) <= 15:
        ax1.legend(fontsize=6, markerscale=2, loc="best", framealpha=0.7)

    # Panel (b): Colored by habitat
    unique_habitats = np.unique(habitats)
    habitat_cmap = plt.cm.get_cmap("Set2", len(unique_habitats))

    for i, hab in enumerate(unique_habitats):
        mask = habitats == hab
        ax2.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[habitat_cmap(i)],
            label=str(hab),
            s=8,
            alpha=0.6,
        )

    ax2.set_title("(b) Colored by Habitat", fontsize=10)
    ax2.set_xlabel("Dimension 1", fontsize=8)
    ax2.set_ylabel("Dimension 2", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.7)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
        logger.info("Saved: %s", output_path)

    return fig


def compare_models(
    feature_files: List[str],
    model_names: List[str],
    output_dir: str = "./outputs/figures",
    method: str = "tsne",
) -> None:
    """Create comparison panels for multiple models.

    Args:
        feature_files: List of .npz feature file paths.
        model_names: Corresponding model names.
        output_dir: Output directory for figures.
        method: 'tsne' or 'umap'.
    """
    n_models = len(feature_files)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for i, (fpath, name) in enumerate(zip(feature_files, model_names)):
        data = load_features(fpath)

        if method == "tsne":
            embedding = compute_tsne(data["features"])
        else:
            embedding = compute_umap(data["features"])

        # Top row: by species
        unique_labels = np.unique(data["labels"])
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))
        for j, lbl in enumerate(unique_labels):
            mask = data["labels"] == lbl
            axes[0, i].scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap(j)], s=5, alpha=0.5,
            )
        axes[0, i].set_title(f"{name}\n(by species)", fontsize=9)

        # Bottom row: by habitat
        unique_habs = np.unique(data["habitats"])
        hab_cmap = plt.cm.get_cmap("Set2", len(unique_habs))
        for j, hab in enumerate(unique_habs):
            mask = data["habitats"] == hab
            axes[1, i].scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[hab_cmap(j)], label=str(hab), s=5, alpha=0.5,
            )
        axes[1, i].set_title(f"{name}\n(by habitat)", fontsize=9)
        axes[1, i].legend(fontsize=6, markerscale=2)

    plt.tight_layout()
    output = Path(output_dir) / f"tsne_comparison_{method}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=300, bbox_inches="tight")
    logger.info("Saved model comparison: %s", output)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="t-SNE / UMAP visualization")
    parser.add_argument("--features-dir", type=str, default="./outputs/features")
    parser.add_argument("--output-dir", type=str, default="./outputs/figures")
    parser.add_argument("--method", choices=["tsne", "umap", "both"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)

    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    feature_files = sorted(features_dir.glob("*.npz"))

    if not feature_files:
        logger.error("No feature files found in %s", features_dir)
        return

    for fpath in feature_files:
        data = load_features(str(fpath))
        name = fpath.stem.replace("_features", "")

        if args.method in ("tsne", "both"):
            embedding = compute_tsne(data["features"], perplexity=args.perplexity)
            output = str(Path(args.output_dir) / f"tsne_{name}.png")
            plot_embedding_panels(
                embedding, data["labels"], data["habitats"],
                title=f"t-SNE: {name}", output_path=output,
            )

        if args.method in ("umap", "both"):
            embedding = compute_umap(data["features"])
            output = str(Path(args.output_dir) / f"umap_{name}.png")
            plot_embedding_panels(
                embedding, data["labels"], data["habitats"],
                title=f"UMAP: {name}", output_path=output,
            )

    plt.close("all")


if __name__ == "__main__":
    main()
