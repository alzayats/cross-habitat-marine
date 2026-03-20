"""Master script to generate all paper figures.

Reads all experiment results and generates Figures 1-12 as both
PDF (for paper) and PNG (for presentations).

Usage:
    python -m figures.generate_all_figures
    python -m figures.generate_all_figures --output-dir outputs/figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_all(
    results_dir: str = "./outputs",
    output_dir: str = "./outputs/figures",
) -> None:
    """Generate all paper figures.

    Args:
        results_dir: Base directory containing all experiment results.
        output_dir: Output directory for figures.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Generating all figures from %s -> %s", results_dir, output_dir)

    # Figure 1: Overview (geographic map + experimental design)
    logger.info("Generating Figure 1: Overview...")
    try:
        from figures.fig1_option_D import generate_fig1_D
        generate_fig1_D(str(output / "fig1_overview"))
        plt.close("all")
    except Exception as e:
        logger.error("Figure 1 failed: %s", e)

    # Figure 2: Dataset samples
    logger.info("Generating Figure 2: Dataset samples...")
    try:
        from figures.fig2_dataset_samples import generate_fig2
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/base_config.yaml")
        data_root = cfg.data.root_dir
        datasets = {
            "DeepFish": f"{data_root}/DeepFish",
            "AQUA20": f"{data_root}/AQUA20_dataset",
            "Moorea Corals": f"{data_root}/Moorea_Labeled_Corals",
            "Coralscapes": f"{data_root}/Coralscapes_dataset",
            "Brackish": f"{data_root}/Brackish_dataset",
        }
        generate_fig2(datasets, output_path=str(output / "fig2_dataset_samples"))
        plt.close("all")
    except Exception as e:
        logger.error("Figure 2 failed: %s", e)

    # Figure 3: Within-habitat heatmap
    logger.info("Generating Figure 3: Within-habitat heatmap...")
    try:
        from figures.fig3_heatmap_within import generate_fig3
        generate_fig3(
            results_dir=f"{results_dir}/within_habitat",
            output_path=str(output / "fig3_heatmap_within"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 3 failed: %s", e)

    # Figure 4: Habitat Difficulty Landscape
    logger.info("Generating Figure 4: Habitat difficulty landscape...")
    try:
        from figures.fig4_habitat_landscape import generate_fig4
        generate_fig4(
            results_dir=f"{results_dir}/within_habitat",
            output_path=str(output / "fig4_habitat_landscape"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 4 failed: %s", e)

    # Figure 5: Class Imbalance Analysis
    logger.info("Generating Figure 5: Class imbalance analysis...")
    try:
        from figures.fig5_class_imbalance import generate_fig7
        generate_fig7(
            results_dir=f"{results_dir}/within_habitat",
            output_path=str(output / "fig5_class_imbalance"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 5 failed: %s", e)

    # Figure 6: Cross-Dataset Transfer (Protocol B)
    logger.info("Generating Figure 6: Cross-dataset transfer...")
    try:
        from figures.fig6_cross_dataset import generate_fig9
        generate_fig9(
            results_dir=f"{results_dir}/cross_dataset",
            output_path=str(output / "fig6_cross_dataset"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 6 failed: %s", e)

    # Figure 7: Transfer Robustness (Protocol B)
    logger.info("Generating Figure 7: Transfer robustness...")
    try:
        from figures.fig7_transfer_robustness import generate_fig10
        generate_fig10(
            cross_dir=f"{results_dir}/cross_dataset",
            within_dir=f"{results_dir}/within_habitat",
            output_path=str(output / "fig7_transfer_robustness"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 7 failed: %s", e)

    # Figure 8: Few-Shot Curves (Protocol C)
    logger.info("Generating Figure 8: Few-shot curves...")
    try:
        from figures.fig8_fewshot_curves import generate_fig11
        generate_fig11(
            results_dir=f"{results_dir}/fewshot_curve",
            output_path=str(output / "fig8_fewshot_curves"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 8 failed: %s", e)

    # Figure 9: Few-Shot Strategy (Protocol C)
    logger.info("Generating Figure 9: Few-shot strategy...")
    try:
        from figures.fig9_fewshot_strategy import generate_fig12
        generate_fig12(
            results_dir=f"{results_dir}/fewshot_curve",
            output_path=str(output / "fig9_fewshot_strategy"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 9 failed: %s", e)

    # Figure 10: t-SNE panels (DINOv2 vs RN50)
    logger.info("Generating Figure 10: t-SNE panels...")
    try:
        from figures.fig10_tsne_panels import generate_fig6 as generate_fig6_tsne
        generate_fig6_tsne(
            features_dir=f"{results_dir}/features",
            output_path=str(output / "fig10_tsne_panels"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 10 failed: %s", e)

    # Figure 11: Training Dynamics
    logger.info("Generating Figure 11: Training dynamics...")
    try:
        from figures.fig11_training_dynamics import generate_fig6
        generate_fig6(
            results_dir=f"{results_dir}/within_habitat",
            output_path=str(output / "fig11_training_dynamics"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 11 failed: %s", e)

    # Figure 12: Efficiency-Performance Tradeoff
    logger.info("Generating Figure 12: Efficiency-performance Pareto...")
    try:
        from figures.fig12_efficiency_pareto import generate_fig5
        generate_fig5(
            results_dir=f"{results_dir}/within_habitat",
            output_path=str(output / "fig12_efficiency_pareto"),
        )
        plt.close("all")
    except Exception as e:
        logger.error("Figure 12 failed: %s", e)

    # fig6b, 6c, 6d — dropped (uninformative)
    # fig7_attention_comparison — dropped from paper

    logger.info("=" * 50)
    logger.info("All figures generated in %s", output_dir)
    logger.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--results-dir", default="./outputs")
    parser.add_argument("--output-dir", default="./outputs/figures")
    args = parser.parse_args()
    generate_all(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
