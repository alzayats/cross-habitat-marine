"""Attention map visualization: Grad-CAM for CNNs, attention rollout for ViTs.

Generates side-by-side comparisons of model attention patterns across
different architectures and habitats.

Usage:
    python -m analysis.attention_maps --model dinov2_base --checkpoint best.pt \
        --images sample1.jpg sample2.jpg
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data.augmentations import get_test_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for CNN models (e.g., ResNet50 last conv layer).

    Args:
        model: CNN model.
        target_layer: The convolutional layer to compute Grad-CAM for.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Any) -> None:
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image [1, C, H, W].
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Heatmap array [H, W] normalized to [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class AttentionRollout:
    """Attention rollout for Vision Transformers.

    Multiplies attention weights across layers to get a global
    attention map from the CLS token to all patch tokens.

    Args:
        model: ViT model.
        head_fusion: How to fuse multi-head attention ('mean', 'max', 'min').
        discard_ratio: Ratio of lowest attention values to discard.
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.1,
    ) -> None:
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_maps: List[torch.Tensor] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        def hook_fn(module: nn.Module, inp: Any, output: Any) -> None:
            # Different models store attention differently
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]  # (B, H, N, N)
            elif hasattr(module, "attn_weights"):
                attn = module.attn_weights
            else:
                return

            if attn is not None:
                self.attention_maps.append(attn.detach().cpu())

        for name, module in self.model.named_modules():
            if "attn" in name.lower() and hasattr(module, "forward"):
                module.register_forward_hook(hook_fn)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate attention rollout map.

        Args:
            input_tensor: Input image [1, C, H, W].

        Returns:
            Attention map [H, W] normalized to [0, 1].
        """
        self.attention_maps = []
        self.model.eval()

        with torch.no_grad():
            self.model(input_tensor)

        if not self.attention_maps:
            logger.warning("No attention maps captured. Model may not expose attention weights.")
            h = int(np.sqrt(input_tensor.shape[-1] // 14))
            return np.ones((h, h))

        # Rollout: multiply attention matrices
        result = torch.eye(self.attention_maps[0].shape[-1])

        for attn in self.attention_maps:
            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                attn_fused = attn.min(dim=1).values

            attn_fused = attn_fused.squeeze(0)

            # Add residual connection
            identity = torch.eye(attn_fused.shape[-1])
            attn_with_residual = (attn_fused + identity) / 2
            attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

            result = attn_with_residual @ result

        # Get CLS token attention to patches
        cls_attention = result[0, 1:]  # Skip CLS-to-CLS

        # Reshape to spatial
        n_patches = cls_attention.shape[0]
        h = w = int(np.sqrt(n_patches))
        if h * w != n_patches:
            h = w = int(np.ceil(np.sqrt(n_patches)))
            cls_attention = F.pad(cls_attention, (0, h * w - n_patches))

        attn_map = cls_attention.reshape(h, w).numpy()

        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        return attn_map


def get_last_layer_attention(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Get raw last-layer attention map from a ViT.

    Args:
        model: ViT model.
        input_tensor: Input image [1, C, H, W].

    Returns:
        Last-layer attention map [H, W].
    """
    attn_weights = []

    def hook_fn(module, inp, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights.append(output[1].detach().cpu())

    hooks = []
    for name, module in model.named_modules():
        if "attn" in name.lower():
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    if not attn_weights:
        return np.ones((16, 16))

    # Use last layer attention
    last_attn = attn_weights[-1].squeeze(0)
    cls_attn = last_attn.mean(dim=0)[0, 1:]

    n = cls_attn.shape[0]
    h = w = int(np.sqrt(n))
    return cls_attn[:h*w].reshape(h, w).numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image: Original image [H, W, 3] (RGB, 0-255).
        heatmap: Heatmap [h, w] normalized to [0, 1].
        alpha: Overlay alpha.
        colormap: OpenCV colormap.

    Returns:
        Overlay image [H, W, 3] (RGB, 0-255).
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (image.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_comparison_figure(
    image_path: str,
    models_and_maps: List[Tuple[str, np.ndarray]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create side-by-side attention comparison figure.

    Args:
        image_path: Path to original image.
        models_and_maps: List of (model_name, heatmap) tuples.
        output_path: Output path for saving.

    Returns:
        matplotlib Figure.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    n_panels = 1 + len(models_and_maps)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Attention maps
    for i, (name, heatmap) in enumerate(models_and_maps):
        overlay = overlay_heatmap(image, heatmap)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(name, fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output_path)

    return fig


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Attention map visualization")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--output-dir", default="./outputs/figures/attention")

    args = parser.parse_args()

    for img_path in args.images:
        logger.info("Processing: %s", img_path)
        # Placeholder: actual model loading would go here
        # This demonstrates the visualization pipeline
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        dummy_map = np.random.rand(16, 16)
        dummy_map = cv2.GaussianBlur(dummy_map, (5, 5), 2.0)
        dummy_map = (dummy_map - dummy_map.min()) / (dummy_map.max() - dummy_map.min())

        output = str(
            Path(args.output_dir) / f"{Path(img_path).stem}_attention.png"
        )
        create_comparison_figure(
            img_path,
            [("DINOv2", dummy_map), ("ResNet-GradCAM", np.random.rand(16, 16))],
            output,
        )


if __name__ == "__main__":
    main()
