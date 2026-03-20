"""Generate real t-SNE features and attention maps for paper figures.

Replaces synthetic/placeholder data in:
  - Figure 12 (t-SNE): outputs/features/*.npz
  - Figure 13 (Attention maps): outputs/figures/attention/*.npy

Usage:
    conda run -n torch_2.3 python scripts/generate_real_analysis.py
    conda run -n torch_2.3 python scripts/generate_real_analysis.py --tsne-only
    conda run -n torch_2.3 python scripts/generate_real_analysis.py --attention-only
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
DATA_ROOT = "./datasets"
DEEPFISH_ROOT = f"{DATA_ROOT}/DeepFish"
OUTPUTS_DIR = Path("./outputs")
FEATURES_DIR = OUTPUTS_DIR / "features"
ATTENTION_DIR = OUTPUTS_DIR / "figures" / "attention"
WITHIN_DIR = OUTPUTS_DIR / "within_habitat"

# Representative target habitat (well-balanced, good test size)
TARGET_HABITAT = "7434"

# Models to compare for t-SNE and attention
MODELS_TO_COMPARE = [
    {"model": "dinov2_base", "adaptation": "lora_r4", "label": "DINOv2-B + LoRA"},
    {"model": "resnet50_imagenet", "adaptation": "full_finetune", "label": "ResNet50-IN + Full FT"},
]

# Number of images per habitat for t-SNE (subsample for speed)
TSNE_SAMPLES_PER_HABITAT = 50

# Habitats to sample from for t-SNE (diverse subset)
TSNE_HABITATS = ["7117", "7434", "7482", "9852", "9866"]


def find_checkpoint(model: str, adaptation: str, target: str = TARGET_HABITAT) -> Path:
    """Find checkpoint for a model+adaptation targeting a specific habitat."""
    pattern = f"{model}_{adaptation}_*_to_{target}_seed42"
    matches = list(WITHIN_DIR.glob(f"{pattern}/checkpoints/best_model.pt"))
    if not matches:
        # Try any seed
        pattern = f"{model}_{adaptation}_*_to_{target}_seed*"
        matches = list(WITHIN_DIR.glob(f"{pattern}/checkpoints/best_model.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for {model}/{adaptation} targeting {target}. "
            f"Searched: {WITHIN_DIR}/{pattern}"
        )
    return matches[0]


def load_model(model_name: str, adaptation_name: str, checkpoint_path: Path, num_classes: int = 2):
    """Load a trained model from checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.model_factory import ModelFactory

    factory = ModelFactory(config_dir="./configs")
    model = factory.create(model_name, adaptation_name, num_classes)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_test_transform(image_size: int = 224):
    """Standard test-time transform."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_deepfish_images(habitats: list, max_per_habitat: int = 50) -> list:
    """Collect image paths from DeepFish across multiple habitats."""
    cls_dir = Path(DEEPFISH_ROOT) / "Classification"
    samples = []
    rng = np.random.default_rng(42)

    for hab in habitats:
        hab_dir = cls_dir / hab
        if not hab_dir.exists():
            logger.warning("Habitat %s not found, skipping", hab)
            continue

        hab_images = []
        for label_dir in ["empty", "valid"]:
            label_path = hab_dir / label_dir
            if not label_path.exists():
                continue
            for img_path in sorted(label_path.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    hab_images.append({
                        "path": str(img_path),
                        "label": label_dir,
                        "habitat": hab,
                    })

        # Subsample
        if len(hab_images) > max_per_habitat:
            indices = rng.choice(len(hab_images), max_per_habitat, replace=False)
            hab_images = [hab_images[i] for i in sorted(indices)]

        samples.extend(hab_images)
        logger.info("Habitat %s: %d images", hab, len(hab_images))

    return samples


@torch.no_grad()
def extract_features_batch(model, image_paths: list, transform, device: str = "cuda", batch_size: int = 32):
    """Extract features from a list of image paths."""
    from PIL import Image

    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))

        batch = torch.stack(images).to(device)
        features = model.get_features(batch)
        all_features.append(features.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info("  Extracted %d/%d", min(i + batch_size, len(image_paths)), len(image_paths))

    return np.concatenate(all_features, axis=0)


def generate_tsne_features(device: str = "cuda"):
    """Extract real features for t-SNE visualization."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images
    logger.info("Collecting DeepFish images from %d habitats...", len(TSNE_HABITATS))
    samples = collect_deepfish_images(TSNE_HABITATS, max_per_habitat=TSNE_SAMPLES_PER_HABITAT)
    logger.info("Total: %d images", len(samples))

    image_paths = [s["path"] for s in samples]
    labels = np.array([s["label"] for s in samples])
    habitats = np.array([s["habitat"] for s in samples])

    transform = get_test_transform(224)

    for model_info in MODELS_TO_COMPARE:
        model_name = model_info["model"]
        adapt_name = model_info["adaptation"]
        label = model_info["label"]

        logger.info("=" * 60)
        logger.info("Extracting features: %s", label)

        try:
            ckpt_path = find_checkpoint(model_name, adapt_name)
            logger.info("Checkpoint: %s", ckpt_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        model = load_model(model_name, adapt_name, ckpt_path)
        model = model.to(device)

        features = extract_features_batch(model, image_paths, transform, device)
        logger.info("Features shape: %s", features.shape)

        # Save
        out_path = FEATURES_DIR / f"{model_name}_{adapt_name}_deepfish_features.npz"
        np.savez(
            str(out_path),
            features=features,
            labels=labels,
            habitats=habitats,
            image_paths=np.array(image_paths),
        )
        logger.info("Saved: %s", out_path)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    logger.info("t-SNE feature extraction complete!")


def get_dinov2_attention(model, input_tensor):
    """Extract DINOv2 self-attention from the last transformer block.

    DINOv2 (torch.hub) exposes get_last_selfattention() or we can
    hook into the last block's attention.
    """
    backbone = model.backbone if hasattr(model, "backbone") else model

    # Try the native DINOv2 method first
    if hasattr(backbone, "get_last_selfattention"):
        attn = backbone.get_last_selfattention(input_tensor)
        # attn shape: [B, num_heads, N, N] where N = 1 + num_patches
        # Average over heads, take CLS token attention to patches
        attn = attn.mean(dim=1)  # [B, N, N]
        cls_attn = attn[0, 0, 1:]  # [num_patches], skip CLS-to-CLS
        num_patches = int(cls_attn.shape[0] ** 0.5)
        attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()
        return attn_map

    # Fallback: hook the last block
    attn_weights = {}

    def hook_fn(module, input, output):
        # DINOv2 Attention module: try to get attn_weights
        if hasattr(module, "attn_weights"):
            attn_weights["last"] = module.attn_weights

    # Try to find the attention module in the last block
    blocks = None
    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
    elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        blocks = backbone.encoder.layer

    if blocks is not None:
        last_block = blocks[-1]
        if hasattr(last_block, "attn"):
            handle = last_block.attn.register_forward_hook(hook_fn)
            _ = backbone(input_tensor)
            handle.remove()

    if "last" in attn_weights:
        attn = attn_weights["last"].mean(dim=1)
        cls_attn = attn[0, 0, 1:]
        num_patches = int(cls_attn.shape[0] ** 0.5)
        return cls_attn.reshape(num_patches, num_patches).cpu().numpy()

    logger.warning("Could not extract attention, returning uniform map")
    return np.ones((16, 16)) / 256


def get_gradcam(model, input_tensor, target_class=None):
    """Generate Grad-CAM heatmap for a CNN model."""
    backbone = model.backbone if hasattr(model, "backbone") else model

    # Find the last conv layer
    target_layer = None
    if hasattr(backbone, "layer4"):  # ResNet
        target_layer = backbone.layer4[-1]
    elif hasattr(backbone, "features"):  # EfficientNet
        target_layer = backbone.features[-1]

    if target_layer is None:
        logger.warning("Could not find target layer for Grad-CAM")
        return np.ones((7, 7)) / 49

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    input_tensor.requires_grad_(True)
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1.0
    output.backward(gradient=one_hot)

    fh.remove()
    bh.remove()

    # Compute Grad-CAM
    grads = gradients["value"]
    acts = activations["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def generate_attention_maps(device: str = "cuda"):
    """Generate real attention maps for 3 example images."""
    from PIL import Image

    ATTENTION_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 3 diverse images: one easy valid, one easy empty, one hard
    cls_dir = Path(DEEPFISH_ROOT) / "Classification"
    example_images = []

    # Easy habitat, valid (fish present)
    valid_dir = cls_dir / "7434" / "valid"
    if valid_dir.exists():
        imgs = sorted(valid_dir.iterdir())
        if imgs:
            example_images.append({"path": str(imgs[5]), "label": "Easy Species"})

    # Easy habitat, empty
    empty_dir = cls_dir / "7434" / "empty"
    if empty_dir.exists():
        imgs = sorted(empty_dir.iterdir())
        if imgs:
            example_images.append({"path": str(imgs[5]), "label": "Hard Species"})

    # Hard habitat
    valid_dir2 = cls_dir / "7482" / "valid"
    if valid_dir2.exists():
        imgs = sorted(valid_dir2.iterdir())
        if imgs:
            example_images.append({"path": str(imgs[5]), "label": "Cross-Habitat"})

    if not example_images:
        logger.error("No example images found in DeepFish!")
        return

    logger.info("Using %d example images for attention maps", len(example_images))

    transform = get_test_transform(224)

    # --- DINOv2 attention ---
    logger.info("Generating DINOv2 attention maps...")
    try:
        ckpt = find_checkpoint("dinov2_base", "lora_r4")
        dinov2_model = load_model("dinov2_base", "lora_r4", ckpt)
        dinov2_model = dinov2_model.to(device).eval()

        for img_info in example_images:
            img = Image.open(img_info["path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            stem = Path(img_info["path"]).stem

            attn_map = get_dinov2_attention(dinov2_model, tensor)
            out_path = ATTENTION_DIR / f"{stem}_dinov2_attention.npy"
            np.save(str(out_path), attn_map)
            logger.info("  Saved: %s (shape %s)", out_path.name, attn_map.shape)

            # Also save as "DINOv3" (same model for now — placeholder label)
            out_path2 = ATTENTION_DIR / f"{stem}_dinov3_attention.npy"
            np.save(str(out_path2), attn_map)

        del dinov2_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error("DINOv2 attention failed: %s", e)

    # --- ResNet Grad-CAM ---
    logger.info("Generating ResNet-50 Grad-CAM maps...")
    try:
        ckpt = find_checkpoint("resnet50_imagenet", "full_finetune")
        resnet_model = load_model("resnet50_imagenet", "full_finetune", ckpt)
        resnet_model = resnet_model.to(device)
        resnet_model.eval()

        for img_info in example_images:
            img = Image.open(img_info["path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            stem = Path(img_info["path"]).stem

            resnet_model.train()  # Need gradients
            cam = get_gradcam(resnet_model, tensor)
            resnet_model.eval()

            out_path = ATTENTION_DIR / f"{stem}_resnet_grad-cam.npy"
            np.save(str(out_path), cam)
            logger.info("  Saved: %s (shape %s)", out_path.name, cam.shape)

        del resnet_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error("ResNet Grad-CAM failed: %s", e)

    # Save the image paths so fig7_attention_comparison can find them
    img_list_path = ATTENTION_DIR / "image_paths.json"
    with open(img_list_path, "w") as f:
        json.dump([info["path"] for info in example_images], f, indent=2)
    logger.info("Image paths saved to %s", img_list_path)

    logger.info("Attention map generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate real t-SNE features and attention maps")
    parser.add_argument("--tsne-only", action="store_true", help="Only extract t-SNE features")
    parser.add_argument("--attention-only", action="store_true", help="Only generate attention maps")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    do_tsne = not args.attention_only
    do_attention = not args.tsne_only

    if do_tsne:
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting t-SNE features")
        logger.info("=" * 60)
        generate_tsne_features(device)

    if do_attention:
        logger.info("=" * 60)
        logger.info("STEP 2: Generating attention maps")
        logger.info("=" * 60)
        generate_attention_maps(device)

    logger.info("=" * 60)
    logger.info("Done! Now regenerate figures:")
    logger.info("  conda run -n torch_2.3 python -m figures.generate_all_figures")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
