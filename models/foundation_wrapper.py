"""Foundation model wrapper for DINOv2, DINOv3, CLIP, and MAE.

This is the core module that loads any supported foundation model,
applies adaptation strategies (LoRA, VPT, linear probe, full fine-tune),
and provides a unified forward() interface returning class logits.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FoundationModelWrapper(nn.Module):
    """Unified wrapper for vision foundation models with adaptation support.

    Loads a pretrained foundation model (DINOv2, DINOv3, CLIP, MAE),
    extracts features from [CLS] token or global average pool, and
    attaches a classification head. Supports LoRA, VPT, linear probing,
    and full fine-tuning adaptation strategies.

    Args:
        model_cfg: Model configuration (from YAML).
        adaptation_cfg: Adaptation strategy configuration.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        adaptation_cfg: DictConfig,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.adaptation_cfg = adaptation_cfg
        self.num_classes = num_classes
        self.feature_dim = model_cfg.feature_dim

        # Load backbone
        self.backbone = self._load_backbone()

        # Apply adaptation strategy
        self._apply_adaptation()

        # Attach classification head
        from models.classification_head import ClassificationHead

        head_type = getattr(adaptation_cfg, "head_type", "linear")
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            head_type=head_type,
        )

        # Log parameter counts
        self._log_param_counts()

    def _load_backbone(self) -> nn.Module:
        """Load the pretrained foundation model backbone.

        Returns:
            The loaded backbone module.
        """
        family = self.model_cfg.family
        source = self.model_cfg.source

        if family == "dinov2":
            return self._load_dinov2()
        elif family == "dinov3":
            return self._load_dinov3()
        elif family == "clip":
            return self._load_clip()
        elif family == "mae":
            return self._load_mae()
        else:
            raise ValueError(f"Unsupported model family: {family}")

    def _load_dinov2(self) -> nn.Module:
        """Load DINOv2 model from torch hub or HuggingFace."""
        try:
            hub_model = self.model_cfg.hub_model
            logger.info("Loading DINOv2 from torch.hub: %s", hub_model)
            model = torch.hub.load(
                self.model_cfg.hub_repo,
                hub_model,
                pretrained=True,
            )
            return model
        except Exception as e:
            logger.warning("torch.hub load failed (%s), trying HuggingFace", e)
            return self._load_dinov2_hf()

    def _load_dinov2_hf(self) -> nn.Module:
        """Load DINOv2 from HuggingFace transformers."""
        from transformers import Dinov2Model

        model_id = self.model_cfg.hf_model_id
        logger.info("Loading DINOv2 from HuggingFace: %s", model_id)
        model = Dinov2Model.from_pretrained(model_id)
        return model

    def _load_dinov3(self) -> nn.Module:
        """Load DINOv3 model from torch hub or HuggingFace."""
        try:
            hub_model = self.model_cfg.hub_model
            logger.info("Loading DINOv3 from torch.hub: %s", hub_model)
            model = torch.hub.load(
                self.model_cfg.hub_repo,
                hub_model,
                pretrained=True,
            )
            return model
        except Exception as e:
            logger.warning("torch.hub load failed (%s), trying HuggingFace", e)
            return self._load_dinov3_hf()

    def _load_dinov3_hf(self) -> nn.Module:
        """Load DINOv3 from HuggingFace."""
        try:
            from transformers import AutoModel

            model_id = self.model_cfg.hf_model_id
            logger.info("Loading DINOv3 from HuggingFace: %s", model_id)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            return model
        except Exception as e:
            logger.error("Failed to load DINOv3: %s", e)
            raise

    def _load_clip(self) -> nn.Module:
        """Load CLIP visual encoder from open_clip."""
        import open_clip

        model_name = self.model_cfg.clip_model_name
        pretrained = self.model_cfg.clip_pretrained
        logger.info("Loading CLIP: %s (pretrained=%s)", model_name, pretrained)

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Store full model for zero-shot use, but keep it off the module graph
        # so its text encoder params aren't counted as trainable or included
        # in the optimizer (which would waste memory and apply weight decay).
        object.__setattr__(self, '_clip_full_model', model)

        # Return just the visual encoder
        visual = model.visual
        return visual

    def _load_mae(self) -> nn.Module:
        """Load MAE (Masked Autoencoder) from HuggingFace."""
        from transformers import ViTMAEModel

        model_id = self.model_cfg.hf_model_id
        logger.info("Loading MAE from HuggingFace: %s", model_id)
        model = ViTMAEModel.from_pretrained(model_id)
        return model

    def _apply_adaptation(self) -> None:
        """Apply the specified adaptation strategy to the backbone."""
        adapt_type = self.adaptation_cfg.type

        if adapt_type == "linear_probe":
            self._freeze_backbone()
        elif adapt_type == "lora":
            self._freeze_backbone()
            self._inject_lora()
        elif adapt_type == "vpt":
            self._freeze_backbone()
            self._inject_vpt()
        elif adapt_type == "full_finetune":
            # All parameters trainable
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown adaptation type: {adapt_type}")

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen: all parameters set to requires_grad=False")

    def _inject_lora(self) -> None:
        """Inject LoRA adapters into the backbone."""
        from models.lora_adapter import inject_lora

        rank = self.adaptation_cfg.lora_rank
        alpha = self.adaptation_cfg.lora_alpha
        dropout = self.adaptation_cfg.lora_dropout
        target_modules = self.model_cfg.lora_target_modules

        self.backbone = inject_lora(
            model=self.backbone,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            model_family=self.model_cfg.family,
        )

    def _inject_vpt(self) -> None:
        """Inject Visual Prompt Tuning tokens."""
        from models.vpt_adapter import VPTDeep

        num_tokens = self.adaptation_cfg.num_prompt_tokens
        prompt_dropout = getattr(self.adaptation_cfg, "prompt_dropout", 0.1)
        embed_dim = self.model_cfg.embed_dim
        num_layers = self.model_cfg.num_layers

        self.vpt = VPTDeep(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_tokens=num_tokens,
            dropout=prompt_dropout,
        )
        logger.info(
            "VPT-Deep injected: %d tokens x %d layers = %d prompt params",
            num_tokens,
            num_layers,
            num_tokens * num_layers * embed_dim,
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone.

        Handles different backbone architectures (HuggingFace, torch.hub, etc.)
        and returns a feature vector per image.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Feature tensor [B, feature_dim].
        """
        family = self.model_cfg.family
        extraction = self.model_cfg.feature_extraction

        if family == "dinov2":
            return self._extract_dinov2_features(x, extraction)
        elif family == "dinov3":
            return self._extract_dinov3_features(x, extraction)
        elif family == "clip":
            return self._extract_clip_features(x)
        elif family == "mae":
            return self._extract_mae_features(x, extraction)
        else:
            raise ValueError(f"Unsupported family: {family}")

    def _extract_dinov2_features(
        self, x: torch.Tensor, extraction: str
    ) -> torch.Tensor:
        """Extract features from DINOv2 backbone."""
        # Check if it's a HuggingFace model
        if hasattr(self.backbone, "config"):
            outputs = self.backbone(x)
            if extraction == "cls_token":
                return outputs.last_hidden_state[:, 0]
            else:
                return outputs.last_hidden_state[:, 1:].mean(dim=1)
        else:
            # torch.hub model
            if hasattr(self, "vpt"):
                return self._forward_with_vpt(x)

            if extraction == "cls_token":
                features = self.backbone(x)
                if isinstance(features, dict):
                    features = features.get(
                        "x_norm_clstoken", features.get("x", None)
                    )
                return features
            else:
                features = self.backbone.forward_features(x)
                if isinstance(features, dict):
                    patch_tokens = features.get(
                        "x_norm_patchtokens", features.get("x", None)
                    )
                    return patch_tokens.mean(dim=1)
                return features[:, 1:].mean(dim=1)

    def _extract_dinov3_features(
        self, x: torch.Tensor, extraction: str
    ) -> torch.Tensor:
        """Extract features from DINOv3 backbone."""
        # Similar to DINOv2 but may have different API
        if hasattr(self.backbone, "config"):
            outputs = self.backbone(x)
            if extraction == "cls_token":
                return outputs.last_hidden_state[:, 0]
            else:
                return outputs.last_hidden_state[:, 1:].mean(dim=1)
        else:
            if hasattr(self, "vpt"):
                return self._forward_with_vpt(x)

            features = self.backbone(x)
            if isinstance(features, dict):
                return features.get("x_norm_clstoken", features.get("x", None))
            return features

    def _extract_clip_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from CLIP visual encoder."""
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def _extract_mae_features(
        self, x: torch.Tensor, extraction: str
    ) -> torch.Tensor:
        """Extract features from MAE backbone."""
        outputs = self.backbone(x)
        if extraction == "cls_token":
            return outputs.last_hidden_state[:, 0]
        else:
            return outputs.last_hidden_state.mean(dim=1)

    def _forward_with_vpt(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Visual Prompt Tuning.

        Injects learnable prompt tokens at each transformer layer.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Feature tensor [B, feature_dim].
        """
        # Get patch embeddings
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            # DINOv2 torch.hub API
            tokens = self.backbone.prepare_tokens_with_masks(x)
        elif hasattr(self.backbone, "patch_embed"):
            tokens = self.backbone.patch_embed(x)
            if hasattr(self.backbone, "cls_token"):
                cls_token = self.backbone.cls_token.expand(tokens.shape[0], -1, -1)
                tokens = torch.cat([cls_token, tokens], dim=1)
            if hasattr(self.backbone, "pos_embed"):
                tokens = tokens + self.backbone.pos_embed
        else:
            raise RuntimeError("Cannot extract patch embeddings for VPT injection")

        # Forward through transformer blocks with prompt injection
        blocks = None
        if hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
        elif hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "layer"
        ):
            blocks = self.backbone.encoder.layer

        if blocks is None:
            raise RuntimeError("Cannot find transformer blocks for VPT")

        for layer_idx, block in enumerate(blocks):
            tokens = self.vpt.inject_prompts(tokens, layer_idx)
            tokens = block(tokens)
            tokens = self.vpt.remove_prompts(tokens, layer_idx)

        # Extract CLS token
        if hasattr(self.backbone, "norm"):
            tokens = self.backbone.norm(tokens)

        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone features -> classification head.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        features = self._extract_features(x)
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head (for analysis).

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Feature tensor [B, feature_dim].
        """
        return self._extract_features(x)

    def clip_zero_shot(
        self,
        x: torch.Tensor,
        class_names: List[str],
        templates: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Perform CLIP zero-shot classification.

        Args:
            x: Input image tensor [B, C, H, W].
            class_names: List of class names for text prompts.
            templates: Text prompt templates with {species} placeholder.

        Returns:
            Similarity logits [B, num_classes].
        """
        if self.model_cfg.family != "clip":
            raise ValueError("Zero-shot only supported for CLIP models")

        import open_clip

        if templates is None:
            templates = [
                "a photo of a {species} fish",
                "an underwater photo of a {species}",
            ]

        # Get image features
        image_features = self._extract_features(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get text features (move full model to device since it's off the module graph)
        tokenizer = open_clip.get_tokenizer(self.model_cfg.clip_model_name)
        self._clip_full_model = self._clip_full_model.to(x.device)
        text_features_list = []

        for class_name in class_names:
            texts = [t.format(species=class_name) for t in templates]
            text_tokens = tokenizer(texts).to(x.device)

            with torch.no_grad():
                text_feats = self._clip_full_model.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                text_feats = text_feats.mean(dim=0)
                text_feats = text_feats / text_feats.norm()

            text_features_list.append(text_feats)

        text_features = torch.stack(text_features_list)  # [num_classes, dim]

        # Compute similarity
        logits = image_features @ text_features.T * 100.0  # Temperature scaling

        return logits

    def _log_param_counts(self) -> None:
        """Log total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        logger.info(
            "Model parameters: total=%s, trainable=%s (%.2f%%), frozen=%s",
            f"{total:,}",
            f"{trainable:,}",
            100.0 * trainable / total if total > 0 else 0,
            f"{frozen:,}",
        )

        self._param_counts = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
        }

    @property
    def param_counts(self) -> Dict[str, Any]:
        """Return parameter count statistics."""
        return self._param_counts

    def merge_lora_weights(self) -> None:
        """Merge LoRA weights into the base model for efficient inference."""
        if self.adaptation_cfg.type != "lora":
            logger.warning("merge_lora_weights called but adaptation is not LoRA")
            return

        from models.lora_adapter import merge_lora

        self.backbone = merge_lora(self.backbone)
        logger.info("LoRA weights merged into backbone")
