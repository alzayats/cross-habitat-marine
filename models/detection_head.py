"""Lightweight detection head for COTS GBR and Brackish detection experiments.

Provides a simple feature pyramid + detection head that can be attached
to foundation model features for bounding box prediction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimpleDetectionHead(nn.Module):
    """Lightweight detection head for ViT-based feature maps.

    Converts ViT patch tokens into a spatial feature map and applies
    a simple anchor-free detection head for bounding box prediction.

    Args:
        in_features: Feature dimension from backbone.
        num_classes: Number of object classes.
        patch_size: Patch size of the ViT (14 for DINOv2, 16 for others).
        hidden_dim: Hidden dimension for detection layers.
        num_anchors: Number of anchor boxes per spatial position.
    """

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 30,
        patch_size: int = 14,
        hidden_dim: int = 256,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_anchors = num_anchors

        # Project ViT features to detection feature space
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * num_classes, 1),
        )

        # Regression branch (4 coordinates per anchor)
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * 4, 1),
        )

        # Objectness branch
        self.obj_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize detection head weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _reshape_to_spatial(
        self, patch_tokens: torch.Tensor, image_size: int = 224
    ) -> torch.Tensor:
        """Reshape ViT patch tokens to a spatial feature map.

        Args:
            patch_tokens: Patch token features [B, N, D] (excluding CLS).
            image_size: Original input image size.

        Returns:
            Spatial feature map [B, D, H, W].
        """
        B, N, D = patch_tokens.shape
        h = w = int(N ** 0.5)

        spatial = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        return spatial

    def forward(
        self,
        patch_tokens: torch.Tensor,
        image_size: int = 224,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for detection.

        Args:
            patch_tokens: ViT patch tokens [B, N, D] (CLS token excluded).
            image_size: Original input image size.

        Returns:
            Dictionary with:
                - 'cls_logits': [B, num_anchors * num_classes, H, W]
                - 'bbox_pred': [B, num_anchors * 4, H, W]
                - 'objectness': [B, num_anchors, H, W]
        """
        # Project features
        features = self.feature_proj(patch_tokens)

        # Reshape to spatial
        spatial = self._reshape_to_spatial(features, image_size)

        # Detection heads
        cls_logits = self.cls_head(spatial)
        bbox_pred = self.reg_head(spatial)
        objectness = self.obj_head(spatial)

        return {
            "cls_logits": cls_logits,
            "bbox_pred": bbox_pred,
            "objectness": objectness,
        }

    def decode_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        image_size: int = 224,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode raw predictions into bounding boxes.

        Args:
            outputs: Raw detection head outputs.
            image_size: Original image size for coordinate scaling.
            conf_threshold: Confidence threshold for filtering.
            nms_threshold: NMS IoU threshold.

        Returns:
            List of dicts per image: {'boxes', 'scores', 'labels'}.
        """
        from torchvision.ops import nms

        cls_logits = outputs["cls_logits"]
        bbox_pred = outputs["bbox_pred"]
        objectness = outputs["objectness"]

        B = cls_logits.shape[0]
        results: List[Dict[str, torch.Tensor]] = []

        for b in range(B):
            # Get objectness scores
            obj_scores = objectness[b].sigmoid()  # [A, H, W]
            A, H, W = obj_scores.shape

            # Flatten spatial dimensions
            obj_flat = obj_scores.reshape(A, -1).transpose(0, 1)  # [H*W, A]

            # Class predictions
            cls_flat = cls_logits[b].reshape(A * self.num_classes, -1).transpose(0, 1)
            cls_flat = cls_flat.reshape(-1, A, self.num_classes)  # [H*W, A, C]

            # Box predictions
            box_flat = bbox_pred[b].reshape(A * 4, -1).transpose(0, 1)
            box_flat = box_flat.reshape(-1, A, 4)  # [H*W, A, 4]

            # Grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=cls_logits.device, dtype=torch.float32),
                torch.arange(W, device=cls_logits.device, dtype=torch.float32),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [H*W, 2]

            # Decode boxes (center offset + size)
            scale = image_size / H
            all_boxes = []
            all_scores = []
            all_labels = []

            for a in range(A):
                cx = (grid[:, 0] + box_flat[:, a, 0].sigmoid()) * scale
                cy = (grid[:, 1] + box_flat[:, a, 1].sigmoid()) * scale
                w = box_flat[:, a, 2].exp() * scale
                h = box_flat[:, a, 3].exp() * scale

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [H*W, 4]

                # Combine objectness and class scores
                cls_scores = cls_flat[:, a].softmax(dim=-1)  # [H*W, C]
                combined = obj_flat[:, a:a+1] * cls_scores   # [H*W, C]

                max_scores, max_labels = combined.max(dim=-1)

                # Filter by confidence
                mask = max_scores > conf_threshold
                if mask.sum() > 0:
                    all_boxes.append(boxes[mask])
                    all_scores.append(max_scores[mask])
                    all_labels.append(max_labels[mask])

            if all_boxes:
                boxes_cat = torch.cat(all_boxes)
                scores_cat = torch.cat(all_scores)
                labels_cat = torch.cat(all_labels)

                # Apply NMS
                keep = nms(boxes_cat, scores_cat, nms_threshold)
                results.append(
                    {
                        "boxes": boxes_cat[keep],
                        "scores": scores_cat[keep],
                        "labels": labels_cat[keep],
                    }
                )
            else:
                results.append(
                    {
                        "boxes": torch.zeros(0, 4, device=cls_logits.device),
                        "scores": torch.zeros(0, device=cls_logits.device),
                        "labels": torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    }
                )

        return results
