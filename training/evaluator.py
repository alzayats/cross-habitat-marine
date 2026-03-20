"""Comprehensive evaluation metrics for classification and detection.

Computes all metrics required for the paper: accuracy, F1, balanced accuracy,
confusion matrix, Cohen's Kappa, per-class metrics, and detection mAP.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation metrics computation for classification and detection.

    Args:
        num_classes: Number of classes.
        class_names: Optional list of class names.
        device: Computation device.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        save_predictions: bool = True,
        use_amp: bool = True,
        num_eval_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on a dataloader and compute all metrics.

        Args:
            model: Trained model.
            dataloader: Evaluation data loader.
            save_predictions: Whether to include raw predictions in results.
            use_amp: Use mixed precision during evaluation to reduce GPU memory.
            num_eval_classes: If set, restrict logits to the first N classes
                before argmax (used when head has source+target classes but
                evaluation is target-only).

        Returns:
            Dictionary of all computed metrics.
        """
        model.eval()
        use_amp = use_amp and torch.cuda.is_available()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[np.ndarray] = []
        all_paths: List[str] = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"]

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
            if num_eval_classes is not None:
                logits = logits[:, :num_eval_classes]
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs)

            if "image_path" in batch:
                all_paths.extend(batch["image_path"])

            # Free GPU memory immediately
            del images, logits
            torch.cuda.empty_cache()

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        metrics = self._compute_classification_metrics(y_true, y_pred, y_probs)

        if save_predictions:
            metrics["predictions"] = {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_probs": y_probs.tolist(),
                "image_paths": all_paths,
            }

        return metrics

    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_probs: Prediction probabilities.

        Returns:
            Dictionary of metrics.
        """
        metrics: Dict[str, Any] = {}

        # Top-1 accuracy
        metrics["top1_accuracy"] = float(accuracy_score(y_true, y_pred))

        # Top-5 accuracy (if enough classes)
        if self.num_classes >= 5:
            try:
                metrics["top5_accuracy"] = float(
                    top_k_accuracy_score(y_true, y_probs, k=5, labels=range(self.num_classes))
                )
            except ValueError:
                metrics["top5_accuracy"] = None

        # Balanced accuracy
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        # F1 scores
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Cohen's Kappa
        metrics["cohens_kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics["per_class"] = {}
        for i, name in enumerate(self.class_names):
            if i < len(precision):
                metrics["per_class"][name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        metrics["confusion_matrix"] = cm.tolist()

        # Sample counts
        metrics["n_samples"] = len(y_true)
        metrics["n_correct"] = int((y_true == y_pred).sum())

        logger.info(
            "Evaluation: acc=%.4f, balanced_acc=%.4f, macro_f1=%.4f, kappa=%.4f",
            metrics["top1_accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            metrics["cohens_kappa"],
        )

        return metrics

    @staticmethod
    def aggregate_across_seeds(
        results_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute mean and std of metrics across multiple seeds/trials.

        Args:
            results_list: List of result dicts from individual runs.

        Returns:
            Aggregated metrics with mean and std.
        """
        aggregated: Dict[str, Any] = {}
        scalar_keys = [
            "top1_accuracy",
            "top5_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "cohens_kappa",
        ]

        for key in scalar_keys:
            values = [r[key] for r in results_list if r.get(key) is not None]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        aggregated["n_runs"] = len(results_list)
        return aggregated

    @staticmethod
    def compute_detection_metrics(
        predictions: List[Dict[str, torch.Tensor]],
        ground_truths: List[Dict[str, torch.Tensor]],
        iou_thresholds: Optional[List[float]] = None,
        num_classes: int = 30,
    ) -> Dict[str, float]:
        """Compute detection metrics (mAP).

        Args:
            predictions: List of prediction dicts with 'boxes', 'scores', 'labels'.
            ground_truths: List of GT dicts with 'boxes', 'labels'.
            iou_thresholds: IoU thresholds for mAP computation.
            num_classes: Number of object classes.

        Returns:
            Detection metrics including mAP@50 and mAP@50:95.
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50:0.95

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            return _compute_coco_metrics(
                predictions, ground_truths, iou_thresholds, num_classes
            )
        except ImportError:
            logger.warning("pycocotools not available. Computing simple mAP.")
            return _compute_simple_map(
                predictions, ground_truths, iou_thresholds, num_classes
            )

    def save_results(
        self,
        metrics: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save evaluation results to JSON.

        Args:
            metrics: Computed metrics dictionary.
            output_path: Path for the output JSON file.
            metadata: Optional experiment metadata to include.
        """
        output = {"metrics": metrics}
        if metadata:
            output["metadata"] = metadata

        # Remove non-serializable items
        clean = _make_json_serializable(output)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)

        logger.info("Results saved to %s", output_path)


def _compute_simple_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float],
    num_classes: int,
) -> Dict[str, float]:
    """Simple mAP computation without pycocotools."""
    metrics: Dict[str, float] = {}

    def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
        y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
        x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
        y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        return inter / (union + 1e-6)

    all_aps = {t: [] for t in iou_thresholds}

    for cls_id in range(num_classes):
        for iou_thresh in iou_thresholds:
            tp_list = []
            scores_list = []
            n_gt = 0

            for pred, gt in zip(predictions, ground_truths):
                pred_mask = pred["labels"] == cls_id
                gt_mask = gt["labels"] == cls_id

                n_gt += gt_mask.sum().item()

                if pred_mask.sum() == 0:
                    continue

                pred_boxes = pred["boxes"][pred_mask]
                pred_scores = pred["scores"][pred_mask]

                if gt_mask.sum() == 0:
                    tp_list.extend([0] * len(pred_boxes))
                    scores_list.extend(pred_scores.tolist())
                    continue

                gt_boxes = gt["boxes"][gt_mask]
                ious = compute_iou(pred_boxes, gt_boxes)

                sorted_idx = pred_scores.argsort(descending=True)
                matched_gt = set()

                for idx in sorted_idx:
                    max_iou, max_gt = ious[idx].max(dim=0)
                    if max_iou >= iou_thresh and max_gt.item() not in matched_gt:
                        tp_list.append(1)
                        matched_gt.add(max_gt.item())
                    else:
                        tp_list.append(0)
                    scores_list.append(pred_scores[idx].item())

            if n_gt == 0:
                continue

            # Compute AP
            sorted_pairs = sorted(zip(scores_list, tp_list), reverse=True)
            tp_cumsum = np.cumsum([p[1] for p in sorted_pairs])
            fp_cumsum = np.cumsum([1 - p[1] for p in sorted_pairs])
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / n_gt

            ap = np.trapz(precision, recall) if len(recall) > 0 else 0.0
            all_aps[iou_thresh].append(ap)

    # Aggregate
    ap50_values = all_aps.get(0.5, [])
    metrics["mAP@50"] = float(np.mean(ap50_values)) if ap50_values else 0.0

    all_thresh_aps = []
    for t in iou_thresholds:
        if all_aps[t]:
            all_thresh_aps.append(np.mean(all_aps[t]))
    metrics["mAP@50:95"] = float(np.mean(all_thresh_aps)) if all_thresh_aps else 0.0

    return metrics


def _compute_coco_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float],
    num_classes: int,
) -> Dict[str, float]:
    """Compute detection metrics using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build COCO-format datasets
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []

    for i in range(num_classes):
        coco_gt["categories"].append({"id": i, "name": str(i)})

    ann_id = 0
    for img_id, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        coco_gt["images"].append({"id": img_id, "width": 1, "height": 1})

        for j in range(len(gt["boxes"])):
            box = gt["boxes"][j]
            x1, y1, x2, y2 = box.tolist()
            coco_gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(gt["labels"][j]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        for j in range(len(pred["boxes"])):
            box = pred["boxes"][j]
            x1, y1, x2, y2 = box.tolist()
            coco_dt.append(
                {
                    "image_id": img_id,
                    "category_id": int(pred["labels"][j]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(pred["scores"][j]),
                }
            )

    coco = COCO()
    coco.dataset = coco_gt
    coco.createIndex()

    coco_eval = coco.loadRes(coco_dt) if coco_dt else coco
    evaluator = COCOeval(coco, coco_eval, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "mAP@50:95": float(evaluator.stats[0]),
        "mAP@50": float(evaluator.stats[1]),
        "mAP@75": float(evaluator.stats[2]),
    }


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
