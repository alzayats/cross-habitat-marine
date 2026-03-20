"""Computational efficiency analysis: FLOPs, parameters, memory, and time.

Computes efficiency metrics for all model + adaptation combinations
to support the Pareto efficiency analysis in the paper.

Usage:
    python -m analysis.compute_efficiency --model dinov2_base --adaptation lora_r4
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "total_params_M": total / 1e6,
        "trainable_params_M": trainable / 1e6,
    }


def estimate_flops(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = "cpu",
) -> Dict[str, float]:
    """Estimate FLOPs for a forward pass.

    Uses a simple profiling approach. For more accurate results,
    use fvcore or ptflops libraries.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device.

    Returns:
        FLOPs estimation.
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        model = model.to(device)
        model.eval()
        x = torch.randn(*input_size).to(device)

        flops_analysis = FlopCountAnalysis(model, x)
        flops = flops_analysis.total()

        return {
            "flops": flops,
            "gflops": flops / 1e9,
        }
    except ImportError:
        logger.warning("fvcore not installed. Estimating FLOPs from parameter count.")
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: ~2 FLOPs per parameter per forward pass
        estimated_flops = total_params * 2
        return {
            "flops": estimated_flops,
            "gflops": estimated_flops / 1e9,
            "estimated": True,
        }


def measure_throughput(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = "cuda",
    n_warmup: int = 10,
    n_measure: int = 100,
    batch_sizes: Optional[list] = None,
) -> Dict[str, float]:
    """Measure inference throughput (images/second).

    Args:
        model: PyTorch model.
        input_size: Input shape (batch dim will be replaced).
        device: Device.
        n_warmup: Warmup iterations.
        n_measure: Measurement iterations.
        batch_sizes: List of batch sizes to test.

    Returns:
        Throughput metrics.
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]

    model = model.to(device)
    model.eval()

    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, *input_size[1:]).to(device)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(n_measure):
            start = time.perf_counter()
            with torch.no_grad():
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_time = np.mean(times)
        throughput = bs / mean_time

        results[f"batch_{bs}_throughput"] = throughput
        results[f"batch_{bs}_latency_ms"] = mean_time * 1000

    return results


def measure_gpu_memory(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure GPU memory usage during forward pass.

    Args:
        model: PyTorch model.
        input_size: Input shape.
        device: Device.

    Returns:
        Memory usage in MB.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"gpu_memory_MB": 0, "note": "CUDA not available"}

    model = model.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        model(x)

    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "gpu_memory_peak_MB": peak_memory,
        "model_memory_MB": sum(
            p.nelement() * p.element_size() for p in model.parameters()
        ) / 1024 / 1024,
    }


def compute_all_efficiency_metrics(
    model_name: str,
    adaptation_name: str,
    num_classes: int = 10,
    image_size: int = 224,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute all efficiency metrics for a model + adaptation combination.

    Args:
        model_name: Model config name.
        adaptation_name: Adaptation config name.
        num_classes: Number of classes.
        image_size: Input image size.
        device: Device.

    Returns:
        Complete efficiency metrics.
    """
    factory = ModelFactory()

    try:
        model = factory.create(model_name, adaptation_name, num_classes)
    except Exception as e:
        logger.error("Failed to create model %s + %s: %s", model_name, adaptation_name, e)
        return {"error": str(e)}

    metrics: Dict[str, Any] = {
        "model": model_name,
        "adaptation": adaptation_name,
    }

    # Parameters
    metrics.update(count_parameters(model))

    # FLOPs
    input_size = (1, 3, image_size, image_size)
    metrics.update(estimate_flops(model, input_size, "cpu"))

    # GPU metrics (if available)
    if device == "cuda" and torch.cuda.is_available():
        metrics.update(measure_gpu_memory(model, input_size, device))
        metrics.update(measure_throughput(model, input_size, device))

    logger.info(
        "%s + %s: %.1fM params (%.1fM trainable), %.1f GFLOPs",
        model_name,
        adaptation_name,
        metrics["total_params_M"],
        metrics["trainable_params_M"],
        metrics.get("gflops", 0),
    )

    del model
    torch.cuda.empty_cache()

    return metrics


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compute efficiency metrics")
    parser.add_argument("--model", type=str)
    parser.add_argument("--adaptation", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="./outputs/efficiency.json")

    args = parser.parse_args()

    import json

    if args.all:
        # Model-dependent adaptation configs (reduced design)
        model_adaptations = {
            "dinov2_base": ["linear_probe", "lora_r4", "vpt_deep"],
            "clip_base": ["linear_probe", "lora_r4", "vpt_deep"],
            "resnet50_imagenet": ["linear_probe", "full_finetune"],
            "efficientnet_b4": ["linear_probe", "full_finetune"],
        }

        all_metrics = []
        for model, adaptations in model_adaptations.items():
            for adapt in adaptations:
                metrics = compute_all_efficiency_metrics(model, adapt)
                all_metrics.append(metrics)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("All efficiency metrics saved to %s", args.output)

    elif args.model and args.adaptation:
        metrics = compute_all_efficiency_metrics(args.model, args.adaptation)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
