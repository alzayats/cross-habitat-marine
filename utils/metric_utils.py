"""Metric computation helpers for aggregation and statistical testing."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def aggregate_metrics(
    results_list: List[Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs (mean +/- std).

    Args:
        results_list: List of result dicts from individual runs.
        metric_keys: Specific metric keys to aggregate. If None, auto-detect.

    Returns:
        Dict mapping metric_name -> {'mean': ..., 'std': ..., 'n': ...}.
    """
    if not results_list:
        return {}

    if metric_keys is None:
        metric_keys = [
            "top1_accuracy",
            "top5_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "cohens_kappa",
        ]

    aggregated: Dict[str, Dict[str, float]] = {}

    for key in metric_keys:
        values = []
        for r in results_list:
            val = r.get(key)
            if val is None and "test_metrics" in r:
                val = r["test_metrics"].get(key)
            if val is not None:
                values.append(float(val))

        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "n": len(values),
            }

    return aggregated


def wilcoxon_test(
    values_a: List[float],
    values_b: List[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Perform Wilcoxon signed-rank test for paired comparisons.

    Used for pairwise model comparisons across seeds/folds.

    Args:
        values_a: Metric values for model/method A.
        values_b: Metric values for model/method B.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Test results including statistic, p-value, and significance.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"Paired test requires equal-length arrays: {len(values_a)} vs {len(values_b)}"
        )

    if len(values_a) < 5:
        logger.warning("Small sample size (%d) for Wilcoxon test", len(values_a))

    try:
        statistic, p_value = stats.wilcoxon(
            values_a, values_b, alternative=alternative
        )
    except ValueError as e:
        logger.warning("Wilcoxon test failed: %s", e)
        return {"error": str(e)}

    effect_size = _compute_rank_biserial(values_a, values_b)

    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "significant_0001": p_value < 0.001,
        "effect_size_r": effect_size,
        "n_pairs": len(values_a),
        "mean_diff": float(np.mean(np.array(values_a) - np.array(values_b))),
        "alternative": alternative,
    }


def _compute_rank_biserial(
    values_a: List[float],
    values_b: List[float],
) -> float:
    """Compute rank-biserial correlation as effect size.

    Args:
        values_a: First set of values.
        values_b: Second set of values.

    Returns:
        Effect size r in [-1, 1].
    """
    diffs = np.array(values_a) - np.array(values_b)
    diffs = diffs[diffs != 0]

    if len(diffs) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(diffs))
    n = len(diffs)

    r_plus = np.sum(ranks[diffs > 0])
    r_minus = np.sum(ranks[diffs < 0])

    r = (r_plus - r_minus) / (n * (n + 1) / 2)
    return float(r)


def confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for the mean.

    Args:
        values: List of metric values.
        confidence: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    n = len(values)
    if n < 2:
        mean = np.mean(values)
        return (mean, mean)

    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return (float(mean - h), float(mean + h))


def format_metric_table(
    results: Dict[str, Dict[str, float]],
    format_str: str = "{mean:.3f} +/- {std:.3f}",
) -> str:
    """Format aggregated metrics as a readable table.

    Args:
        results: Aggregated metrics from aggregate_metrics().
        format_str: Format string for each metric.

    Returns:
        Formatted string table.
    """
    lines = []
    for key, values in sorted(results.items()):
        formatted = format_str.format(**values)
        lines.append(f"  {key}: {formatted} (n={values['n']})")

    return "\n".join(lines)
