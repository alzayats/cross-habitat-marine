"""Few-shot sampling with stratified class-balanced selection.

Provides deterministic k-shot sampling for few-shot adaptation experiments.
Supports stratified sampling, handling class imbalance, and multiple trials.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FewShotSampler:
    """Stratified few-shot sampler for marine datasets.

    Samples exactly k examples per class from a target dataset, ensuring
    balanced class representation and deterministic results given a seed.

    Args:
        k_values: List of shot values to sample. Default: [0, 1, 2, 5, 10, 20, 50, 100].
        n_trials: Number of random sampling trials per k value.
        base_seed: Base random seed for reproducibility.
    """

    DEFAULT_K_VALUES = [0, 1, 2, 5, 10, 20, 50, 100]

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        n_trials: int = 5,
        base_seed: int = 42,
    ) -> None:
        self.k_values = k_values or self.DEFAULT_K_VALUES
        self.n_trials = n_trials
        self.base_seed = base_seed

    def sample(
        self,
        labels: np.ndarray,
        k: int,
        trial: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample k examples per class from the dataset.

        Args:
            labels: Array of integer labels for all samples.
            k: Number of examples to sample per class.
            trial: Trial index for seed variation.

        Returns:
            Tuple of (support_indices, query_indices) as numpy arrays.
            support_indices: Indices of the k-shot support set.
            query_indices: Remaining indices for evaluation.
        """
        if k == 0:
            return np.array([], dtype=np.int64), np.arange(len(labels))

        seed = self.base_seed + trial * 1000 + k
        rng = np.random.RandomState(seed)

        unique_classes = np.unique(labels)
        support_indices: List[int] = []
        all_indices = np.arange(len(labels))

        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]

            if len(cls_indices) <= k:
                # Class has fewer than k samples; use all available
                logger.warning(
                    "Class %d has only %d samples (requested k=%d). Using all.",
                    cls,
                    len(cls_indices),
                    k,
                )
                selected = cls_indices
            else:
                selected = rng.choice(cls_indices, size=k, replace=False)

            support_indices.extend(selected.tolist())

        support_set = np.array(support_indices, dtype=np.int64)
        query_set = np.setdiff1d(all_indices, support_set)

        logger.info(
            "Few-shot sampling: k=%d, trial=%d, support=%d, query=%d, "
            "classes=%d",
            k,
            trial,
            len(support_set),
            len(query_set),
            len(unique_classes),
        )

        return support_set, query_set

    def sample_all_trials(
        self,
        labels: np.ndarray,
        k: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample k-shot support sets across all trials.

        Args:
            labels: Array of integer labels for all samples.
            k: Number of examples per class.

        Returns:
            List of (support_indices, query_indices) tuples, one per trial.
        """
        return [self.sample(labels, k, trial) for trial in range(self.n_trials)]

    def sample_all_k_values(
        self,
        labels: np.ndarray,
    ) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
        """Sample for all k values and all trials.

        Args:
            labels: Array of integer labels for all samples.

        Returns:
            Dictionary mapping k -> list of (support, query) index tuples.
        """
        results: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for k in self.k_values:
            results[k] = self.sample_all_trials(labels, k)
        return results

    def get_class_counts_in_support(
        self,
        labels: np.ndarray,
        support_indices: np.ndarray,
    ) -> Dict[int, int]:
        """Count samples per class in the support set.

        Args:
            labels: Full label array.
            support_indices: Indices of support set samples.

        Returns:
            Dictionary mapping class index to sample count.
        """
        support_labels = labels[support_indices]
        unique, counts = np.unique(support_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @staticmethod
    def verify_determinism(
        labels: np.ndarray,
        k: int,
        seed: int = 42,
    ) -> bool:
        """Verify that sampling is deterministic for given seed.

        Args:
            labels: Label array.
            k: Shot value.
            seed: Random seed.

        Returns:
            True if two runs produce identical results.
        """
        sampler1 = FewShotSampler(base_seed=seed)
        sampler2 = FewShotSampler(base_seed=seed)

        s1, q1 = sampler1.sample(labels, k, trial=0)
        s2, q2 = sampler2.sample(labels, k, trial=0)

        return np.array_equal(s1, s2) and np.array_equal(q1, q2)
