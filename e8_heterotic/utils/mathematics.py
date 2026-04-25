"""
Mathematical utilities for root system calculations.

The clustering helpers have moved to :mod:`e8_heterotic.core.clustering`.
The remaining helpers here are simple geometric utilities used outside the
clustering pipeline.
"""

import logging
import warnings
from typing import Optional

import numpy as np

from e8_heterotic.core.constants import (
    E8_ROOT_NORM,
    INNER_PRODUCT_TOLERANCE,
    NORM_TOLERANCE,
)

logger = logging.getLogger(__name__)


def normalize_vector(v: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Return ``v`` rescaled to unit length (or zero if it was zero)."""
    if dtype is None:
        dtype = v.dtype

    norm = np.linalg.norm(v)
    if norm < NORM_TOLERANCE:
        return np.zeros_like(v, dtype=dtype)

    return (v / norm).astype(dtype)


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between ``v1`` and ``v2`` in degrees."""
    v1n = normalize_vector(v1)
    v2n = normalize_vector(v2)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def validate_root_norms(
    roots: np.ndarray,
    expected_norm: float = E8_ROOT_NORM,
    tolerance: float = NORM_TOLERANCE,
) -> dict:
    """Check that all rows of ``roots`` have norm ``expected_norm``."""
    if len(roots) == 0:
        return {"valid": False, "message": "No roots provided"}

    norms = np.linalg.norm(roots, axis=1)
    norm_diffs = np.abs(norms - expected_norm)
    max_diff = float(np.max(norm_diffs))

    if max_diff >= tolerance:
        logger.warning(
            "Root-norm validation: max deviation %g exceeds tolerance %g",
            max_diff, tolerance,
        )

    return {
        "valid": max_diff < tolerance,
        "expected_norm": expected_norm,
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_norm": float(np.mean(norms)),
        "max_deviation": max_diff,
        "tolerance": tolerance,
    }


def validate_dot_products(
    roots: np.ndarray,
    tolerance: float = INNER_PRODUCT_TOLERANCE,
) -> dict:
    """Check that pairwise inner products take values in ``{-2,-1,0,1,2}``."""
    if len(roots) == 0:
        return {"valid": False, "message": "No roots provided"}

    g = roots @ roots.T
    iu = np.triu_indices(len(roots), k=1)
    values = g[iu]
    rounded = np.round(values, 6)
    unique = np.unique(rounded)

    expected = {-2.0, -1.0, 0.0, 1.0, 2.0}
    unexpected = sorted(v for v in unique.tolist() if v not in expected)

    if unexpected:
        logger.warning(
            "Inner-product validation: unexpected values %s", unexpected[:10]
        )

    return {
        "valid": not unexpected,
        "unique_dot_products": unique.tolist(),
        "expected_dot_products": sorted(expected),
        "unexpected": unexpected,
        "tolerance": tolerance,
    }


def count_triangles_and_triplets(adjacency_matrix: np.ndarray):
    """Deprecated. Use :func:`e8_heterotic.core.clustering.count_triangles_and_wedges`."""
    warnings.warn(
        "count_triangles_and_triplets is deprecated; use "
        "e8_heterotic.core.clustering.count_triangles_and_wedges instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from e8_heterotic.core.clustering import count_triangles_and_wedges
    return count_triangles_and_wedges(adjacency_matrix)


def calculate_clustering_coefficient(adjacency_matrix: np.ndarray) -> float:
    """Deprecated. Use :func:`e8_heterotic.core.clustering.global_clustering_coefficient`."""
    warnings.warn(
        "calculate_clustering_coefficient is deprecated; use "
        "e8_heterotic.core.clustering.global_clustering_coefficient or "
        "mean_local_clustering_coefficient instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from e8_heterotic.core.clustering import mean_local_clustering_coefficient
    return mean_local_clustering_coefficient(adjacency_matrix)
