"""
Adjacency conventions for a root system.

Every adjacency rule here is determined by the inner product matrix alone,
applied uniformly to all pairs. There are no special-case cross-couplings.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _validate_roots(roots: np.ndarray) -> None:
    if not isinstance(roots, np.ndarray):
        raise ValueError(f"roots must be a numpy.ndarray; got {type(roots)!r}")
    if roots.ndim != 2:
        raise ValueError(
            f"roots must be 2-D (N, D); got shape {roots.shape}"
        )
    if roots.shape[0] < 2:
        raise ValueError(
            f"roots must contain at least two vectors; got {roots.shape[0]}"
        )


def _gram(roots: np.ndarray) -> np.ndarray:
    return roots @ roots.T


def _finalize(adj: np.ndarray, name: str) -> np.ndarray:
    np.fill_diagonal(adj, False)
    n_edges = int(adj.sum()) // 2
    logger.debug("Adjacency [%s]: %d edges over %d vertices",
                 name, n_edges, adj.shape[0])
    return adj


def adjacency_inner_product_one(
    roots: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Convention A — the canonical root-system graph: ⟨α, β⟩ = +1.

    For E8 this yields the 56-regular graph on 240 vertices that appears
    throughout Coxeter / root-system literature.
    """
    _validate_roots(roots)
    g = _gram(roots)
    adj = np.abs(g - 1.0) < tolerance
    return _finalize(adj, "A: ⟨α,β⟩ = +1")


def adjacency_inner_product_nonzero(
    roots: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Convention B — non-orthogonal roots: ⟨α, β⟩ ≠ 0, α ≠ β.

    Includes ⟨α, β⟩ ∈ {±1, ±2}. For E8, this yields the 113-regular graph.
    """
    _validate_roots(roots)
    g = _gram(roots)
    adj = np.abs(g) > tolerance
    return _finalize(adj, "B: ⟨α,β⟩ ≠ 0")


def adjacency_absolute_inner_product_one(
    roots: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Convention C — |⟨α, β⟩| = 1.

    Symmetric in α ↔ -α. For E8, yields the 112-regular graph.
    """
    _validate_roots(roots)
    g = _gram(roots)
    adj = np.abs(np.abs(g) - 1.0) < tolerance
    return _finalize(adj, "C: |⟨α,β⟩| = 1")


def adjacency_inner_product_minus_one(
    roots: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Convention D — ⟨α, β⟩ = -1.

    For E8, this is also a 56-regular graph (each root has 56 neighbours
    at inner product -1), but it is NOT isomorphic to Convention A: the map
    α → -α sends ⟨α,β⟩ = +1 triples to ⟨α,β⟩ = +1 triples, not to
    ⟨α,β⟩ = -1 triples, so the triangle structures differ. Convention D is
    included so that this asymmetry can be observed directly rather than
    assumed.
    """
    _validate_roots(roots)
    g = _gram(roots)
    adj = np.abs(g + 1.0) < tolerance
    return _finalize(adj, "D: ⟨α,β⟩ = -1")


CONVENTIONS = {
    "A": ("⟨α,β⟩ = +1", adjacency_inner_product_one),
    "B": ("⟨α,β⟩ ≠ 0", adjacency_inner_product_nonzero),
    "C": ("|⟨α,β⟩| = 1", adjacency_absolute_inner_product_one),
    "D": ("⟨α,β⟩ = -1", adjacency_inner_product_minus_one),
}
