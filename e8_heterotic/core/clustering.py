"""
Honest counting of triangles, wedges, and clustering coefficients.

No hardcoded returns. No "validation against theoretical predictions" that
overrides a computed result. Clustering values flow from the adjacency
matrix directly.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _as_bool_symmetric(adjacency: np.ndarray) -> np.ndarray:
    if not isinstance(adjacency, np.ndarray):
        raise ValueError(
            f"adjacency must be a numpy.ndarray; got {type(adjacency)!r}"
        )
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must be a square 2-D array; got shape {adjacency.shape}"
        )
    a = adjacency.astype(bool, copy=False)
    if not np.array_equal(a, a.T):
        raise ValueError("adjacency must be symmetric")
    if a.diagonal().any():
        raise ValueError("adjacency diagonal must be all-False")
    return a


def count_triangles_and_wedges(adjacency: np.ndarray) -> tuple[int, int]:
    """Return ``(num_triangles, num_wedges)``.

    ``num_triangles`` is the number of unordered triples ``{i, j, k}`` whose
    three pairs are all adjacent. Equivalently ``trace(A^3) / 6``.

    ``num_wedges`` is the number of ordered pairs of edges sharing a centre
    vertex, ``sum_v C(deg(v), 2)``. (This is the centred-triple count used
    in the standard transitivity definition; it counts each triangle's
    three internal wedges.)
    """
    a = _as_bool_symmetric(adjacency)
    a_int = a.astype(np.int64)

    a2 = a_int @ a_int
    a3_diag = np.einsum("ij,ji->i", a2, a_int)
    triangles_sum = int(a3_diag.sum())
    if triangles_sum % 6 != 0:
        logger.warning(
            "trace(A^3) = %d is not divisible by 6 — adjacency likely "
            "asymmetric or has self-loops",
            triangles_sum,
        )
    num_triangles = triangles_sum // 6

    degrees = a_int.sum(axis=1)
    num_wedges = int(np.sum(degrees * (degrees - 1)) // 2)

    return num_triangles, num_wedges


def degree_distribution(adjacency: np.ndarray) -> np.ndarray:
    """Return vertex degrees in descending order."""
    a = _as_bool_symmetric(adjacency)
    degrees = a.astype(np.int64).sum(axis=1)
    return np.sort(degrees)[::-1]


def global_clustering_coefficient(adjacency: np.ndarray) -> float:
    """Return ``3 * num_triangles / num_wedges`` (transitivity).

    Returns 0.0 for graphs with no wedges and logs a WARNING.
    """
    num_triangles, num_wedges = count_triangles_and_wedges(adjacency)
    if num_wedges == 0:
        logger.warning(
            "Graph has no wedges (no vertex has degree ≥ 2); "
            "global clustering is undefined — returning 0.0"
        )
        return 0.0
    return 3.0 * num_triangles / num_wedges


def mean_local_clustering_coefficient(adjacency: np.ndarray) -> float:
    """Return the mean of per-vertex local clustering coefficients.

    Per vertex ``v`` with degree ``d_v ≥ 2``::

        C_local(v) = (number of triangles through v) / C(d_v, 2)

    Vertices with degree < 2 are excluded from the mean. For a vertex-
    transitive graph this equals the global clustering coefficient.
    """
    a = _as_bool_symmetric(adjacency)
    a_int = a.astype(np.int64)
    a2 = a_int @ a_int

    triangles_through_v = np.einsum("ij,ji->i", a2, a_int) // 2
    degrees = a_int.sum(axis=1)
    mask = degrees >= 2
    excluded = int((~mask).sum())
    if excluded:
        logger.info(
            "mean_local_clustering: excluded %d vertices with degree < 2",
            excluded,
        )
    if not mask.any():
        logger.warning(
            "Graph has no vertex with degree ≥ 2; mean local clustering "
            "is undefined — returning 0.0"
        )
        return 0.0
    pair_counts = degrees[mask] * (degrees[mask] - 1) // 2
    locals_ = triangles_through_v[mask] / pair_counts
    return float(locals_.mean())


def count_triangles_networkx(adjacency: np.ndarray) -> int:
    """Independent triangle count via NetworkX, used for cross-checking.

    Imports NetworkX lazily so the module remains importable in environments
    that lack it.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "networkx is required for count_triangles_networkx"
        ) from exc

    a = _as_bool_symmetric(adjacency)
    g = nx.from_numpy_array(a.astype(np.uint8))
    return sum(nx.triangles(g).values()) // 3


def num_connected_components(adjacency: np.ndarray) -> int:
    """Return the number of connected components."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "networkx is required for num_connected_components"
        ) from exc

    a = _as_bool_symmetric(adjacency)
    g = nx.from_numpy_array(a.astype(np.uint8))
    return nx.number_connected_components(g)
