"""Tests for triangle counting and clustering on the E8 / E8 ⊕ E8 graphs."""

import numpy as np
import pytest

from e8_heterotic.core.adjacency import (
    adjacency_absolute_inner_product_one,
    adjacency_inner_product_minus_one,
    adjacency_inner_product_nonzero,
    adjacency_inner_product_one,
)
from e8_heterotic.core.clustering import (
    count_triangles_and_wedges,
    count_triangles_networkx,
    global_clustering_coefficient,
    mean_local_clustering_coefficient,
)
from e8_heterotic.core.root_system import (
    construct_e8_roots,
    construct_e8xe8_roots,
)


def _complete_graph(n: int) -> np.ndarray:
    a = np.ones((n, n), dtype=bool)
    np.fill_diagonal(a, False)
    return a


def _cycle_graph(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a[i, (i + 1) % n] = True
        a[(i + 1) % n, i] = True
    return a


def _petersen_graph() -> np.ndarray:
    # Standard Petersen graph adjacency.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),       # outer 5-cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),       # inner pentagram
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),       # spokes
    ]
    a = np.zeros((10, 10), dtype=bool)
    for i, j in edges:
        a[i, j] = True
        a[j, i] = True
    return a


# ---------------------------------------------------------------------------
# Sanity tests on small known graphs
# ---------------------------------------------------------------------------

def test_complete_graph_clustering():
    a = _complete_graph(5)
    triangles, wedges = count_triangles_and_wedges(a)
    assert triangles == 10  # C(5, 3)
    assert wedges == 5 * (4 * 3 // 2)
    assert global_clustering_coefficient(a) == pytest.approx(1.0)
    assert mean_local_clustering_coefficient(a) == pytest.approx(1.0)


def test_cycle_graph_no_triangles():
    a = _cycle_graph(6)
    triangles, _ = count_triangles_and_wedges(a)
    assert triangles == 0
    assert global_clustering_coefficient(a) == 0.0


def test_petersen_graph_no_triangles():
    a = _petersen_graph()
    triangles, _ = count_triangles_and_wedges(a)
    assert triangles == 0
    assert global_clustering_coefficient(a) == 0.0


def test_clustering_zero_on_empty_graph():
    a = np.zeros((10, 10), dtype=bool)
    assert global_clustering_coefficient(a) == 0.0


def test_clustering_zero_on_path_graph():
    a = np.zeros((10, 10), dtype=bool)
    for i in range(9):
        a[i, i + 1] = True
        a[i + 1, i] = True
    assert global_clustering_coefficient(a) == 0.0


# ---------------------------------------------------------------------------
# Triangle counting cross-check
# ---------------------------------------------------------------------------

def test_triangle_count_methods_agree_complete():
    a = _complete_graph(7)
    trace_count, _ = count_triangles_and_wedges(a)
    nx_count = count_triangles_networkx(a)
    assert trace_count == nx_count == 35  # C(7, 3)


def test_triangle_count_methods_agree_e8():
    a = adjacency_inner_product_one(construct_e8_roots())
    trace_count, _ = count_triangles_and_wedges(a)
    nx_count = count_triangles_networkx(a)
    assert trace_count == nx_count


# ---------------------------------------------------------------------------
# E8 root graph properties
# ---------------------------------------------------------------------------

def test_e8_convention_a_is_56_regular():
    a = adjacency_inner_product_one(construct_e8_roots())
    degrees = a.sum(axis=1)
    assert np.all(degrees == 56)


def test_e8_convention_b_is_113_regular():
    a = adjacency_inner_product_nonzero(construct_e8_roots())
    degrees = a.sum(axis=1)
    assert np.all(degrees == 113)


def test_e8_convention_c_is_112_regular():
    a = adjacency_absolute_inner_product_one(construct_e8_roots())
    degrees = a.sum(axis=1)
    assert np.all(degrees == 112)


def test_e8_convention_d_is_56_regular():
    a = adjacency_inner_product_minus_one(construct_e8_roots())
    degrees = a.sum(axis=1)
    assert np.all(degrees == 56)


def test_e8_global_equals_local_mean_convention_a():
    a = adjacency_inner_product_one(construct_e8_roots())
    cg = global_clustering_coefficient(a)
    cl = mean_local_clustering_coefficient(a)
    np.testing.assert_allclose(cg, cl, rtol=1e-12)


def test_e8_convention_a_and_d_disagree():
    """The involution α → -α takes +1 triples to +1 triples, not -1 triples,
    so the +1 graph and the -1 graph are NOT isomorphic. They share edge
    count and degree but the triangle counts differ.
    """
    e8 = construct_e8_roots()
    a = adjacency_inner_product_one(e8)
    d = adjacency_inner_product_minus_one(e8)
    edges_a = int(a.sum()) // 2
    edges_d = int(d.sum()) // 2
    assert edges_a == edges_d == 6720

    cg_a = global_clustering_coefficient(a)
    cg_d = global_clustering_coefficient(d)
    assert cg_a > cg_d
    assert not np.isclose(cg_a, cg_d, atol=1e-3)


# ---------------------------------------------------------------------------
# E8 ⊕ E8 graph properties
# ---------------------------------------------------------------------------

def test_e8xe8_convention_a_disconnected_two_components():
    """With Convention A and orthogonal blocks, the graph should split."""
    from e8_heterotic.core.clustering import num_connected_components

    a = adjacency_inner_product_one(construct_e8xe8_roots())
    assert num_connected_components(a) == 2


def test_e8xe8_clustering_matches_single_e8_for_convention_a():
    """Direct sum of identical regular graphs has the same C_global."""
    e8 = construct_e8_roots()
    e8xe8 = construct_e8xe8_roots()

    cg_e8 = global_clustering_coefficient(adjacency_inner_product_one(e8))
    cg_e8xe8 = global_clustering_coefficient(
        adjacency_inner_product_one(e8xe8)
    )
    np.testing.assert_allclose(cg_e8, cg_e8xe8, rtol=1e-12)


# ---------------------------------------------------------------------------
# Hardcoded-return guard
# ---------------------------------------------------------------------------

def test_no_hardcoded_returns_zero_graph():
    a = np.zeros((10, 10), dtype=bool)
    assert global_clustering_coefficient(a) == 0.0
    assert mean_local_clustering_coefficient(a) == 0.0


def test_no_hardcoded_returns_path_graph():
    a = np.zeros((10, 10), dtype=bool)
    for i in range(9):
        a[i, i + 1] = True
        a[i + 1, i] = True
    assert global_clustering_coefficient(a) == 0.0


def test_clustering_does_not_return_literature_value():
    """A controlled graph must not produce 25/32 by accident."""
    a = _complete_graph(5)
    assert global_clustering_coefficient(a) != pytest.approx(25.0 / 32.0)
