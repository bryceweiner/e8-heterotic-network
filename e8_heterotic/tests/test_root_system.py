"""Tests for the E8 and E8 ⊕ E8 root-system construction."""

import numpy as np
import pytest

from e8_heterotic.core.root_system import (
    construct_e8_roots,
    construct_e8xe8_roots,
)


def test_e8_root_count_and_shape():
    roots = construct_e8_roots()
    assert roots.shape == (240, 8)
    assert roots.dtype == np.float64


def test_e8_squared_norms():
    roots = construct_e8_roots()
    norms_sq = np.einsum("ij,ij->i", roots, roots)
    np.testing.assert_allclose(norms_sq, 2.0, atol=1e-10)


def test_e8_inner_product_distribution():
    roots = construct_e8_roots()
    g = roots @ roots.T
    g_off = g.copy().astype(np.float64)
    np.fill_diagonal(g_off, np.nan)
    distinct = np.unique(np.round(g_off[~np.isnan(g_off)], 6))
    expected = np.array([-2.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(distinct, expected, atol=1e-9)


def test_e8_inner_product_multiplicities():
    roots = construct_e8_roots()
    g = roots @ roots.T
    iu = np.triu_indices(240, k=1)
    rounded = np.round(g[iu], 6)
    unique, counts = np.unique(rounded, return_counts=True)
    multiplicities = dict(zip(unique.tolist(), counts.tolist()))

    # Each root has exactly one antipode, giving 240 pairs at ⟨,⟩ = -2; over
    # unordered pairs that's 240/2 = 120.
    assert multiplicities[-2.0] == 120
    # Each root has 56 neighbours at +1 and 56 at -1; in unordered pairs,
    # the +1 and -1 counts are 240 * 56 / 2 = 6720 each.
    assert multiplicities[1.0] == 6720
    assert multiplicities[-1.0] == 6720
    # Remaining pairs are orthogonal: C(240, 2) - 120 - 6720 - 6720 = 15,120.
    assert multiplicities[0.0] == 15120


def test_e8_no_duplicate_roots():
    roots = construct_e8_roots()
    seen = {tuple(np.round(r, 9)) for r in roots}
    assert len(seen) == 240


def test_e8_closed_under_negation():
    roots = construct_e8_roots()
    seen = {tuple(np.round(r, 9)) for r in roots}
    for r in roots:
        assert tuple(np.round(-r, 9)) in seen


def test_e8xe8_construction():
    roots = construct_e8xe8_roots()
    assert roots.shape == (480, 16)

    # First 240 zero in coords 8..15, last 240 zero in coords 0..7.
    assert np.allclose(roots[:240, 8:], 0)
    assert np.allclose(roots[240:, :8], 0)

    # Cross-block inner products all zero.
    cross = roots[:240] @ roots[240:].T
    assert np.allclose(cross, 0)

    # All squared norms exactly 2.
    norms_sq = np.einsum("ij,ij->i", roots, roots)
    np.testing.assert_allclose(norms_sq, 2.0, atol=1e-10)
