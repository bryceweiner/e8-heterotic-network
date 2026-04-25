"""
Honest orchestrator for the E8 and E8 ⊕ E8 root systems.

No tunable parameters. No cross-coupling rules. No hardcoded clustering
values. Construction → adjacency → clustering, end to end.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from e8_heterotic.core.adjacency import CONVENTIONS
from e8_heterotic.core.clustering import (
    count_triangles_and_wedges,
    count_triangles_networkx,
    degree_distribution,
    global_clustering_coefficient,
    mean_local_clustering_coefficient,
    num_connected_components,
)
from e8_heterotic.core.constants import (
    E8_ROOTS,
    E8XE8_EMBEDDING_DIM,
    E8XE8_ROOTS,
    INNER_PRODUCT_TOLERANCE,
    NORM_TOLERANCE,
)
from e8_heterotic.core.root_system import (
    construct_e8_roots,
    construct_e8xe8_roots,
)

logger = logging.getLogger(__name__)


class E8xE8RootSystem:
    """Honest construction and adjacency-graph analysis of E8 and E8 ⊕ E8."""

    def __init__(self) -> None:
        logger.info("Constructing E8 root system")
        self.e8_roots = construct_e8_roots()
        logger.info("Constructing E8 ⊕ E8 root system")
        self.e8xe8_roots = construct_e8xe8_roots()
        self._validate_root_system()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_root_system(self) -> None:
        if self.e8_roots.shape != (E8_ROOTS, 8):
            raise ValueError(
                f"E8 root array has shape {self.e8_roots.shape}; "
                f"expected ({E8_ROOTS}, 8)"
            )
        if self.e8xe8_roots.shape != (E8XE8_ROOTS, E8XE8_EMBEDDING_DIM):
            raise ValueError(
                f"E8×E8 root array has shape {self.e8xe8_roots.shape}; "
                f"expected ({E8XE8_ROOTS}, {E8XE8_EMBEDDING_DIM})"
            )

        e8_norms_sq = np.einsum("ij,ij->i", self.e8_roots, self.e8_roots)
        if not np.all(np.abs(e8_norms_sq - 2.0) < NORM_TOLERANCE):
            bad = np.argwhere(np.abs(e8_norms_sq - 2.0) >= NORM_TOLERANCE)
            raise ValueError(
                f"E8 has roots with squared norm ≠ 2 at indices {bad.ravel()[:5]}…"
            )

        e8xe8_norms_sq = np.einsum(
            "ij,ij->i", self.e8xe8_roots, self.e8xe8_roots
        )
        if not np.all(np.abs(e8xe8_norms_sq - 2.0) < NORM_TOLERANCE):
            raise ValueError("E8 ⊕ E8 has roots with squared norm ≠ 2")

        gram = self.e8_roots @ self.e8_roots.T
        rounded = np.round(gram, 6)
        unique = np.unique(rounded)
        allowed = {-2.0, -1.0, 0.0, 1.0, 2.0}
        unexpected = sorted(v for v in unique.tolist() if v not in allowed)
        if unexpected:
            raise ValueError(
                "E8 inner-product matrix has unexpected values: "
                f"{unexpected[:10]}"
            )

        cross = self.e8xe8_roots[:E8_ROOTS] @ self.e8xe8_roots[E8_ROOTS:].T
        max_cross = float(np.max(np.abs(cross)))
        if max_cross > INNER_PRODUCT_TOLERANCE:
            raise ValueError(
                f"E8 ⊕ E8 cross-block inner products exceed tolerance: "
                f"max |⟨,⟩| = {max_cross:g}"
            )

        logger.info(
            "Root system validated: E8 (240 roots, norm²=2), "
            "E8 ⊕ E8 (480 roots, orthogonal blocks)"
        )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def inner_product_distribution(self, system: str = "e8") -> dict[float, int]:
        """Return a histogram of distinct unordered-pair inner products.

        Counts each pair ``{i, j}`` with ``i < j`` once.
        """
        roots = self._select(system)
        gram = roots @ roots.T
        n = roots.shape[0]
        iu = np.triu_indices(n, k=1)
        values = np.round(gram[iu], 6)
        unique, counts = np.unique(values, return_counts=True)
        return {float(v): int(c) for v, c in zip(unique, counts)}

    # ------------------------------------------------------------------
    # Per-convention analysis
    # ------------------------------------------------------------------

    def analyze(self, system: str = "e8xe8") -> dict[str, dict[str, Any]]:
        """Compute clustering for each convention on the chosen system.

        ``system`` is ``"e8"`` or ``"e8xe8"``. Returns a dict keyed by
        convention name (``"A"``, ``"B"``, ``"C"``, ``"D"``).
        """
        roots = self._select(system)
        results: dict[str, dict[str, Any]] = {}

        for name, (label, fn) in CONVENTIONS.items():
            logger.info("Convention %s (%s) — computing adjacency on '%s' "
                        "(%d vertices)",
                        name, label, system, roots.shape[0])
            adjacency = fn(roots, tolerance=INNER_PRODUCT_TOLERANCE)

            triangles, wedges = count_triangles_and_wedges(adjacency)
            triangles_nx = count_triangles_networkx(adjacency)
            if triangles != triangles_nx:
                raise RuntimeError(
                    f"Convention {name}: trace(A^3)/6 = {triangles} "
                    f"disagrees with NetworkX = {triangles_nx}"
                )

            c_global = global_clustering_coefficient(adjacency)
            c_local_mean = mean_local_clustering_coefficient(adjacency)
            degs = degree_distribution(adjacency)
            n_components = num_connected_components(adjacency)
            n_edges = int(adjacency.sum()) // 2
            min_deg = int(degs.min())
            max_deg = int(degs.max())

            notes: list[str] = []
            if min_deg == max_deg:
                degree_repr: int | dict[str, int] = min_deg
            else:
                degree_repr = {
                    "min": min_deg,
                    "max": max_deg,
                    "mean": float(degs.mean()),
                }
                notes.append("graph is not regular")

            if n_components > 1:
                notes.append(f"graph has {n_components} components")

            logger.info(
                "Convention %s: degree=%s edges=%d triangles=%d wedges=%d "
                "C_global=%.10f C_local_mean=%.10f components=%d",
                name, degree_repr, n_edges, triangles, wedges,
                c_global, c_local_mean, n_components,
            )

            results[name] = {
                "label": label,
                "num_vertices": int(adjacency.shape[0]),
                "num_edges": n_edges,
                "degree": degree_repr,
                "num_triangles": triangles,
                "num_wedges": wedges,
                "global_clustering": c_global,
                "mean_local_clustering": c_local_mean,
                "num_components": n_components,
                "notes": "; ".join(notes) if notes else "",
            }

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select(self, system: str) -> np.ndarray:
        if system == "e8":
            return self.e8_roots
        if system == "e8xe8":
            return self.e8xe8_roots
        raise ValueError(f"unknown system {system!r}; expected 'e8' or 'e8xe8'")
