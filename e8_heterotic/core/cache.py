"""
On-disk cache for the E8 ⊕ E8 root system, adjacency, and clustering.

The cache stores results as-computed. There is no override of clustering
values against any "theoretical" target.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Optional

import numpy as np

from e8_heterotic.core.adjacency import CONVENTIONS
from e8_heterotic.core.clustering import (
    count_triangles_and_wedges,
    degree_distribution,
    global_clustering_coefficient,
    mean_local_clustering_coefficient,
    num_connected_components,
)
from e8_heterotic.core.constants import (
    E8XE8_EMBEDDING_DIM,
    E8XE8_ROOTS,
    INNER_PRODUCT_TOLERANCE,
)
from e8_heterotic.core.root_system import construct_e8xe8_roots

logger = logging.getLogger(__name__)


class E8Cache:
    """File-backed cache of the E8 ⊕ E8 root system and its adjacency graph."""

    def __init__(
        self,
        cache_dir: str = "e8_cache",
        adjacency_convention: str = "A",
    ) -> None:
        if adjacency_convention not in CONVENTIONS:
            raise ValueError(
                f"unknown adjacency convention {adjacency_convention!r}; "
                f"expected one of {sorted(CONVENTIONS)}"
            )
        self.cache_dir = cache_dir
        self.adjacency_convention = adjacency_convention
        os.makedirs(cache_dir, exist_ok=True)

        self._root_system: Optional[np.ndarray] = None
        self._adjacency_matrix: Optional[np.ndarray] = None
        self._network_properties: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{name}.pkl")

    def _exists(self, name: str) -> bool:
        return os.path.exists(self._path(name))

    def _save(self, name: str, data: Any) -> None:
        try:
            with open(self._path(name), "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Cached %s → %s", name, self._path(name))
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Failed to cache %s: %s", name, exc)

    def _load(self, name: str) -> Optional[Any]:
        try:
            with open(self._path(name), "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Failed to load cache %s: %s", name, exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_root_system(self, force_regenerate: bool = False) -> np.ndarray:
        name = "e8xe8_root_system"
        if not force_regenerate and self._exists(name):
            data = self._load(name)
            if data is not None and data.shape == (E8XE8_ROOTS, E8XE8_EMBEDDING_DIM):
                self._root_system = data
                return data

        logger.info("Constructing E8 ⊕ E8 root system (%d roots in R^%d)",
                    E8XE8_ROOTS, E8XE8_EMBEDDING_DIM)
        t0 = time.perf_counter()
        roots = construct_e8xe8_roots()
        logger.debug("Root system built in %.3fs", time.perf_counter() - t0)

        self._save(name, roots)
        self._root_system = roots
        return roots

    def get_adjacency_matrix(
        self, force_regenerate: bool = False
    ) -> np.ndarray:
        name = f"e8xe8_adjacency_{self.adjacency_convention}"
        if not force_regenerate and self._exists(name):
            data = self._load(name)
            if data is not None and data.shape == (E8XE8_ROOTS, E8XE8_ROOTS):
                self._adjacency_matrix = data
                return data

        roots = self.get_root_system()
        label, fn = CONVENTIONS[self.adjacency_convention]
        logger.info(
            "Computing adjacency matrix using Convention %s (%s)",
            self.adjacency_convention, label,
        )
        t0 = time.perf_counter()
        adjacency = fn(roots, tolerance=INNER_PRODUCT_TOLERANCE)
        logger.debug("Adjacency built in %.3fs", time.perf_counter() - t0)

        self._save(name, adjacency)
        self._adjacency_matrix = adjacency
        return adjacency

    def compute_network_properties(
        self, force_regenerate: bool = False
    ) -> dict[str, Any]:
        name = f"e8xe8_network_properties_{self.adjacency_convention}"
        if not force_regenerate and self._exists(name):
            data = self._load(name)
            if data is not None:
                self._network_properties = data
                return data

        adjacency = self.get_adjacency_matrix()
        logger.info(
            "Computing network properties for Convention %s",
            self.adjacency_convention,
        )

        triangles, wedges = count_triangles_and_wedges(adjacency)
        c_global = global_clustering_coefficient(adjacency)
        c_local_mean = mean_local_clustering_coefficient(adjacency)
        n_components = num_connected_components(adjacency)
        degs = degree_distribution(adjacency)
        n_edges = int(adjacency.sum()) // 2

        properties: dict[str, Any] = {
            "adjacency_convention": self.adjacency_convention,
            "num_nodes": int(adjacency.shape[0]),
            "num_edges": n_edges,
            "num_triangles": triangles,
            "num_wedges": wedges,
            "clustering_coefficient": c_global,
            "mean_local_clustering": c_local_mean,
            "min_degree": int(degs.min()),
            "max_degree": int(degs.max()),
            "average_degree": float(degs.mean()),
            "is_connected": n_components == 1,
            "components": n_components,
        }

        logger.info(
            "Properties: nodes=%d edges=%d triangles=%d wedges=%d "
            "C_global=%.10f components=%d",
            properties["num_nodes"], n_edges, triangles, wedges,
            c_global, n_components,
        )

        self._save(name, properties)
        self._network_properties = properties
        return properties

    def get_clustering_coefficient(self) -> float:
        if self._network_properties is None:
            self.compute_network_properties()
        return float(self._network_properties["clustering_coefficient"])

    def clear_cache(self) -> None:
        import shutil

        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        self._root_system = None
        self._adjacency_matrix = None
        self._network_properties = None


# Module-level singletons keyed by convention so different layers can share.
_caches: dict[str, E8Cache] = {}


def get_e8_cache(adjacency_convention: str = "A") -> E8Cache:
    """Return a process-wide singleton cache for the given convention."""
    cache = _caches.get(adjacency_convention)
    if cache is None:
        cache = E8Cache(adjacency_convention=adjacency_convention)
        _caches[adjacency_convention] = cache
    return cache


def get_e8_clustering_coefficient(adjacency_convention: str = "A") -> float:
    return get_e8_cache(adjacency_convention).get_clustering_coefficient()


def get_e8_root_system() -> np.ndarray:
    return get_e8_cache().get_root_system()


def get_e8_adjacency_matrix(adjacency_convention: str = "A") -> np.ndarray:
    return get_e8_cache(adjacency_convention).get_adjacency_matrix()
