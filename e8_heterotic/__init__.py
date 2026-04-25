"""
E8×E8 Heterotic Network: honest E8 ⊕ E8 root system, adjacency, and clustering.

The clustering coefficient of the E8 ⊕ E8 root graph is a derived quantity
computed from the adjacency matrix; this package never returns a hardcoded
clustering value. See :mod:`e8_heterotic.core.clustering` and
:mod:`e8_heterotic.core.adjacency`.
"""

from e8_heterotic.core.adjacency import (
    CONVENTIONS,
    adjacency_absolute_inner_product_one,
    adjacency_inner_product_minus_one,
    adjacency_inner_product_nonzero,
    adjacency_inner_product_one,
)
from e8_heterotic.core.cache import (
    E8Cache,
    get_e8_adjacency_matrix,
    get_e8_cache,
    get_e8_clustering_coefficient,
    get_e8_root_system,
)
from e8_heterotic.core.clustering import (
    count_triangles_and_wedges,
    degree_distribution,
    global_clustering_coefficient,
    mean_local_clustering_coefficient,
)
from e8_heterotic.core.constants import (
    E8_CARTAN,
    E8_CLUSTERING_LITERATURE_CLAIM,
    E8_CLUSTERING_LITERATURE_FRACTION,
    E8_ROOT_NORM,
    E8_ROOT_NORM_SQUARED,
    E8_ROOTS,
    E8XE8_EMBEDDING_DIM,
    E8XE8_ROOTS,
)
from e8_heterotic.core.construction import E8xE8RootSystem
from e8_heterotic.core.root_system import (
    construct_cartan_subalgebra,
    construct_e8_roots,
    construct_e8xe8_roots,
)

__version__ = "0.2.0"
__author__ = "E8 Heterotic Network Team"
__description__ = "Honest E8 ⊕ E8 root system and clustering analysis"
__license__ = "MIT"

__all__ = [
    # Core orchestration
    "E8xE8RootSystem",
    # Construction
    "construct_e8_roots",
    "construct_e8xe8_roots",
    "construct_cartan_subalgebra",
    # Adjacency
    "CONVENTIONS",
    "adjacency_inner_product_one",
    "adjacency_inner_product_nonzero",
    "adjacency_absolute_inner_product_one",
    "adjacency_inner_product_minus_one",
    # Clustering
    "count_triangles_and_wedges",
    "degree_distribution",
    "global_clustering_coefficient",
    "mean_local_clustering_coefficient",
    # Caching
    "E8Cache",
    "get_e8_cache",
    "get_e8_clustering_coefficient",
    "get_e8_root_system",
    "get_e8_adjacency_matrix",
    # Constants
    "E8_ROOTS",
    "E8_CARTAN",
    "E8XE8_ROOTS",
    "E8XE8_EMBEDDING_DIM",
    "E8_ROOT_NORM",
    "E8_ROOT_NORM_SQUARED",
    "E8_CLUSTERING_LITERATURE_CLAIM",
    "E8_CLUSTERING_LITERATURE_FRACTION",
]


def _try_import_layer():
    """Make :class:`E8E8Layer` importable when PyTorch is present."""
    try:
        from e8_heterotic.core.network import E8E8Layer, create_e8_layer
    except ImportError:
        return None
    globals()["E8E8Layer"] = E8E8Layer
    globals()["create_e8_layer"] = create_e8_layer
    __all__.extend(["E8E8Layer", "create_e8_layer"])
    return E8E8Layer


_try_import_layer()
