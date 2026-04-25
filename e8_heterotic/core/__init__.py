"""Core components of the E8 ⊕ E8 root system analysis."""

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
    E8_ROOT_NORM,
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

__all__ = [
    "CONVENTIONS",
    "E8Cache",
    "E8xE8RootSystem",
    "E8_CARTAN",
    "E8_CLUSTERING_LITERATURE_CLAIM",
    "E8_ROOTS",
    "E8_ROOT_NORM",
    "E8XE8_EMBEDDING_DIM",
    "E8XE8_ROOTS",
    "adjacency_absolute_inner_product_one",
    "adjacency_inner_product_minus_one",
    "adjacency_inner_product_nonzero",
    "adjacency_inner_product_one",
    "construct_cartan_subalgebra",
    "construct_e8_roots",
    "construct_e8xe8_roots",
    "count_triangles_and_wedges",
    "degree_distribution",
    "get_e8_adjacency_matrix",
    "get_e8_cache",
    "get_e8_clustering_coefficient",
    "get_e8_root_system",
    "global_clustering_coefficient",
    "mean_local_clustering_coefficient",
]
