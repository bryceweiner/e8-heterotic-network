"""
E8×E8 Heterotic Core Components

This module contains the core mathematical and computational components
of the E8×E8 heterotic network implementation.
"""

from .constants import *
from .construction import *
from .cache import *
from .network import *

__all__ = [
    # From constants
    'E8_CLUSTERING', 'E8_ROOTS', 'E8_CARTAN', 'E8_DIMENSION',
    'E8XE8_TOTAL_GENERATORS', 'E8XE8_EMBEDDING_DIM',
    'get_e8_clustering_coefficient', 'get_e8_dimensions',
    'get_information_bounds', 'validate_constants',

    # From construction
    'E8HeteroticSystem', 'verify_e8_construction',

    # From cache
    'E8Cache', 'get_e8_cache', 'get_e8_clustering_coefficient',
    'get_e8_root_system', 'get_e8_adjacency_matrix',
    'get_e8_3d_coordinates', 'get_e8_fold_coordinates',

    # From network
    'E8E8Layer', 'create_e8_layer', 'get_e8_network_properties'
]