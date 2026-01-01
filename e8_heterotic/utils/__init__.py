"""
E8×E8 Heterotic Utilities

This module contains utility functions for mathematical operations,
device management, and helper functions used throughout the E8×E8 framework.
"""

from .mathematics import *
from .device import *

__all__ = [
    # From mathematics
    'normalize_vector', 'calculate_angle', 'check_adjacency_condition',
    'calculate_clustering_coefficient', 'count_triangles_and_triplets',
    'validate_root_norms', 'validate_dot_products', 'calculate_geometric_properties',

    # From device
    'get_available_devices', 'get_optimal_device', 'get_sparse_safe_device',
    'is_sparse_supported', 'get_device_info', 'print_device_info',
    'setup_e8_computation_environment'
]