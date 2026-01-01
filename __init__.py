"""
E8×E8 Heterotic Network: Geometric Deep Learning Layer

This package implements the 496-dimensional E8×E8 heterotic structure from string theory
as a differentiable PyTorch layer with exact clustering coefficient C(G) = 25/32 (0.78125).

The E8×E8 heterotic construction combines two independent E8 exceptional Lie algebras
to create an optimal information processing architecture for geometric deep learning.

Key Features:
- Exact mathematical implementation of E8×E8 heterotic structure
- Guaranteed clustering coefficient of 25/32 (0.78125)
- 496-dimensional geometric latent space
- Holographic information bounds enforcement
- Hardware acceleration (CUDA/MPS/CPU) with sparse matrix optimization
- Comprehensive caching system for expensive computations

Basic Usage:
    import torch
    from e8_heterotic import E8E8Layer

    # Create E8×E8 layer
    layer = E8E8Layer(input_dim=128, output_dim=64)

    # Forward pass
    x = torch.randn(32, 128)  # Batch of 32, 128 features
    output = layer(x)  # Shape: (32, 64)

For more information, see the documentation and examples.
"""

from e8_heterotic.core.constants import (
    E8_CLUSTERING,
    E8_ROOTS,
    E8_CARTAN,
    E8_DIMENSION,
    E8XE8_TOTAL_GENERATORS,
    E8XE8_EMBEDDING_DIM,
    get_e8_clustering_coefficient,
    get_e8_dimensions,
    get_information_bounds,
    validate_constants
)

from e8_heterotic.core.construction import (
    E8HeteroticSystem,
    verify_e8_construction
)

from e8_heterotic.core.cache import (
    E8Cache,
    get_e8_cache,
    get_e8_clustering_coefficient as get_cached_clustering,
    get_e8_root_system,
    get_e8_adjacency_matrix,
    get_e8_3d_coordinates,
    get_e8_fold_coordinates
)

from e8_heterotic.core.network import (
    E8E8Layer,
    create_e8_layer,
    get_e8_network_properties
)

from e8_heterotic.utils.mathematics import (
    normalize_vector,
    calculate_angle,
    check_adjacency_condition,
    calculate_clustering_coefficient,
    count_triangles_and_triplets,
    validate_root_norms,
    validate_dot_products,
    calculate_geometric_properties
)

from e8_heterotic.utils.device import (
    get_available_devices,
    get_optimal_device,
    get_sparse_safe_device,
    is_sparse_supported,
    get_device_info,
    print_device_info,
    setup_e8_computation_environment
)

__version__ = "0.1.0"
__author__ = "E8 Heterotic Network Team"
__description__ = "E8×E8 Heterotic Structure for Geometric Deep Learning"
__license__ = "MIT"

__all__ = [
    # Core classes
    'E8HeteroticSystem',
    'E8Cache',
    'E8E8Layer',

    # Constants and utilities
    'E8_CLUSTERING',
    'E8_ROOTS',
    'E8_CARTAN',
    'E8_DIMENSION',
    'E8XE8_TOTAL_GENERATORS',
    'E8XE8_EMBEDDING_DIM',

    # Functions
    'get_e8_clustering_coefficient',
    'get_e8_dimensions',
    'get_information_bounds',
    'validate_constants',
    'verify_e8_construction',
    'get_e8_cache',
    'get_cached_clustering',
    'get_e8_root_system',
    'get_e8_adjacency_matrix',
    'get_e8_3d_coordinates',
    'get_e8_fold_coordinates',
    'create_e8_layer',
    'get_e8_network_properties',

    # Mathematics utilities
    'normalize_vector',
    'calculate_angle',
    'check_adjacency_condition',
    'calculate_clustering_coefficient',
    'count_triangles_and_triplets',
    'validate_root_norms',
    'validate_dot_products',
    'calculate_geometric_properties',

    # Device utilities
    'get_available_devices',
    'get_optimal_device',
    'get_sparse_safe_device',
    'is_sparse_supported',
    'get_device_info',
    'print_device_info',
    'setup_e8_computation_environment'
]

def __validate_installation():
    """Validate that the package is properly installed and all dependencies are available."""
    try:
        import torch
        import numpy
        import scipy
        import networkx

        # Try sklearn import (optional)
        try:
            import sklearn
            sklearn_available = True
        except ImportError:
            sklearn_available = False

        # Validate constants
        validation = validate_constants()

        if validation.get('overall', {}).get('valid', False):
            print("✓ E8×E8 Heterotic Network package validated successfully")
            if not sklearn_available:
                print("⚠ sklearn not available - geometric projections will be limited")
        else:
            print("⚠ Some constants failed validation - package may not work correctly")

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required dependencies: pip install torch numpy scipy networkx")

# Run validation on import (only in interactive mode)
if __name__ != "__main__":
    __validate_installation()