"""
Mathematical utilities for E8×E8 heterotic structure calculations.

This module provides helper functions for geometric calculations, root system
operations, and numerical validation used throughout the E8×E8 heterotic framework.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from itertools import combinations

from e8_heterotic.core.constants import (
    E8_ROOT_NORM, DOT_PRODUCT_TOLERANCE, ANGLE_TOLERANCE,
    ADJACENCY_DOT_PRODUCTS, NORM_TOLERANCE
)

def normalize_vector(v: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector
        dtype: Optional dtype for the result

    Returns:
        numpy.ndarray: Normalized vector
    """
    if dtype is None:
        dtype = v.dtype

    norm = np.linalg.norm(v)
    if norm < NORM_TOLERANCE:
        return np.zeros_like(v, dtype=dtype)

    return (v / norm).astype(dtype)

def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in degrees.

    Args:
        v1, v2: Input vectors

    Returns:
        float: Angle in degrees [0, 180]
    """
    # Normalize vectors
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    # Calculate dot product
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    # Convert to angle
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def check_adjacency_condition(v1: np.ndarray, v2: np.ndarray,
                            tolerance: float = DOT_PRODUCT_TOLERANCE) -> bool:
    """
    Check if two vectors satisfy the E8×E8 adjacency condition.

    Two roots are adjacent if their angle is 60°, 90°, or 120°,
    corresponding to dot products of 0.5, 0.0, or -0.5 for normalized vectors.

    Args:
        v1, v2: Input vectors
        tolerance: Numerical tolerance for comparison

    Returns:
        bool: True if vectors are adjacent
    """
    # Normalize vectors
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    # Calculate dot product
    dot_product = np.dot(v1_norm, v2_norm)

    # Check against adjacency dot products
    for target_dot in ADJACENCY_DOT_PRODUCTS:
        if abs(dot_product - target_dot) < tolerance:
            return True

    return False

def generate_permutations(base_pattern: List[float]) -> List[np.ndarray]:
    """
    Generate all unique permutations of a base pattern for E8 root construction.

    Used to generate the spinor weights from the base pattern [1,1,1,1,1,1,-2,-2,-2].

    Args:
        base_pattern: Base pattern to permute

    Returns:
        List[np.ndarray]: List of all unique permutations
    """
    n = len(base_pattern)
    unique_perms = set()

    # Generate all permutations (this is expensive but necessary for E8)
    from itertools import permutations

    for perm in permutations(base_pattern):
        # Convert to tuple for set uniqueness
        perm_tuple = tuple(perm)
        unique_perms.add(perm_tuple)

    # Convert back to numpy arrays
    return [np.array(list(perm)) for perm in unique_perms]

def project_to_trace_zero(v: np.ndarray, target_dim: int = 8) -> np.ndarray:
    """
    Project a 9D vector to 8D trace-zero subspace.

    Used in E8 construction to project from 9D to 8D space.

    Args:
        v: 9D input vector
        target_dim: Target dimension (should be 8)

    Returns:
        np.ndarray: Projected 8D vector
    """
    if len(v) != 9:
        raise ValueError(f"Input vector must be 9D, got {len(v)}D")

    if target_dim != 8:
        raise ValueError(f"Target dimension must be 8, got {target_dim}")

    # The trace-zero condition removes one degree of freedom
    # We simply take the first 8 coordinates
    return v[:8].copy()

def calculate_clustering_coefficient(adjacency_matrix: np.ndarray) -> float:
    """
    Calculate the clustering coefficient of a graph.

    The clustering coefficient C(G) is the average of the local clustering
    coefficients of all nodes, where the local clustering coefficient of
    node i is: C_i = (number of triangles connected to i) / (degree(i) choose 2)

    Args:
        adjacency_matrix: Binary adjacency matrix

    Returns:
        float: Clustering coefficient C(G)
    """
    n = adjacency_matrix.shape[0]

    if n == 0:
        return 0.0

    total_clustering = 0.0
    valid_nodes = 0

    for i in range(n):
        # Get neighbors of node i
        neighbors = np.where(adjacency_matrix[i] == 1)[0]

        # Remove self if present (shouldn't be)
        neighbors = neighbors[neighbors != i]

        degree = len(neighbors)

        if degree < 2:
            # Node with degree < 2 cannot form triangles
            continue

        # Count triangles: number of edges between neighbors
        triangles = 0
        for j_idx in range(degree):
            for k_idx in range(j_idx + 1, degree):
                j, k = neighbors[j_idx], neighbors[k_idx]
                if adjacency_matrix[j, k] == 1:
                    triangles += 1

        # Local clustering coefficient
        possible_triangles = degree * (degree - 1) // 2
        if possible_triangles > 0:
            local_clustering = triangles / possible_triangles
            total_clustering += local_clustering
            valid_nodes += 1

    # Global clustering coefficient
    if valid_nodes == 0:
        return 0.0

    return total_clustering / valid_nodes

def count_triangles_and_triplets(adjacency_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Count the total number of triangles and triplets in the graph.

    A triplet is three nodes where at least two are connected.
    A triangle is three nodes where all three pairs are connected.

    Args:
        adjacency_matrix: Binary adjacency matrix

    Returns:
        Tuple[int, int]: (triangles, triplets)
    """
    n = adjacency_matrix.shape[0]
    triangles = 0
    triplets = 0

    # Use the efficient method: for each node, count triangles among its neighbors
    for i in range(n):
        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        neighbors = neighbors[neighbors != i]  # Remove self

        degree = len(neighbors)

        if degree < 2:
            continue

        # Count triplets: all possible pairs of neighbors
        triplets += degree * (degree - 1) // 2

        # Count triangles: pairs of neighbors that are connected
        for j_idx in range(degree):
            for k_idx in range(j_idx + 1, degree):
                j, k = neighbors[j_idx], neighbors[k_idx]
                if adjacency_matrix[j, k] == 1:
                    triangles += 1

    return triangles, triplets

def validate_root_norms(roots: np.ndarray,
                       expected_norm: float = E8_ROOT_NORM,
                       tolerance: float = NORM_TOLERANCE) -> dict:
    """
    Validate that all roots have the expected norm.

    Args:
        roots: Array of root vectors
        expected_norm: Expected norm value
        tolerance: Tolerance for norm comparison

    Returns:
        dict: Validation results
    """
    if len(roots) == 0:
        return {'valid': False, 'message': 'No roots provided'}

    norms = np.array([np.linalg.norm(root) for root in roots])

    min_norm = np.min(norms)
    max_norm = np.max(norms)
    mean_norm = np.mean(norms)

    # Check if all norms are within tolerance of expected norm
    norm_diffs = np.abs(norms - expected_norm)
    max_diff = np.max(norm_diffs)
    valid = max_diff < tolerance

    return {
        'valid': valid,
        'expected_norm': expected_norm,
        'min_norm': min_norm,
        'max_norm': max_norm,
        'mean_norm': mean_norm,
        'max_deviation': max_diff,
        'tolerance': tolerance
    }

def validate_dot_products(roots: np.ndarray,
                         tolerance: float = DOT_PRODUCT_TOLERANCE) -> dict:
    """
    Validate that dot products between roots are consistent with E8 geometry.

    For simply laced root systems like E8, dot products should be in
    {-2, -1, 0, 1, 2} for unnormalized roots.

    Args:
        roots: Array of root vectors
        tolerance: Tolerance for dot product validation

    Returns:
        dict: Validation results
    """
    if len(roots) == 0:
        return {'valid': False, 'message': 'No roots provided'}

    n = len(roots)
    dot_products = []

    # Calculate all pairwise dot products
    for i in range(n):
        for j in range(i + 1, n):
            dot_ij = np.dot(roots[i], roots[j])
            dot_products.append(dot_ij)

    dot_products = np.array(dot_products)
    unique_dots = np.unique(np.round(dot_products, decimals=6))

    # For E8, expected dot products are in {-2, -1, 0, 1, 2}
    expected_dots = [-2, -1, 0, 1, 2]

    # Check if unique dot products are close to expected values
    valid_dots = []
    for expected in expected_dots:
        close_dots = [d for d in unique_dots if abs(d - expected) < 0.1]
        if close_dots:
            valid_dots.append(expected)

    # All expected dot products should be present
    valid = len(valid_dots) == len(expected_dots)

    return {
        'valid': valid,
        'unique_dot_products': unique_dots.tolist(),
        'expected_dot_products': expected_dots,
        'found_expected': valid_dots,
        'missing_expected': [d for d in expected_dots if d not in valid_dots]
    }

def calculate_geometric_properties(roots: np.ndarray) -> dict:
    """
    Calculate comprehensive geometric properties of a root system.

    Args:
        roots: Array of root vectors

    Returns:
        dict: Geometric properties
    """
    if len(roots) == 0:
        return {'error': 'No roots provided'}

    # Basic properties
    n_roots = len(roots)
    dim = roots.shape[1] if len(roots.shape) > 1 else 1

    # Norms
    norms = np.array([np.linalg.norm(root) for root in roots])
    unique_norms = np.unique(np.round(norms, 6))

    # Angles and dot products
    angles = []
    dot_products = []

    for i in range(min(n_roots, 1000)):  # Sample for large systems
        for j in range(i + 1, min(n_roots, 1000)):
            angle_ij = calculate_angle(roots[i], roots[j])
            dot_ij = np.dot(normalize_vector(roots[i]), normalize_vector(roots[j]))

            angles.append(angle_ij)
            dot_products.append(dot_ij)

    angles = np.array(angles)
    dot_products = np.array(dot_products)

    # Angle distribution
    unique_angles = np.unique(np.round(angles, decimals=1))

    return {
        'num_roots': n_roots,
        'dimension': dim,
        'norms': {
            'min': float(np.min(norms)),
            'max': float(np.max(norms)),
            'mean': float(np.mean(norms)),
            'unique': unique_norms.tolist()
        },
        'angles': {
            'unique': unique_angles.tolist(),
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'mean': float(np.mean(angles))
        },
        'dot_products': {
            'unique': np.unique(np.round(dot_products, 6)).tolist(),
            'min': float(np.min(dot_products)),
            'max': float(np.max(dot_products))
        }
    }

# =============================================================================
# E8 SPECIFIC MATHEMATICAL FUNCTIONS
# =============================================================================

def construct_a8_roots() -> np.ndarray:
    """
    Construct the 72 roots of the A8 subsystem of E8.

    R(A8) = {e_i - e_j | 1 ≤ i,j ≤ 9, i ≠ j} projected to 8D trace-zero space.

    Returns:
        np.ndarray: Array of 72 A8 roots in 8D space
    """
    roots = []

    for i in range(9):
        for j in range(9):
            if i != j:
                # Create e_i - e_j in 9D space
                root_9d = np.zeros(9)
                root_9d[i] = 1.0
                root_9d[j] = -1.0

                # Project to 8D trace-zero space
                root_8d = project_to_trace_zero(root_9d)
                roots.append(root_8d)

    return np.array(roots)

def construct_spinor_weights() -> np.ndarray:
    """
    Construct the 168 spinor weights of E8.

    These are the ±(1/3) permutations of [1,1,1,1,1,1,-2,-2,-2] projected to 8D.

    Returns:
        np.ndarray: Array of 168 spinor weight vectors in 8D space
    """
    base_pattern = [1, 1, 1, 1, 1, 1, -2, -2, -2]
    permutations = generate_permutations(base_pattern)

    spinor_weights = []
    for perm in permutations:
        # Project to 8D and scale by 1/3
        weight_8d = project_to_trace_zero(perm) / 3.0

        # Add both positive and negative versions
        spinor_weights.append(weight_8d)
        spinor_weights.append(-weight_8d)

    return np.array(spinor_weights)

def construct_e8_simple_roots() -> np.ndarray:
    """
    Construct the 8 simple roots of E8.

    Returns:
        np.ndarray: Array of 8 simple roots in 8D space
    """
    # Simple roots of E8 in the standard basis
    simple_roots = [
        np.array([-1, 1, 0, 0, 0, 0, 0, 0]),    # α₁
        np.array([0, -1, 1, 0, 0, 0, 0, 0]),    # α₂
        np.array([0, 0, -1, 1, 0, 0, 0, 0]),    # α₃
        np.array([0, 0, 0, -1, 1, 0, 0, 0]),    # α₄
        np.array([0, 0, 0, 0, -1, 1, 0, 0]),    # α₅
        np.array([0, 0, 0, 0, 0, -1, 1, 0]),    # α₆
        np.array([0, 0, 0, 0, 0, 0, -1, 1]),    # α₇
        np.array([1, 1, 1, 1, 1, 1, -2, -2]) / 3.0  # α₈ (special root)
    ]

    return np.array(simple_roots)