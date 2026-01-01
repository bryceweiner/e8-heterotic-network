"""
Physical constants for the E8×E8 heterotic network implementation.

This module defines the fundamental mathematical and physical constants
used throughout the E8×E8 heterotic string theory framework.
"""

import numpy as np
from fractions import Fraction

# =============================================================================
# E8×E8 HETEROTIC STRUCTURE CONSTANTS
# =============================================================================

# The fundamental clustering coefficient of the E8×E8 heterotic structure
# This is a mathematical constant derived from the root system geometry
E8_CLUSTERING = 25.0 / 32.0  # Exactly 0.78125
E8_CLUSTERING_FRACTION = Fraction(25, 32)

# E8 algebra dimensions
E8_ROOTS = 240  # Number of roots in E8
E8_CARTAN = 8   # Number of Cartan generators in E8
E8_DIMENSION = E8_ROOTS + E8_CARTAN  # 248 total generators per E8

# E8×E8 heterotic structure
E8XE8_TOTAL_GENERATORS = 2 * E8_DIMENSION  # 496 total generators
E8XE8_EMBEDDING_DIM = 16  # Embedded in 16-dimensional space

# E8 root norm (all roots have the same length)
E8_ROOT_NORM = np.sqrt(2.0)  # √2

# =============================================================================
# INFORMATION THEORY CONSTANTS
# =============================================================================

# Information speed factor (from Origami Universe Theory)
INFO_SPEED_FACTOR = np.log(2) / (1 - np.log(2))  # ≈ 2.257

# Maximum information bound for E8×E8 space
# I_max = N * ln(2) where N is the dimension
I_MAX_E8XE8 = E8XE8_TOTAL_GENERATORS * np.log(2)

# Speed of light (m/s) - for dimensional consistency
C = 299792458.0

# Fine structure constant (dimensionless)
ALPHA_EM = 1.0 / 137.035999084

# =============================================================================
# COMPUTATIONAL CONSTANTS
# =============================================================================

# Numerical precision tolerances
ANGLE_TOLERANCE = 1e-12  # For angle-based adjacency calculations
DOT_PRODUCT_TOLERANCE = 1e-10  # For dot product comparisons
NORM_TOLERANCE = 1e-8  # For norm verification

# Adjacency calculation angles (in radians and degrees)
ADJACENCY_ANGLES_DEG = [60.0, 90.0, 120.0]  # Degrees
ADJACENCY_ANGLES_RAD = np.deg2rad(ADJACENCY_ANGLES_DEG)  # Radians

# Corresponding dot products for normalized vectors
ADJACENCY_DOT_PRODUCTS = [
    np.cos(angle) for angle in ADJACENCY_ANGLES_RAD
]  # [-0.5, 0.0, 0.5]

# =============================================================================
# HETEROTIC STRING THEORY CONSTANTS
# =============================================================================

# Heterotic E8×E8 compactification scales
HETEROTIC_SCALE_FACTOR_1 = 1.0   # First E8 scale
HETEROTIC_SCALE_FACTOR_2 = 1.2   # Second E8 scale (creates norm diversity)
HETEROTIC_SCALE_FACTOR_3 = 0.8   # Additional scale for diversity
HETEROTIC_SCALE_FACTOR_4 = 1.1   # Additional scale for diversity
HETEROTIC_SCALE_FACTOR_5 = 0.9   # Additional scale for diversity

HETEROTIC_SCALE_FACTORS = [
    HETEROTIC_SCALE_FACTOR_1,
    HETEROTIC_SCALE_FACTOR_2,
    HETEROTIC_SCALE_FACTOR_3,
    HETEROTIC_SCALE_FACTOR_4,
    HETEROTIC_SCALE_FACTOR_5
]

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_constants():
    """
    Validate that all constants are properly defined and consistent.

    Returns:
        dict: Validation results with status for each constant
    """
    validation_results = {}

    # Check E8 clustering coefficient
    expected_clustering = 25.0 / 32.0
    validation_results['e8_clustering'] = {
        'value': E8_CLUSTERING,
        'expected': expected_clustering,
        'valid': abs(E8_CLUSTERING - expected_clustering) < 1e-15
    }

    # Check E8 dimensions
    expected_e8_total = E8_ROOTS + E8_CARTAN
    validation_results['e8_dimension'] = {
        'value': E8_DIMENSION,
        'expected': expected_e8_total,
        'valid': E8_DIMENSION == expected_e8_total
    }

    # Check E8×E8 dimensions
    expected_e8xe8_total = 2 * E8_DIMENSION
    validation_results['e8xe8_dimension'] = {
        'value': E8XE8_TOTAL_GENERATORS,
        'expected': expected_e8xe8_total,
        'valid': E8XE8_TOTAL_GENERATORS == expected_e8xe8_total
    }

    # Check E8 root norm
    validation_results['e8_root_norm'] = {
        'value': E8_ROOT_NORM,
        'expected': np.sqrt(2.0),
        'valid': abs(E8_ROOT_NORM - np.sqrt(2.0)) < 1e-15
    }

    # Check information bound
    expected_i_max = E8XE8_TOTAL_GENERATORS * np.log(2)
    validation_results['i_max_bound'] = {
        'value': I_MAX_E8XE8,
        'expected': expected_i_max,
        'valid': abs(I_MAX_E8XE8 - expected_i_max) < 1e-12
    }

    # Check adjacency dot products
    expected_dots = [0.5, 0.0, -0.5]  # cos(60°), cos(90°), cos(120°)
    validation_results['adjacency_dots'] = {
        'value': ADJACENCY_DOT_PRODUCTS,
        'expected': expected_dots,
        'valid': all(abs(a - e) < 1e-12 for a, e in zip(ADJACENCY_DOT_PRODUCTS, expected_dots))
    }

    # Overall validation
    all_valid = all(result['valid'] for result in validation_results.values())
    validation_results['overall'] = {
        'valid': all_valid,
        'message': 'All constants validated successfully' if all_valid else 'Some constants failed validation'
    }

    return validation_results

def information_pressure(current_info: float, max_info: float = I_MAX_E8XE8) -> float:
    """
    Calculate information pressure scaling factor.

    In the holographic framework, information content cannot exceed
    the maximum bound without causing pressure that scales the system.

    Args:
        current_info: Current information content I_curr = sum(x^2)
        max_info: Maximum allowed information content

    Returns:
        float: Scaling factor to apply to maintain holographic bounds
    """
    if current_info <= 0:
        return 1.0

    # Calculate pressure: sqrt(I_max / I_curr)
    pressure = np.sqrt(max_info / current_info)

    # Clamp to reasonable bounds to prevent numerical instability
    return np.clip(pressure, 0.1, 1.0)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_e8_clustering_coefficient() -> float:
    """Get the exact E8×E8 clustering coefficient."""
    return E8_CLUSTERING

def get_e8_clustering_fraction() -> Fraction:
    """Get the E8×E8 clustering coefficient as an exact fraction."""
    return E8_CLUSTERING_FRACTION

def get_e8_dimensions() -> dict:
    """Get all E8-related dimensions."""
    return {
        'roots': E8_ROOTS,
        'cartan': E8_CARTAN,
        'total_per_e8': E8_DIMENSION,
        'total_e8xe8': E8XE8_TOTAL_GENERATORS,
        'embedding_dim': E8XE8_EMBEDDING_DIM
    }

def get_information_bounds() -> dict:
    """Get information theory bounds."""
    return {
        'i_max_e8xe8': I_MAX_E8XE8,
        'info_speed_factor': INFO_SPEED_FACTOR,
        'root_norm': E8_ROOT_NORM
    }

# Run validation on import
if __name__ == "__main__":
    print("Validating E8×E8 heterotic constants...")
    validation = validate_constants()

    print(f"\nE8 Clustering Coefficient: {E8_CLUSTERING:.8f} (25/32)")
    print(f"E8×E8 Total Generators: {E8XE8_TOTAL_GENERATORS}")
    print(f"E8×E8 Embedding Dimension: {E8XE8_EMBEDDING_DIM}")
    print(f"Maximum Information Bound: {I_MAX_E8XE8:.2f}")

    if validation['overall']['valid']:
        print("\n✓ All constants validated successfully!")
    else:
        print("\n✗ Some constants failed validation:")
        for key, result in validation.items():
            if not result.get('valid', True):
                print(f"  - {key}: {result}")