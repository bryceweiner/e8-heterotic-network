"""
Constants for the E8×E8 heterotic root system analysis.

This module deliberately does NOT export a returnable clustering coefficient.
The clustering coefficient of the E8×E8 root graph is a derived quantity
that must flow from the adjacency matrix; see ``e8_heterotic.core.clustering``.
"""

import numpy as np
from fractions import Fraction

# =============================================================================
# E8 / E8×E8 STRUCTURAL CONSTANTS
# =============================================================================

# Number of roots in a single E8 root system.
E8_ROOTS = 240

# Number of simple roots / Cartan generators for E8.
E8_CARTAN = 8

# Number of roots in E8 ⊕ E8 (direct sum, not tensor product).
E8XE8_ROOTS = 2 * E8_ROOTS  # 480

# Embedding dimension of the direct-sum root system.
E8XE8_EMBEDDING_DIM = 16

# Squared norm of every E8 root in the standard normalization.
E8_ROOT_NORM_SQUARED = 2.0
E8_ROOT_NORM = np.sqrt(E8_ROOT_NORM_SQUARED)

# =============================================================================
# REFERENCE / LITERATURE VALUES
# =============================================================================

# 25/32 is a value claimed in some literature for the E8×E8 heterotic
# clustering coefficient. It is retained here ONLY as a reference value to
# compare computed results against. It must NEVER be returned by any function
# that purports to compute the clustering coefficient.
E8_CLUSTERING_LITERATURE_CLAIM = 25.0 / 32.0
E8_CLUSTERING_LITERATURE_FRACTION = Fraction(25, 32)

# =============================================================================
# COMPUTATIONAL TOLERANCES
# =============================================================================

# Inner products between E8 roots are small integers or half-integers in
# float64; 1e-9 is comfortably tighter than any expected drift while still
# absorbing rounding from the half-integer (spinor) construction.
INNER_PRODUCT_TOLERANCE = 1e-9
NORM_TOLERANCE = 1e-9
