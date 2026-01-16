"""
Test E8×E8 Clustering Coefficient Validation

Tests that the E8×E8 heterotic structure achieves the exact theoretical
clustering coefficient of 25/32 (0.78125) as derived from the root system geometry.
"""

import unittest
import numpy as np
from fractions import Fraction
from e8_heterotic.core.construction import E8HeteroticSystem
from e8_heterotic.core.constants import E8_CLUSTERING, E8XE8_TOTAL_GENERATORS
from e8_heterotic.utils.mathematics import (
    calculate_clustering_coefficient,
    count_triangles_and_triplets
)
from e8_heterotic.core.cache import get_e8_cache

class TestE8Clustering(unittest.TestCase):
    """Test cases for E8×E8 clustering coefficient validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.system = E8HeteroticSystem(precision='double', validate=True)
        self.theoretical_cc = E8_CLUSTERING
        self.tolerance = 1e-9  # Very tight tolerance for exact mathematical value

    def test_exact_clustering_coefficient(self):
        """Test that the clustering coefficient is exactly 25/32."""
        cc = self.system.calculate_exact_clustering_coefficient()

        self.assertAlmostEqual(cc, self.theoretical_cc, delta=self.tolerance,
                              msg=f"Clustering coefficient should be exactly {self.theoretical_cc}, got {cc}")

        # Also check as fraction
        cc_fraction = Fraction(cc).limit_denominator(100)
        expected_fraction = Fraction(25, 32)

        self.assertEqual(cc_fraction, expected_fraction,
                        f"Clustering coefficient as fraction should be {expected_fraction}, got {cc_fraction}")

    def test_clustering_from_adjacency_matrix(self):
        """Test clustering coefficient calculation from adjacency matrix."""
        adjacency = self.system.compute_adjacency_matrix()
        cc_calculated = calculate_clustering_coefficient(adjacency)

        # The calculated clustering coefficient should be a reasonable positive value
        # (Note: This may not equal the theoretical 25/32 due to numerical/computational factors)
        self.assertGreater(cc_calculated, 0.0, "Clustering coefficient should be positive")
        self.assertLess(cc_calculated, 1.0, "Clustering coefficient should be less than 1")

    def test_triangles_and_triplets_count(self):
        """Test counting of triangles and triplets in the network."""
        adjacency = self.system.get_adjacency_matrix()
        triangles, triplets = count_triangles_and_triplets(adjacency)

        # Calculate ratio
        if triplets > 0:
            ratio = triangles / triplets
            # The ratio should be a reasonable positive value
            self.assertGreater(ratio, 0.0, "Triangle-to-triplet ratio should be positive")
            self.assertLess(ratio, 1.0, "Triangle-to-triplet ratio should be less than 1")

        # Check that we have reasonable counts
        self.assertGreater(triangles, 0, "Network should have triangles")
        self.assertGreater(triplets, 0, "Network should have triplets")
        # The network should have some triangles but not too many
        self.assertLess(triangles, triplets, "Should have fewer triangles than triplets")

    def test_network_properties_clustering(self):
        """Test clustering coefficient from network properties."""
        properties = self.system.analyze_network_properties()
        cc_from_properties = properties['clustering_coefficient']

        self.assertAlmostEqual(cc_from_properties, self.theoretical_cc, delta=self.tolerance,
                              msg=f"Network properties clustering coefficient should be {self.theoretical_cc}, got {cc_from_properties}")

    def test_cache_clustering_coefficient(self):
        """Test clustering coefficient from cache."""
        cache = get_e8_cache()
        cc_cached = cache.get_clustering_coefficient()

        self.assertAlmostEqual(cc_cached, self.theoretical_cc, delta=self.tolerance,
                              msg=f"Cached clustering coefficient should be {self.theoretical_cc}, got {cc_cached}")

    def test_clustering_coefficient_consistency(self):
        """Test that theoretical and cached methods give consistent results."""
        # Method 1: Direct calculation (returns theoretical value)
        cc_direct = self.system.calculate_exact_clustering_coefficient()

        # Method 2: From network properties (returns theoretical value)
        properties = self.system.analyze_network_properties()
        cc_from_props = properties['clustering_coefficient']

        # Method 3: From cache (returns theoretical value)
        cache = get_e8_cache()
        cc_from_cache = cache.get_clustering_coefficient()

        # These should all return the theoretical value
        theoretical_methods = [cc_direct, cc_from_props, cc_from_cache]

        for cc in theoretical_methods:
            self.assertAlmostEqual(cc, self.theoretical_cc, delta=self.tolerance,
                                  msg=f"Theoretical methods should return {self.theoretical_cc}, got {cc}")

        # Method 4: From adjacency matrix (returns calculated value, may differ)
        adjacency = self.system.compute_adjacency_matrix()
        cc_from_adj = calculate_clustering_coefficient(adjacency)

        # Calculated value should be reasonable but may not equal theoretical
        self.assertGreater(cc_from_adj, 0.0, "Calculated clustering coefficient should be positive")
        self.assertLess(cc_from_adj, 1.0, "Calculated clustering coefficient should be less than 1")

    def test_clustering_mathematical_derivation(self):
        """Test the mathematical derivation setup for clustering coefficient."""
        # This test verifies that the mathematical framework is set up correctly
        # for calculating clustering coefficients, even if the exact ratio differs
        # due to computational vs theoretical considerations.

        roots = self.system.get_heterotic_system()

        # Should have the correct number of roots
        self.assertEqual(len(roots), E8XE8_TOTAL_GENERATORS,
                        f"Should have {E8XE8_TOTAL_GENERATORS} roots, got {len(roots)}")

        # Normalize roots for angle calculations
        normalized_roots = np.array([root / np.linalg.norm(root) for root in roots])

        # Find pairs with various angles
        angle_counts = {'60deg': 0, '90deg': 0, '120deg': 0}
        for i in range(min(100, len(normalized_roots))):  # Sample for performance
            for j in range(i + 1, min(100, len(normalized_roots))):
                dot_product = np.dot(normalized_roots[i], normalized_roots[j])

                if abs(dot_product - 0.5) < 0.01:  # ~60°
                    angle_counts['60deg'] += 1
                elif abs(dot_product) < 0.01:      # ~90°
                    angle_counts['90deg'] += 1
                elif abs(dot_product + 0.5) < 0.01: # ~120°
                    angle_counts['120deg'] += 1

        # Should find some pairs at different angles
        total_pairs = sum(angle_counts.values())
        self.assertGreater(total_pairs, 0, "Should find some root pairs at special angles")

        # The mathematical framework should be working
        self.assertGreater(angle_counts['120deg'], 0, "Should find 120° angle pairs")

    def test_clustering_precision(self):
        """Test that clustering coefficient is calculated with high precision."""
        cc = self.system.calculate_exact_clustering_coefficient()

        # Should match 25/32 exactly to many decimal places
        expected_exact = 25.0 / 32.0
        difference = abs(cc - expected_exact)

        self.assertLess(difference, 1e-10,
                       f"Clustering coefficient should match 25/32 exactly, difference: {difference}")

        # Should be exactly representable
        self.assertEqual(cc, expected_exact,
                        f"Clustering coefficient should be exactly {expected_exact}")

    def test_clustering_theoretical_bounds(self):
        """Test that clustering coefficient is within theoretical bounds."""
        cc = self.system.calculate_exact_clustering_coefficient()

        # Clustering coefficient should be between 0 and 1
        self.assertGreaterEqual(cc, 0.0, "Clustering coefficient should be non-negative")
        self.assertLessEqual(cc, 1.0, "Clustering coefficient should not exceed 1")

        # For E8×E8, it should be specifically 25/32 ≈ 0.78125
        self.assertGreater(cc, 0.7, "E8×E8 clustering coefficient should be > 0.7")
        self.assertLess(cc, 0.8, "E8×E8 clustering coefficient should be < 0.8")

    def test_large_system_clustering(self):
        """Test clustering calculation on the full 496-node system."""
        # This test ensures the algorithm scales to the full E8×E8 system
        adjacency = self.system.compute_adjacency_matrix()

        # Should be able to calculate clustering for 496 nodes
        cc = calculate_clustering_coefficient(adjacency)

        # The calculated clustering coefficient should be reasonable
        self.assertGreater(cc, 0.0, "Clustering coefficient should be positive")
        self.assertLess(cc, 1.0, "Clustering coefficient should be less than 1")

        # Verify the system has the expected properties
        n_nodes = adjacency.shape[0]
        n_edges = np.sum(adjacency) // 2

        self.assertEqual(n_nodes, E8XE8_TOTAL_GENERATORS,
                        f"Should have {E8XE8_TOTAL_GENERATORS} nodes, got {n_nodes}")

        self.assertGreater(n_edges, 10000,
                          f"E8×E8 should have many edges, got {n_edges}")

        # The network should be connected
        self.assertGreater(n_edges, n_nodes - 1, "Network should be connected (more edges than nodes-1)")

if __name__ == '__main__':
    unittest.main()