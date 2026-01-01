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

        self.assertAlmostEqual(cc_calculated, self.theoretical_cc, delta=self.tolerance,
                              msg=f"Calculated clustering coefficient should be {self.theoretical_cc}, got {cc_calculated}")

    def test_triangles_and_triplets_count(self):
        """Test counting of triangles and triplets in the network."""
        adjacency = self.system.get_adjacency_matrix()
        triangles, triplets = count_triangles_and_triplets(adjacency)

        # Calculate ratio
        if triplets > 0:
            ratio = triangles / triplets
            self.assertAlmostEqual(ratio, self.theoretical_cc, delta=self.tolerance,
                                  msg=f"Triangle-to-triplet ratio should be {self.theoretical_cc}, got {ratio}")

        # Check that we have reasonable counts
        self.assertGreater(triangles, 0, "Network should have triangles")
        self.assertGreater(triplets, 0, "Network should have triplets")
        self.assertGreater(triangles, triplets * self.theoretical_cc * 0.9,
                          "Should have approximately correct number of triangles")

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
        """Test that all methods give consistent clustering coefficient."""
        # Method 1: Direct calculation
        cc_direct = self.system.calculate_exact_clustering_coefficient()

        # Method 2: From adjacency matrix
        adjacency = self.system.compute_adjacency_matrix()
        cc_from_adj = calculate_clustering_coefficient(adjacency)

        # Method 3: From network properties
        properties = self.system.analyze_network_properties()
        cc_from_props = properties['clustering_coefficient']

        # Method 4: From cache
        cache = get_e8_cache()
        cc_from_cache = cache.get_clustering_coefficient()

        # All should be equal within tolerance
        ccs = [cc_direct, cc_from_adj, cc_from_props, cc_from_cache]

        for i, cc1 in enumerate(ccs):
            for j, cc2 in enumerate(ccs):
                if i != j:
                    self.assertAlmostEqual(cc1, cc2, delta=self.tolerance,
                                          msg=f"Clustering coefficients from different methods should match: {cc1} vs {cc2}")

    def test_clustering_mathematical_derivation(self):
        """Test the mathematical derivation of the clustering coefficient."""
        # The clustering coefficient C(G) = 25/32 is derived from the geometry
        # of the E8×E8 root system. This test verifies the key mathematical properties.

        roots = self.system.get_heterotic_system()

        # Normalize roots for angle calculations
        normalized_roots = np.array([root / np.linalg.norm(root) for root in roots])

        # Find pairs with 120° angle (dot product = -0.5)
        angle_120_pairs = []
        for i in range(len(normalized_roots)):
            for j in range(i + 1, len(normalized_roots)):
                dot_product = np.dot(normalized_roots[i], normalized_roots[j])
                if abs(dot_product + 0.5) < 1e-10:  # 120° angle
                    angle_120_pairs.append((i, j))

        # Should find many such pairs
        self.assertGreater(len(angle_120_pairs), 100,
                          f"Should find many 120° angle pairs, found {len(angle_120_pairs)}")

        # For a subset of these pairs, verify the triangle formation ratio
        sample_pairs = angle_120_pairs[:min(50, len(angle_120_pairs))]

        triangle_ratios = []
        for i, j in sample_pairs:
            # Count how many third roots form triangles with this pair
            triangles_with_pair = 0
            total_candidates = 0

            for k in range(len(normalized_roots)):
                if k != i and k != j:
                    angle_ik = np.dot(normalized_roots[i], normalized_roots[k])
                    angle_jk = np.dot(normalized_roots[j], normalized_roots[k])

                    # Count valid candidates
                    if abs(angle_ik) < 0.99 and abs(angle_jk) < 0.99:
                        total_candidates += 1

                        # Check for triangle (all angles 120°)
                        if (abs(angle_ik + 0.5) < 1e-10 and abs(angle_jk + 0.5) < 1e-10):
                            triangles_with_pair += 1

            if total_candidates > 0:
                ratio = triangles_with_pair / total_candidates
                triangle_ratios.append(ratio)

        # Average ratio should be close to 25/32
        if triangle_ratios:
            avg_ratio = np.mean(triangle_ratios)
            self.assertAlmostEqual(avg_ratio, self.theoretical_cc, delta=0.01,
                                  msg=f"Average triangle formation ratio should be ~{self.theoretical_cc}, got {avg_ratio}")

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

        self.assertAlmostEqual(cc, self.theoretical_cc, delta=self.tolerance,
                              msg=f"Full system clustering coefficient should be {self.theoretical_cc}, got {cc}")

        # Verify the system has the expected properties
        n_nodes = adjacency.shape[0]
        n_edges = np.sum(adjacency) // 2

        self.assertEqual(n_nodes, E8XE8_TOTAL_GENERATORS,
                        f"Should have {E8XE8_TOTAL_GENERATORS} nodes, got {n_nodes}")

        self.assertGreater(n_edges, 10000,
                          f"E8×E8 should have many edges, got {n_edges}")

if __name__ == '__main__':
    unittest.main()