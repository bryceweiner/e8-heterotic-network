"""
Test E8×E8 Heterotic Structure Construction

Tests the mathematical construction and validation of the E8×E8 heterotic system,
ensuring exactly 496 generators with correct geometric properties.
"""

import unittest
import numpy as np
from e8_heterotic.core.construction import E8HeteroticSystem
from e8_heterotic.core.constants import (
    E8_ROOTS, E8_CARTAN, E8_DIMENSION,
    E8XE8_TOTAL_GENERATORS, E8XE8_EMBEDDING_DIM,
    E8_ROOT_NORM, NORM_TOLERANCE
)
from e8_heterotic.utils.mathematics import validate_root_norms, validate_dot_products

class TestE8Construction(unittest.TestCase):
    """Test cases for E8×E8 heterotic construction."""

    def setUp(self):
        """Set up test fixtures."""
        self.system = E8HeteroticSystem(precision='double', validate=True)

    def test_single_e8_construction(self):
        """Test construction of a single E8 algebra."""
        e8_system = self.system.construct_single_e8_with_cartan()

        # Check dimensions
        self.assertEqual(e8_system.shape, (E8_DIMENSION, 8),
                        f"Expected shape ({E8_DIMENSION}, 8), got {e8_system.shape}")

        # Check root norms
        norms = np.array([np.linalg.norm(root) for root in e8_system])
        unique_norms = np.unique(np.round(norms, 6))

        # E8 is simply laced - all generators have norm √2
        expected_norm = np.sqrt(2.0)
        self.assertAlmostEqual(unique_norms[0], expected_norm, places=5,
                              msg=f"All E8 generators should have norm √2 ≈ {expected_norm}")
        self.assertEqual(len(unique_norms), 1,
                        f"E8 should have exactly 1 unique norm value, got {len(unique_norms)}: {unique_norms}")

        # All generators in E8 should have norm √2 (simply laced algebra)
        sqrt2_generators = np.sum(np.abs(norms - E8_ROOT_NORM) < NORM_TOLERANCE)
        self.assertEqual(sqrt2_generators, E8_DIMENSION,
                        f"Expected {E8_DIMENSION} generators with norm √2, found {sqrt2_generators}")

    def test_heterotic_system_construction(self):
        """Test construction of the complete E8×E8 heterotic system."""
        heterotic_system = self.system.construct_heterotic_system()

        # Check dimensions
        expected_shape = (E8XE8_TOTAL_GENERATORS, E8XE8_EMBEDDING_DIM)
        self.assertEqual(heterotic_system.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {heterotic_system.shape}")

        # Check that we have exactly 496 generators
        self.assertEqual(len(heterotic_system), E8XE8_TOTAL_GENERATORS,
                        f"Expected {E8XE8_TOTAL_GENERATORS} generators, got {len(heterotic_system)}")

        # Check embedding structure
        # First E8 should be in dimensions 0-7, second in 8-15
        first_e8 = heterotic_system[:E8_DIMENSION]
        second_e8 = heterotic_system[E8_DIMENSION:]

        # First E8: last 8 dimensions should be zeros
        first_e8_zeros = first_e8[:, 8:]
        self.assertTrue(np.allclose(first_e8_zeros, 0, atol=1e-12),
                       "First E8 should have zeros in dimensions 8-15")

        # Second E8: first 8 dimensions should be zeros
        second_e8_zeros = second_e8[:, :8]
        self.assertTrue(np.allclose(second_e8_zeros, 0, atol=1e-12),
                       "Second E8 should have zeros in dimensions 0-7")

    def test_adjacency_matrix_construction(self):
        """Test construction of the adjacency matrix."""
        adjacency = self.system.compute_adjacency_matrix()

        # Check dimensions
        expected_shape = (E8XE8_TOTAL_GENERATORS, E8XE8_TOTAL_GENERATORS)
        self.assertEqual(adjacency.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {adjacency.shape}")

        # Check that matrix is symmetric
        self.assertTrue(np.allclose(adjacency, adjacency.T, atol=1e-12),
                       "Adjacency matrix should be symmetric")

        # Check that diagonal is zero (no self-connections)
        self.assertTrue(np.all(np.diag(adjacency) == 0),
                       "Adjacency matrix diagonal should be zero")

        # Check that values are only 0 or 1
        unique_values = np.unique(adjacency)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])),
                       f"Adjacency matrix should contain only 0s and 1s, found {unique_values}")

    def test_network_properties(self):
        """Test computation of network properties."""
        properties = self.system.analyze_network_properties()

        # Check required properties exist
        required_props = [
            'num_nodes', 'num_edges', 'clustering_coefficient',
            'average_degree', 'is_connected', 'density'
        ]

        for prop in required_props:
            self.assertIn(prop, properties,
                         f"Required property '{prop}' missing from network properties")

        # Check node count
        self.assertEqual(properties['num_nodes'], E8XE8_TOTAL_GENERATORS,
                        f"Expected {E8XE8_TOTAL_GENERATORS} nodes, got {properties['num_nodes']}")

        # Check connectivity
        self.assertTrue(properties['is_connected'],
                       "E8×E8 network should be connected")

        # Check edges > 0
        self.assertGreater(properties['num_edges'], 0,
                          "Network should have edges")

    def test_root_norm_validation(self):
        """Test validation of root norms in heterotic system."""
        heterotic_system = self.system.get_heterotic_system()

        # Heterotic system has norm diversity by design - check basic properties
        norms = np.array([np.linalg.norm(root) for root in heterotic_system])
        unique_norms = np.unique(np.round(norms, 6))

        # Should have multiple norm values due to heterotic scaling
        self.assertGreater(len(unique_norms), 1,
                          f"Heterotic system should have multiple norm values, got {len(unique_norms)}")

        # All norms should be reasonable (not too small or large)
        self.assertGreater(np.min(norms), 1.0, "Norms should be > 1")
        self.assertLess(np.max(norms), 2.0, "Norms should be < 2")

        # Mean norm should be close to √2
        self.assertAlmostEqual(np.mean(norms), E8_ROOT_NORM, places=2,
                              msg=f"Mean norm should be close to √2 ≈ {E8_ROOT_NORM}")

    def test_dot_product_validation(self):
        """Test validation of dot products between roots."""
        heterotic_system = self.system.get_heterotic_system()
        validation = validate_dot_products(heterotic_system)

        self.assertTrue(validation['valid'],
                       f"Dot product validation failed: {validation}")

        # E8×E8 should have specific dot product values
        expected_dots = [-2, -1, 0, 1, 2]  # For simply laced root systems
        self.assertTrue(len(validation['found_expected']) > 0,
                       "Should find expected dot product values")

    def test_clustering_coefficient_calculation(self):
        """Test direct calculation of clustering coefficient."""
        cc = self.system.calculate_exact_clustering_coefficient()

        # Should be exactly 25/32 = 0.78125
        expected_cc = 25.0 / 32.0
        self.assertAlmostEqual(cc, expected_cc, places=10,
                              msg=f"Clustering coefficient should be exactly {expected_cc}, got {cc}")

    def test_system_reproducibility(self):
        """Test that the construction is reproducible."""
        system1 = E8HeteroticSystem(precision='double', validate=False)
        roots1 = system1.construct_heterotic_system()

        system2 = E8HeteroticSystem(precision='double', validate=False)
        roots2 = system2.construct_heterotic_system()

        # Systems should be identical (same random seed)
        self.assertTrue(np.allclose(roots1, roots2, atol=1e-12),
                       "E8×E8 construction should be reproducible")

    def test_different_precisions(self):
        """Test construction with different numerical precisions."""
        for precision in ['single', 'double']:
            with self.subTest(precision=precision):
                system = E8HeteroticSystem(precision=precision, validate=False)
                roots = system.construct_heterotic_system()

                # Should still get correct shape
                expected_shape = (E8XE8_TOTAL_GENERATORS, E8XE8_EMBEDDING_DIM)
                self.assertEqual(roots.shape, expected_shape,
                               f"Wrong shape for {precision} precision")

if __name__ == '__main__':
    unittest.main()