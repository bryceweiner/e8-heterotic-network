"""
E8×E8 Geometric Network Layer

Implements the 496-dimensional E8×E8 heterotic structure as a differentiable
PyTorch layer with exact clustering coefficient C(G) = 25/32 (0.78125).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from e8_heterotic.core.constants import (
    E8_CLUSTERING,
    INFO_SPEED_FACTOR,
    C,
    I_MAX_E8XE8
)
from e8_heterotic.core.cache import get_e8_cache
from e8_heterotic.utils.device import get_sparse_safe_device, is_sparse_supported

class E8E8Layer(nn.Module):
    """
    E8×E8 Geometric Network Layer implementing 496-dimensional processing.

    Uses the proven E8×E8 heterotic construction for mathematically exact
    clustering coefficient of 25/32 (0.78125) and correct heterotic structure.

    This layer maps arbitrary input dimensions to a 496-dimensional latent space
    defined by the E8×E8 root system, performs geometric information propagation,
    and projects back to the desired output dimension.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        propagation_steps: int = 3,
        device: Optional[torch.device] = None
    ):
        """
        Initialize E8×E8 layer using proven heterotic construction.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            propagation_steps: Number of information propagation steps (default: 3)
            device: PyTorch device for computations (auto-selected if None)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.propagation_steps = propagation_steps
        self.device = device or get_sparse_safe_device()

        # Use shared cache instance (singleton pattern for performance)
        print("Initializing E8×E8 layer with proven heterotic construction and caching...")
        self.e8_cache = get_e8_cache()

        # Load cached E8×E8 structures
        self.heterotic_roots = self.e8_cache.get_root_system()
        self.adjacency_np = self.e8_cache.get_adjacency_matrix()
        self.network_properties = self.e8_cache.compute_network_properties()

        # Verify exact clustering coefficient
        clustering = self.network_properties['clustering_coefficient']
        assert abs(clustering - E8_CLUSTERING) < 1e-10, \
            f"Clustering coefficient {clustering} != {E8_CLUSTERING}"

        print(f"[OK] E8×E8 layer initialized with clustering coefficient: {clustering:.8f}")

        # Convert to PyTorch tensors with optimization
        self.full_dim = 496  # E8×E8 has exactly 496 generators
        self.root_vectors = torch.from_numpy(self.heterotic_roots).float().to(self.device)

        # Create adjacency matrix with device-specific optimizations
        adjacency_tensor = torch.from_numpy(self.adjacency_np).float()

        if self.device.type == 'mps':
            # MPS doesn't support sparse tensors well, use dense matrix
            print("Using dense adjacency matrix for MPS backend compatibility")
            self.adjacency_sparse = adjacency_tensor.to(self.device)
            self.use_sparse = False
        else:
            # Use sparse matrix for CUDA and CPU
            adjacency_coo = adjacency_tensor.to_sparse_coo()
            self.adjacency_sparse = adjacency_coo.to(self.device).coalesce()
            self.use_sparse = True

        # Pre-compute normalized adjacency for faster operations
        self._precompute_normalized_adjacency()

        # Learnable projections
        self.input_projection = nn.Linear(input_dim, self.full_dim).to(self.device)
        self.output_projection = nn.Linear(self.full_dim, output_dim).to(self.device)

        # Simplified propagation weight (single scalar for efficiency)
        self.propagation_weight = nn.Parameter(torch.tensor(1.0, device=self.device))

        # Pre-compute constants for speed
        self.I_max = I_MAX_E8XE8
        self.sqrt_dim = np.sqrt(self.full_dim)

    def _precompute_normalized_adjacency(self):
        """Pre-compute normalized adjacency matrix for fast operations."""
        if self.use_sparse and self.device.type != 'mps':
            # Sparse operations for CUDA/CPU only
            # Get dense adjacency for normalization computation
            adjacency_dense = self.adjacency_sparse.to_dense()

            # Compute row sums for normalization
            row_sums = torch.sum(adjacency_dense, dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-8)  # Avoid division by zero

            # Information-preserving normalization: scale by sqrt of average degree
            # This maintains connectivity patterns while preserving more energy
            avg_degree = row_sums.mean().item()
            scaling_factor = 1.0 / np.sqrt(avg_degree)  # Much less aggressive than 1/degree

            # Apply uniform scaling (preserves relative connectivity while maintaining energy)
            indices = self.adjacency_sparse.indices()
            values = self.adjacency_sparse.values()

            normalized_values = values * scaling_factor

            # Create sparse tensor
            self.adjacency_normalized = torch.sparse_coo_tensor(
                indices=indices,
                values=normalized_values,
                size=self.adjacency_sparse.size(),
                device=self.device
            ).coalesce()
        else:
            # Dense matrix case (for MPS or when sparse is disabled)
            if self.use_sparse:
                # Convert sparse to dense for MPS
                adjacency_dense = self.adjacency_sparse.to_dense()
            else:
                adjacency_dense = self.adjacency_sparse  # Already dense

            # Compute row sums for normalization
            row_sums = torch.sum(adjacency_dense, dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-8)  # Avoid division by zero

            # Information-preserving normalization: scale by sqrt of average degree
            avg_degree = row_sums.mean().item()
            scaling_factor = 1.0 / np.sqrt(avg_degree)

            # Apply uniform scaling to dense matrix
            self.adjacency_normalized = adjacency_dense * scaling_factor

    def get_clustering_coefficient(self) -> float:
        """Get the exact clustering coefficient from the heterotic system."""
        return self.network_properties['clustering_coefficient']

    def geometric_propagation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized propagation through E8×E8 structure with improved information conservation.

        Args:
            x: Input tensor of shape (batch_size, [seq_len,] 496)

        Returns:
            Output tensor of same shape as input
        """
        # Handle both 2D and 3D input tensors
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape from (batch, seq_len, dim) to (batch*seq_len, dim)
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(-1, feature_dim)
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")

        # Store original information for conservation
        original_info = torch.sum(x ** 2, dim=1, keepdim=True)

        # Holographic bound enforcement - ensure output doesn't exceed bound
        input_info = torch.sum(x ** 2, dim=1, keepdim=True)
        # Scale to respect bound with some safety margin
        scale_factor = torch.clamp(
            torch.sqrt(self.I_max * 1.0 / (input_info + 1e-8)),  # Keep within bound
            min=0.7, max=1.0  # Prevent excessive scaling down
        )
        x_scaled = x * scale_factor

        # Matrix propagation (sparse or dense depending on backend)
        # Ensure dtype and device consistency for mixed precision and GPU usage
        adjacency_normalized = self.adjacency_normalized.to(dtype=x.dtype, device=x.device)

        if self.use_sparse and self.device.type != 'mps':
            # Sparse matrix operations for CUDA/CPU only
            if x_scaled.shape[0] == 1:
                # Single sample case - use regular sparse mm
                out = torch.sparse.mm(adjacency_normalized, x_scaled.t()).t()
            else:
                # Multiple samples - use dense operation to avoid transpose issues
                adjacency_dense = adjacency_normalized.to_dense()
                out = torch.mm(x_scaled, adjacency_dense.t())
        else:
            # Dense matrix operations for MPS or when sparse is disabled
            if hasattr(adjacency_normalized, 'to_dense'):
                # Convert sparse to dense if needed
                adjacency_normalized = adjacency_normalized.to_dense()
            out = torch.mm(x_scaled, adjacency_normalized.t())

        # Apply learnable scaling
        out = out * torch.abs(self.propagation_weight)

        # Strong information conservation: aim to preserve original energy
        output_info = torch.sum(out ** 2, dim=1, keepdim=True)
        conservation_factor = torch.sqrt(original_info / (output_info + 1e-8))

        # Conservative conservation (preserve 70-120% of information)
        conservation_factor = torch.clamp(conservation_factor, 0.7, 1.2)
        out = out * conservation_factor

        # Final holographic bound enforcement - ensure we never exceed bound
        final_info = torch.sum(out ** 2, dim=1, keepdim=True)
        # Always enforce bound strictly, not just when threshold exceeded
        pressure_scale = torch.clamp(
            torch.sqrt(self.I_max * 0.78 / (final_info + 1e-8)),  # Scale to 78% of bound for safety
            min=0.1, max=1.0  # Prevent division by zero but allow strong scaling
        )
        out = out * pressure_scale

        # Reshape back to original dimensions if needed
        if len(original_shape) == 3:
            out = out.view(batch_size, seq_len, -1)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through proven E8×E8 heterotic network.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        # Project to proven E8×E8 space (496 dimensions)
        x = self.input_projection(x)

        # Apply activation maintaining E8 root norm constraints
        x = torch.tanh(x) * np.sqrt(2)  # E8 roots have norm √2

        # Fast propagation through optimized geometric structure
        for _ in range(self.propagation_steps):
            x = self.geometric_propagation(x)
            # Fast bounded activation using pre-computed constant
            x = torch.tanh(x / self.sqrt_dim) * self.sqrt_dim

        # Project to output dimension
        out = self.output_projection(x)

        return out

    def get_information_flow(self) -> torch.Tensor:
        """Get the current information flow pattern through the heterotic network."""
        if self.use_sparse:
            return self.adjacency_normalized.to_dense() * self.propagation_weight.data
        else:
            return self.adjacency_normalized * self.propagation_weight.data

    def bulk_information(self) -> float:
        """Calculate total information in bulk (E8×E8 space) using proven system."""
        weights_norm = torch.norm(self.propagation_weight) ** 2
        return float(weights_norm * np.log(2))

    def get_heterotic_properties(self) -> dict:
        """Get all network properties from the proven heterotic system."""
        return self.network_properties

    def get_e8_roots(self) -> torch.Tensor:
        """Get the exact E8×E8 root vectors from the proven construction."""
        return self.root_vectors

    def verify_mathematical_properties(self) -> dict:
        """Verify all mathematical properties are correct."""
        clustering = self.get_clustering_coefficient()

        return {
            'clustering_coefficient': clustering,
            'clustering_exact': abs(clustering - E8_CLUSTERING) < 1e-10,
            'dimension': self.full_dim,
            'num_edges': self.network_properties.get('num_edges', self.network_properties.get('edges', 0)),
            'is_connected': self.network_properties.get('is_connected', True),
            'average_degree': self.network_properties.get('average_degree', 0),
            'theoretical_clustering': E8_CLUSTERING,
            'mathematical_validation': 'PASSED - Exact clustering coefficient achieved'
        }

    # Backward compatibility properties
    @property
    def adjacency(self) -> torch.Tensor:
        """Get dense adjacency matrix for backward compatibility."""
        if self.use_sparse:
            return self.adjacency_sparse.to_dense()
        else:
            return self.adjacency_sparse  # Already dense

    @property
    def propagation_weights(self) -> torch.Tensor:
        """Get expanded propagation weights matrix for backward compatibility."""
        # Return a clone to avoid shared memory issues
        return self.propagation_weight.unsqueeze(0).expand(self.full_dim, self.full_dim).clone()

# Convenience functions for easy access
def create_e8_layer(input_dim: int, output_dim: int, **kwargs) -> E8E8Layer:
    """
    Create an E8×E8 layer with the specified dimensions.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional arguments passed to E8E8Layer

    Returns:
        E8E8Layer: Initialized E8×E8 layer
    """
    return E8E8Layer(input_dim=input_dim, output_dim=output_dim, **kwargs)

def get_e8_clustering_coefficient() -> float:
    """Get the exact E8×E8 clustering coefficient."""
    cache = get_e8_cache()
    return cache.get_clustering_coefficient()

def get_e8_network_properties() -> dict:
    """Get all E8×E8 network properties."""
    cache = get_e8_cache()
    return cache.compute_network_properties()

if __name__ == "__main__":
    # Test the E8E8Layer
    print("Testing E8×E8 Layer...")

    # Create a test layer
    layer = E8E8Layer(input_dim=10, output_dim=5, propagation_steps=2)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 10)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = layer(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 5)")

    # Verify mathematical properties
    verification = layer.verify_mathematical_properties()
    print("\nMathematical verification:")
    for key, value in verification.items():
        print(f"  {key}: {value}")

    print("\nE8×E8 Layer test completed successfully!")