"""
E8 ⊕ E8 differentiable layer.

Uses the honest E8 ⊕ E8 root system and an adjacency matrix from the
selected convention. The clustering coefficient is logged at construction
time but is NOT asserted against any pre-determined target.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from e8_heterotic.core.cache import get_e8_cache
from e8_heterotic.core.constants import E8XE8_EMBEDDING_DIM, E8XE8_ROOTS
from e8_heterotic.utils.device import get_sparse_safe_device

logger = logging.getLogger(__name__)


class E8E8Layer(nn.Module):
    """480-dimensional E8 ⊕ E8 root-graph layer.

    Parameters
    ----------
    input_dim, output_dim:
        Linear projection dimensions.
    propagation_steps:
        Number of times the adjacency-mediated propagation step is applied.
    adjacency_convention:
        Which adjacency convention to use. ``"A"`` is the canonical
        ⟨α, β⟩ = +1 root-system graph.
    device:
        Optional explicit PyTorch device.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        propagation_steps: int = 3,
        adjacency_convention: str = "A",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.propagation_steps = propagation_steps
        self.adjacency_convention = adjacency_convention
        self.device = device or get_sparse_safe_device()
        self.full_dim = E8XE8_ROOTS  # 480

        cache = get_e8_cache(adjacency_convention=adjacency_convention)
        roots_np = cache.get_root_system()
        adjacency_np = cache.get_adjacency_matrix()
        properties = cache.compute_network_properties()

        clustering = float(properties["clustering_coefficient"])
        logger.info(
            "E8 ⊕ E8 layer initialised (Convention %s): "
            "edges=%d triangles=%d wedges=%d C_global=%.10f",
            adjacency_convention,
            properties["num_edges"],
            properties["num_triangles"],
            properties["num_wedges"],
            clustering,
        )
        self.clustering_coefficient = clustering

        if roots_np.shape != (E8XE8_ROOTS, E8XE8_EMBEDDING_DIM):
            raise ValueError(
                f"unexpected root system shape {roots_np.shape}; "
                f"expected ({E8XE8_ROOTS}, {E8XE8_EMBEDDING_DIM})"
            )

        self.root_vectors = torch.from_numpy(roots_np).float().to(self.device)

        adjacency_tensor = torch.from_numpy(adjacency_np.astype(np.float32))
        if self.device.type == "mps":
            self.adjacency = adjacency_tensor.to(self.device)
            self.use_sparse = False
        else:
            self.adjacency = adjacency_tensor.to_sparse_coo().to(self.device).coalesce()
            self.use_sparse = True

        self._precompute_normalized_adjacency()

        self.input_projection = nn.Linear(input_dim, self.full_dim).to(self.device)
        self.output_projection = nn.Linear(self.full_dim, output_dim).to(self.device)
        self.propagation_weight = nn.Parameter(torch.tensor(1.0, device=self.device))

        self.sqrt_dim = float(np.sqrt(self.full_dim))

    # ------------------------------------------------------------------

    def _precompute_normalized_adjacency(self) -> None:
        if self.use_sparse:
            dense = self.adjacency.to_dense()
        else:
            dense = self.adjacency

        avg_degree = dense.sum(dim=1).mean().clamp(min=1e-8).item()
        scaling = 1.0 / float(np.sqrt(avg_degree))
        self.adjacency_normalized = (dense * scaling).to(self.device)

    def get_clustering_coefficient(self) -> float:
        return self.clustering_coefficient

    def geometric_propagation(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(-1, feature_dim)
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")

        a = self.adjacency_normalized.to(dtype=x.dtype, device=x.device)
        out = torch.mm(x, a.t()) * torch.abs(self.propagation_weight)

        if len(original_shape) == 3:
            out = out.view(batch_size, seq_len, -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = torch.tanh(x) * np.sqrt(2)

        for _ in range(self.propagation_steps):
            x = self.geometric_propagation(x)
            x = torch.tanh(x / self.sqrt_dim) * self.sqrt_dim

        return self.output_projection(x)

    def get_e8_roots(self) -> torch.Tensor:
        return self.root_vectors


def create_e8_layer(input_dim: int, output_dim: int, **kwargs) -> E8E8Layer:
    return E8E8Layer(input_dim=input_dim, output_dim=output_dim, **kwargs)
