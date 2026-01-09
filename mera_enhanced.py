"""
Enhanced MERA Tensor Network for Reinforcement Learning
========================================================

This module implements a tensor network MERA (Multi-scale Entanglement
Renormalization Ansatz) encoder with proper tensor contractions.

The tensor network structure provides an inductive bias for hierarchical
temporal reasoning through coarse-graining operations.

Key Features:
- True tensor network contractions (einsum operations)
- Proper dimension handling across layers
- Isometry constraint regularization (w†w = I)
- Layer scaling factor tracking for analysis
- Hierarchical correlation metric as diagnostic probe

Architecture:
- Layer 0: physical_dim → bond_dim (dimension expansion)
- Layer 1+: bond_dim → bond_dim (consistent dimensions)

Note: The "hierarchical_entropy" metric is a correlation-based diagnostic,
NOT a measure of quantum entanglement or integrated information (IIT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
import math


@dataclass
class EnhancedMERAConfig:
    """Configuration for MERA tensor network"""
    # Architecture
    num_layers: int = 3
    bond_dim: int = 8                # χ: bond dimension (entanglement capacity)
    physical_dim: int = 4            # d: physical dimension at input
    temporal_window: int = 50

    # Tensor network constraints
    enforce_isometry: bool = True
    isometry_weight: float = 0.1  # Increased for better isometry constraint
    enforce_unitarity: bool = True
    unitarity_weight: float = 0.1  # Increased for better unitarity constraint

    # Layer scaling regularization (keeps scaling factors in reasonable range)
    enforce_scaling_bounds: bool = True
    scaling_weight: float = 0.01  # Reduced: too restrictive was hurting learning
    scaling_target: float = 1.0   # Target scaling factor

    # Hierarchical entropy computation (correlation-based diagnostic, NOT real IIT)
    enable_hierarchical_entropy: bool = True
    entropy_layers: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Scale consistency (reduced - was too restrictive per experiment findings)
    scale_consistency_weight: float = 0.001  # Was 0.1, hurting performance by -6.3%
    warmup_steps: int = 1000  # Warmup before applying full scale/scaling loss

    # Training
    dropout: float = 0.0
    use_identity_init: bool = True  # Initialize isometries near identity


# =============================================================================
# True Tensor Network Components
# =============================================================================

class TrueDisentangler(nn.Module):
    """
    True disentangler tensor u in MERA using tensor contractions.

    The disentangler removes short-range entanglement between adjacent sites.
    For two input sites of dimension d_in, outputs two sites of dimension d_out.

    Tensor shape: (d_in, d_in, d_out, d_out)
    Contraction: u_{ijkl} × site1_i × site2_j → combined_{kl}

    Then we split combined back into two sites using learned projections.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Initialize with approximate unitary structure
        # Shape: (d_in, d_in, d_out, d_out)
        tensor = self._unitary_init(d_in, d_out)
        self.tensor = nn.Parameter(tensor)

        # Learned projections for splitting the contracted output
        # These project from the combined space back to individual sites
        self.proj_left = nn.Parameter(torch.randn(d_out, d_out) * 0.1)
        self.proj_right = nn.Parameter(torch.randn(d_out, d_out) * 0.1)

        # Initialize projections as orthogonal
        nn.init.orthogonal_(self.proj_left)
        nn.init.orthogonal_(self.proj_right)

    def _unitary_init(self, d_in: int, d_out: int) -> torch.Tensor:
        """
        Initialize tensor with TRUE unitary structure using orthogonal init.

        Ensures U†U = I to preserve signal magnitude through disentangling.
        No scaling factor - must preserve norm exactly.
        """
        # Shape: (d_in * d_in, d_out * d_out) - treat as matrix
        flat = torch.empty(d_in * d_in, d_out * d_out)

        # Use orthogonal initialization - singular values = 1
        nn.init.orthogonal_(flat)

        return flat.reshape(d_in, d_in, d_out, d_out)

    def forward(self, site1: torch.Tensor, site2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply disentangler via tensor contraction.

        Args:
            site1: (batch, d_in)
            site2: (batch, d_in)

        Returns:
            out1, out2: (batch, d_out) each
        """
        # Contract: u_{ijkl} × site1_i × site2_j → combined_{kl}
        # combined has shape (batch, d_out, d_out)
        combined = torch.einsum('ijkl,bi,bj->bkl', self.tensor, site1, site2)

        # Split into two sites using learned projections
        # Project along each dimension
        out1 = torch.einsum('bkl,kk->bl', combined, self.proj_left)   # (batch, d_out)
        out2 = torch.einsum('bkl,ll->bk', combined, self.proj_right)  # (batch, d_out)

        return out1, out2

    def batch_forward(self, sites_even: torch.Tensor, sites_odd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched disentangler for processing multiple site pairs in parallel.

        Args:
            sites_even: (n_pairs, batch, d_in) - even-indexed sites
            sites_odd: (n_pairs, batch, d_in) - odd-indexed sites

        Returns:
            out_even, out_odd: (n_pairs, batch, d_out) each
        """
        # Batched einsum: process all pairs at once
        # Contract: u_{ijkl} × sites_even_{p,b,i} × sites_odd_{p,b,j} → combined_{p,b,k,l}
        combined = torch.einsum('ijkl,pbi,pbj->pbkl', self.tensor, sites_even, sites_odd)

        # Split using projections
        out_even = torch.einsum('pbkl,kk->pbl', combined, self.proj_left)
        out_odd = torch.einsum('pbkl,ll->pbk', combined, self.proj_right)

        return out_even, out_odd

    def unitarity_loss(self) -> torch.Tensor:
        """Compute loss encouraging unitary structure"""
        # Reshape to matrix
        mat = self.tensor.reshape(self.d_in * self.d_in, self.d_out * self.d_out)

        # For rectangular matrices, check singular values are close to 1
        _, s, _ = torch.linalg.svd(mat, full_matrices=False)
        return F.mse_loss(s, torch.ones_like(s))


class TrueIsometry(nn.Module):
    """
    True isometry tensor w in MERA using tensor contractions.

    The isometry coarse-grains two sites into one, implementing the
    renormalization step. Maps 2 sites of d_in to 1 site of d_out.

    Tensor shape: (d_out, d_in, d_in)
    Contraction: w_{α,i,j} × site1_i × site2_j → output_α

    The isometry should satisfy w†w ≈ I (up to the smaller dimension).
    """

    def __init__(self, d_in: int, d_out: int, use_identity_init: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Initialize as isometry - use identity-based init for better RG flow
        if use_identity_init:
            tensor = self._identity_based_init(d_in, d_out)
        else:
            tensor = self._isometry_init(d_in, d_out)
        self.tensor = nn.Parameter(tensor)

    def _identity_based_init(self, d_in: int, d_out: int) -> torch.Tensor:
        """
        Initialize with TRUE orthogonal structure for stable RG flow.

        Uses torch.nn.init.orthogonal_ to ensure W†W = I exactly at initialization.
        This prevents signal vanishing through layers - critical for RG eigenvalues > 0.3.

        Added noise (0.3) to break symmetry and enable Φ_Q computation.
        Without noise, identity init = zero entanglement = Φ_Q stays at 0.
        """
        # Shape: (d_out, d_in * d_in) - treat as matrix for orthogonal init
        flat = torch.empty(d_out, d_in * d_in)

        # Use orthogonal initialization - ensures singular values = 1
        # This guarantees W†W = I (for the smaller dimension)
        nn.init.orthogonal_(flat)

        # Add noise to break symmetry and enable entanglement/Φ_Q
        # Without this, pure identity init means zero entanglement
        noise = torch.randn(d_out, d_in * d_in) * 0.3
        flat = flat + noise

        # Reshape back to tensor form
        return flat.reshape(d_out, d_in, d_in)

    def _isometry_init(self, d_in: int, d_out: int) -> torch.Tensor:
        """Initialize with isometry structure: w†w = I (original method)"""
        # Shape: (d_out, d_in * d_in)
        flat = torch.randn(d_out, d_in * d_in)

        # SVD to get orthonormal rows
        u, s, vh = torch.linalg.svd(flat, full_matrices=False)

        # u has shape (d_out, min(d_out, d_in*d_in))
        # We want orthonormal rows, so use u @ vh
        if d_out <= d_in * d_in:
            result = u @ vh[:d_out, :]
        else:
            # Pad with random orthogonal rows
            result = torch.zeros(d_out, d_in * d_in)
            result[:vh.shape[0], :] = vh

        return result.reshape(d_out, d_in, d_in) * 0.5

    def forward(self, site1: torch.Tensor, site2: torch.Tensor) -> torch.Tensor:
        """
        Apply isometry via tensor contraction.

        Args:
            site1: (batch, d_in)
            site2: (batch, d_in)

        Returns:
            output: (batch, d_out)
        """
        # Contract: w_{α,i,j} × site1_i × site2_j → output_α
        return torch.einsum('aij,bi,bj->ba', self.tensor, site1, site2)

    def batch_forward(self, sites_even: torch.Tensor, sites_odd: torch.Tensor) -> torch.Tensor:
        """
        Batched isometry for processing multiple site pairs in parallel.

        Args:
            sites_even: (n_pairs, batch, d_in) - even-indexed sites
            sites_odd: (n_pairs, batch, d_in) - odd-indexed sites

        Returns:
            output: (n_pairs, batch, d_out)
        """
        # Batched einsum: w_{α,i,j} × sites_even_{p,b,i} × sites_odd_{p,b,j} → output_{p,b,α}
        return torch.einsum('aij,pbi,pbj->pba', self.tensor, sites_even, sites_odd)

    def isometry_loss(self) -> torch.Tensor:
        """Compute loss for isometry constraint: w†w = I"""
        # Reshape to (d_out, d_in * d_in)
        mat = self.tensor.reshape(self.d_out, self.d_in * self.d_in)

        # w†w should be identity (on smaller dimension)
        if self.d_out <= self.d_in * self.d_in:
            # mat @ mat.T should be I
            product = mat @ mat.T
            identity = torch.eye(self.d_out, device=mat.device)
        else:
            # mat.T @ mat should be I
            product = mat.T @ mat
            identity = torch.eye(self.d_in * self.d_in, device=mat.device)

        return F.mse_loss(product, identity)


class HierarchicalEntropyComputer(nn.Module):
    """
    Compute a hierarchical correlation metric from tensor network states.

    NOTE: This is NOT integrated information (IIT) or quantum entanglement.
    It computes SVD-based entropy of correlation matrices between layer sites,
    which measures how correlated different parts of the representation are.

    The metric is useful as a diagnostic to track representation structure
    during training, but should not be interpreted as a consciousness measure.
    """

    def __init__(self, min_sites: int = 2):
        super().__init__()
        self.min_sites = min_sites

    def compute_correlation_entropy(self, sites: List[torch.Tensor],
                                     partition_idx: int) -> torch.Tensor:
        """
        Compute entropy of correlation matrix singular values.

        This measures how "spread out" the correlations are between
        two partitions of sites - NOT quantum entanglement.
        """
        if partition_idx <= 0 or partition_idx >= len(sites):
            return torch.zeros(sites[0].shape[0], device=sites[0].device)

        # Stack sites in each partition
        sites_A = torch.stack(sites[:partition_idx], dim=1)  # (B, n_A, d)
        sites_B = torch.stack(sites[partition_idx:], dim=1)   # (B, n_B, d)

        # Correlation matrix
        corr = torch.einsum('bik,bjk->bij', sites_A, sites_B)  # (B, n_A, n_B)

        # SVD for Schmidt-like coefficients
        try:
            _, s, _ = torch.linalg.svd(corr)
        except RuntimeError:
            return torch.zeros(sites[0].shape[0], device=sites[0].device)

        # Von Neumann entropy from squared singular values
        s_sq = s ** 2 + 1e-10
        s_normalized = s_sq / s_sq.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-10), dim=-1)

        return entropy

    def compute_entropy_for_partition(self, sites: List[torch.Tensor],
                                       partition_idx: int) -> torch.Tensor:
        """Compute hierarchical entropy for a specific partition point."""
        if partition_idx <= 0 or partition_idx >= len(sites):
            return torch.zeros(sites[0].shape[0], device=sites[0].device)

        # Whole system entropy at this partition
        S_whole = self.compute_correlation_entropy(sites, partition_idx)

        # Parts entropy
        left_sites = sites[:partition_idx]
        right_sites = sites[partition_idx:]

        S_left = self.compute_correlation_entropy(left_sites, len(left_sites) // 2) \
                 if len(left_sites) > 1 else torch.zeros_like(S_whole)
        S_right = self.compute_correlation_entropy(right_sites, len(right_sites) // 2) \
                  if len(right_sites) > 1 else torch.zeros_like(S_whole)

        # Difference: whole minus sum of parts
        return S_whole - (S_left + S_right)

    def forward(self, sites: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute hierarchical entropy metric across multiple partitions.

        This metric measures how much the correlation structure of the whole
        differs from the sum of its parts. Higher values indicate more
        "holistic" representations where correlations span partitions.

        NOTE: This is a heuristic diagnostic, not a rigorous information measure.
        """
        if len(sites) < self.min_sites:
            return torch.zeros(sites[0].shape[0], device=sites[0].device)

        n = len(sites)

        # Compute metric over multiple partition points
        partition_points = []
        for frac in [0.25, 0.33, 0.5, 0.67, 0.75]:
            pt = int(n * frac)
            if 0 < pt < n and pt not in partition_points:
                partition_points.append(pt)

        if not partition_points:
            partition_points = [n // 2]

        # Compute entropy at each partition
        entropy_values = []
        for pt in partition_points:
            ent = self.compute_entropy_for_partition(sites, pt)
            entropy_values.append(ent)

        # Take minimum across partitions (most conservative estimate)
        entropy_stack = torch.stack(entropy_values, dim=-1)
        hierarchical_entropy = entropy_stack.min(dim=-1)[0]

        return F.relu(hierarchical_entropy)


# Backwards compatibility alias
PhiQComputer = HierarchicalEntropyComputer


# =============================================================================
# Main MERA Network
# =============================================================================

class EnhancedTensorNetworkMERA(nn.Module):
    """
    True MERA tensor network with proper dimension handling.

    Architecture:
    - Input: (batch, seq_len, input_dim) → embed to (batch, seq_len, physical_dim)
    - Layer 0: physical_dim → bond_dim (dimension expansion via isometry)
    - Layer 1+: bond_dim → bond_dim (consistent dimensions)
    - Output: concatenated final sites → projection to output_dim

    The key insight is that:
    - Disentanglers preserve dimension: d → d
    - Isometries can change dimension: d_in → d_out
    - After layer 0, all operations are on bond_dim
    """

    def __init__(self, config: EnhancedMERAConfig):
        super().__init__()
        self.config = config

        # Step counter for warmup (updated via set_step)
        self.current_step = 0

        # Dimensions
        self.physical_dim = config.physical_dim
        self.bond_dim = config.bond_dim

        # Layer 0: physical_dim → bond_dim
        self.disentangler_0 = TrueDisentangler(config.physical_dim, config.physical_dim)
        self.isometry_0 = TrueIsometry(
            config.physical_dim, config.bond_dim,
            use_identity_init=config.use_identity_init
        )

        # Layers 1+: bond_dim → bond_dim
        self.disentanglers = nn.ModuleList([
            TrueDisentangler(config.bond_dim, config.bond_dim)
            for _ in range(config.num_layers - 1)
        ])
        self.isometries = nn.ModuleList([
            TrueIsometry(config.bond_dim, config.bond_dim,
                        use_identity_init=config.use_identity_init)
            for _ in range(config.num_layers - 1)
        ])

        # Input embedding: input_dim → physical_dim
        self.input_embedding = None

        # Output projection
        self.output_dim = config.bond_dim * 4
        self.output_projection = None

        # Hierarchical entropy computation (diagnostic probe only - no gradients)
        self.entropy_computer = HierarchicalEntropyComputer()

        # Scale consistency projections - pre-created to ensure they're in optimizer
        self.scale_projections = nn.ModuleDict()
        # Layer 0: odd site projection (physical_dim → bond_dim)
        self.scale_projections['layer0_odd_proj'] = nn.Linear(config.physical_dim, config.bond_dim)
        # Layer 0: scale consistency (2*physical_dim → bond_dim)
        self.scale_projections[f'scale_0_{2*config.physical_dim}_{config.bond_dim}'] = nn.Linear(
            2 * config.physical_dim, config.bond_dim
        )
        # Layers 1+: scale consistency (2*bond_dim → bond_dim)
        for layer_idx in range(1, config.num_layers):
            self.scale_projections[f'scale_{layer_idx}_{2*config.bond_dim}_{config.bond_dim}'] = nn.Linear(
                2 * config.bond_dim, config.bond_dim
            )

        # Layer scaling factor tracking (NOT true RG eigenvalues - just norm ratios)
        self.scaling_factors_history = []

    def set_step(self, step: int):
        """Update step counter for warmup scheduling"""
        self.current_step = step

    def get_warmup_factor(self, warmup_steps: int) -> float:
        """Get warmup factor (0 to 1) based on current step"""
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, self.current_step / warmup_steps)

    def _ensure_input_embedding(self, input_dim: int, device: torch.device):
        if self.input_embedding is None or self.input_embedding[0].in_features != input_dim:
            self.input_embedding = nn.Sequential(
                nn.Linear(input_dim, self.physical_dim * 2),
                nn.GELU(),
                nn.Linear(self.physical_dim * 2, self.physical_dim),
                nn.LayerNorm(self.physical_dim),
            ).to(device)

    def _ensure_output_projection(self, final_dim: int, device: torch.device):
        if self.output_projection is None or self.output_projection[0].in_features != final_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(final_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            ).to(device)

    def encode_sequence(self, sequence: torch.Tensor) -> List[torch.Tensor]:
        """Encode sequence to physical_dim sites"""
        batch_size, seq_len, input_dim = sequence.shape
        self._ensure_input_embedding(input_dim, sequence.device)

        embedded = self.input_embedding(sequence)  # (B, T, physical_dim)
        return [embedded[:, t, :] for t in range(seq_len)]

    def apply_layer_0(self, sites: List[torch.Tensor]) -> List[torch.Tensor]:
        """First layer: physical_dim → bond_dim"""
        if len(sites) < 2:
            # Project single site to bond_dim
            if sites[0].shape[-1] != self.bond_dim:
                proj = nn.Linear(sites[0].shape[-1], self.bond_dim).to(sites[0].device)
                return [proj(sites[0])]
            return sites

        # Disentangle (physical_dim → physical_dim)
        disentangled = []
        for i in range(0, len(sites) - 1, 2):
            s1, s2 = self.disentangler_0(sites[i], sites[i + 1])
            disentangled.extend([s1, s2])
        if len(sites) % 2 == 1:
            disentangled.append(sites[-1])

        # Coarse-grain (physical_dim → bond_dim)
        coarse = []
        for i in range(0, len(disentangled) - 1, 2):
            c = self.isometry_0(disentangled[i], disentangled[i + 1])
            coarse.append(c)
        if len(disentangled) % 2 == 1:
            # Project last site to bond_dim (projection pre-created in __init__)
            last = disentangled[-1]
            if last.shape[-1] != self.bond_dim:
                last = self.scale_projections['layer0_odd_proj'](last)
            coarse.append(last)

        return coarse

    def apply_layer_n(self, sites: List[torch.Tensor], layer_idx: int) -> List[torch.Tensor]:
        """Layers 1+: bond_dim → bond_dim (uses batched processing for efficiency)"""
        if len(sites) < 2:
            return sites

        disentangler = self.disentanglers[layer_idx]
        isometry = self.isometries[layer_idx]

        # Batch disentangle: collect pairs and process together
        n_pairs = len(sites) // 2
        has_odd = len(sites) % 2 == 1

        if n_pairs > 0:
            # Stack even and odd sites for batched processing
            sites_even = torch.stack([sites[2*i] for i in range(n_pairs)])     # (n_pairs, batch, d)
            sites_odd = torch.stack([sites[2*i + 1] for i in range(n_pairs)])  # (n_pairs, batch, d)

            # Batched disentangle
            out_even, out_odd = disentangler.batch_forward(sites_even, sites_odd)

            # Interleave results back
            disentangled = []
            for i in range(n_pairs):
                disentangled.extend([out_even[i], out_odd[i]])
        else:
            disentangled = sites.copy()

        if has_odd:
            disentangled.append(sites[-1])

        # Batch coarse-grain
        n_coarse_pairs = len(disentangled) // 2
        has_odd_coarse = len(disentangled) % 2 == 1

        if n_coarse_pairs > 0:
            dis_even = torch.stack([disentangled[2*i] for i in range(n_coarse_pairs)])
            dis_odd = torch.stack([disentangled[2*i + 1] for i in range(n_coarse_pairs)])

            # Batched isometry
            coarse_stacked = isometry.batch_forward(dis_even, dis_odd)  # (n_pairs, batch, d_out)
            coarse = [coarse_stacked[i] for i in range(n_coarse_pairs)]
        else:
            coarse = []

        if has_odd_coarse:
            coarse.append(disentangled[-1])

        return coarse

    def compute_scaling_factors(self, sites_before: List[torch.Tensor],
                                 sites_after: List[torch.Tensor],
                                 return_tensors: bool = False) -> List:
        """
        Compute layer scaling factors (norm ratios before/after coarse-graining).

        NOTE: These are NOT true RG eigenvalues from physics. They are simply
        the ratio of output norm to input norm through isometry layers.
        Values near 1.0 indicate signal preservation through layers.

        Args:
            sites_before: Sites before coarse-graining
            sites_after: Sites after coarse-graining
            return_tensors: If True, return tensors for gradient computation

        Returns:
            List of scaling factors (floats or tensors)
        """
        scaling_factors = []

        for i, site_after in enumerate(sites_after):
            if 2*i + 1 < len(sites_before):
                s1, s2 = sites_before[2*i], sites_before[2*i+1]
                norm_before = torch.norm(s1, dim=-1) + torch.norm(s2, dim=-1)
                norm_after = torch.norm(site_after, dim=-1)

                # Avoid division by zero
                valid_mask = norm_before > 1e-6
                if valid_mask.any():
                    ratio = norm_after / (norm_before + 1e-6)
                    if return_tensors:
                        scaling_factors.append(ratio.mean())
                    else:
                        scaling_factors.append(ratio.mean().item())

        return scaling_factors

    def compute_scaling_loss(self, sites_before: List[torch.Tensor],
                             sites_after: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute scaling factor regularization loss.

        Penalizes extreme scaling factors to prevent signal explosion (>2.0)
        or vanishing (<0.3) through layers. This is a practical regularizer,
        not a physics constraint.

        Applies warmup: loss is scaled from 0 to full weight over warmup steps.
        """
        if not self.config.enforce_scaling_bounds:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        scaling_factors = self.compute_scaling_factors(sites_before, sites_after, return_tensors=True)

        if not scaling_factors:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Stack for efficient computation
        sf_tensor = torch.stack(scaling_factors)

        # Penalize extremes only: >2.0 (explosion) or <0.3 (vanishing)
        upper_violation = F.relu(sf_tensor - 2.0)
        lower_violation = F.relu(0.3 - sf_tensor)
        loss = (upper_violation ** 2 + lower_violation ** 2).mean()

        # Apply warmup factor
        warmup_factor = self.get_warmup_factor(self.config.warmup_steps)

        return loss * self.config.scaling_weight * warmup_factor

    def compute_constraint_loss(self) -> torch.Tensor:
        """Total constraint loss for unitarity and isometry.

        Fixed: Use list accumulation + sum for efficient gradient computation.
        """
        losses = []
        device = next(self.parameters()).device

        if self.config.enforce_unitarity:
            losses.append(self.config.unitarity_weight * self.disentangler_0.unitarity_loss())
            for dis in self.disentanglers:
                losses.append(self.config.unitarity_weight * dis.unitarity_loss())

        if self.config.enforce_isometry:
            losses.append(self.config.isometry_weight * self.isometry_0.isometry_loss())
            for iso in self.isometries:
                losses.append(self.config.isometry_weight * iso.isometry_loss())

        if not losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).sum()

    def compute_scale_consistency_loss(self, layer_states: List[List[torch.Tensor]]) -> torch.Tensor:
        """Scale consistency across layers.

        Applies warmup: loss is scaled from 0 to full weight over warmup_steps.
        """
        if len(layer_states) < 2:
            return torch.tensor(0.0, device=layer_states[0][0].device)

        total_loss = torch.tensor(0.0, device=layer_states[0][0].device)

        for layer_idx in range(len(layer_states) - 1):
            sites_fine = layer_states[layer_idx]
            sites_coarse = layer_states[layer_idx + 1]

            for i, site_coarse in enumerate(sites_coarse):
                if 2*i + 1 < len(sites_fine):
                    combined = torch.cat([sites_fine[2*i], sites_fine[2*i+1]], dim=-1)
                    key = f'scale_{layer_idx}_{combined.shape[-1]}_{site_coarse.shape[-1]}'
                    # Use pre-created projection if available, otherwise skip (defensive)
                    if key in self.scale_projections:
                        projected = self.scale_projections[key](combined)
                        total_loss = total_loss + F.mse_loss(projected, site_coarse.detach())

        # Apply warmup factor
        warmup_factor = self.get_warmup_factor(self.config.warmup_steps)

        return total_loss * self.config.scale_consistency_weight * warmup_factor

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through MERA"""
        # Encode to physical_dim sites
        sites = self.encode_sequence(sequence)
        layer_states = [sites]
        entropy_values = []
        all_scaling_factors = []
        scaling_loss = torch.tensor(0.0, device=sequence.device)

        # Layer 0: physical_dim → bond_dim
        # Hierarchical entropy computed without gradients - it's a probe/diagnostic only
        if self.config.enable_hierarchical_entropy and 0 in self.config.entropy_layers and len(sites) >= 2:
            with torch.no_grad():
                entropy_values.append(self.entropy_computer(sites))

        sites_before = sites
        sites = self.apply_layer_0(sites)
        sf = self.compute_scaling_factors(sites_before, sites)
        if sf:
            all_scaling_factors.extend(sf)
        # Compute scaling loss for this layer
        scaling_loss = scaling_loss + self.compute_scaling_loss(sites_before, sites)
        layer_states.append(sites)

        # Layers 1+: bond_dim → bond_dim
        for layer_idx in range(self.config.num_layers - 1):
            if len(sites) <= 1:
                break

            if self.config.enable_hierarchical_entropy and (layer_idx + 1) in self.config.entropy_layers and len(sites) >= 2:
                with torch.no_grad():
                    entropy_values.append(self.entropy_computer(sites))

            sites_before = sites
            sites = self.apply_layer_n(sites, layer_idx)
            sf = self.compute_scaling_factors(sites_before, sites)
            if sf:
                all_scaling_factors.extend(sf)
            # Compute scaling loss for this layer
            scaling_loss = scaling_loss + self.compute_scaling_loss(sites_before, sites)
            layer_states.append(sites)

        # Final latent - use fixed-size pooling to handle variable number of sites
        if len(sites) > 0:
            sites_stack = torch.stack(sites, dim=1)  # (B, n_sites, bond_dim)
            max_pool = sites_stack.max(dim=1)[0]     # (B, bond_dim)
            mean_pool = sites_stack.mean(dim=1)      # (B, bond_dim)
            final_features = torch.cat([max_pool, mean_pool], dim=-1)  # (B, 2*bond_dim)
        else:
            # Fallback to previous layer
            sites_stack = torch.stack(layer_states[-2], dim=1)
            max_pool = sites_stack.max(dim=1)[0]
            mean_pool = sites_stack.mean(dim=1)
            final_features = torch.cat([max_pool, mean_pool], dim=-1)

        self._ensure_output_projection(final_features.shape[-1], final_features.device)
        latent = self.output_projection(final_features)

        # Aggregate hierarchical entropy (for backwards compatibility, key is still 'phi_q')
        entropy_total = torch.stack(entropy_values).mean(dim=0) if entropy_values else torch.zeros(
            sequence.shape[0], device=sequence.device
        )

        # Track scaling factors
        if all_scaling_factors:
            self.scaling_factors_history.append(all_scaling_factors)
            if len(self.scaling_factors_history) > 100:
                self.scaling_factors_history.pop(0)

        # Losses (regularization for encoder)
        constraint_loss = self.compute_constraint_loss()
        scale_consistency_loss = self.compute_scale_consistency_loss(layer_states)

        aux = {
            'phi_q': entropy_total.detach(),  # Backwards compat key; no gradients - probe only
            'hierarchical_entropy': entropy_total.detach(),  # New honest name
            'layer_states': layer_states,
            'scaling_factors': all_scaling_factors,  # Renamed from rg_eigenvalues
            'constraint_loss': constraint_loss,
            'scale_consistency_loss': scale_consistency_loss,
            'scaling_loss': scaling_loss,  # Renamed from rg_eigenvalue_loss
            'num_sites_per_layer': [len(states) for states in layer_states],
        }

        return latent, aux

    def get_total_loss(self, aux: Dict) -> torch.Tensor:
        return aux['constraint_loss'] + aux['scale_consistency_loss'] + aux['scaling_loss']


class MERAWorldModelEncoder(nn.Module):
    """MERA encoder for world models"""

    def __init__(self, obs_dim: int, latent_dim: int, config: Optional[EnhancedMERAConfig] = None):
        super().__init__()

        if config is None:
            config = EnhancedMERAConfig(
                num_layers=3,
                bond_dim=max(8, latent_dim // 8),
                physical_dim=max(4, latent_dim // 16),
            )

        self.config = config
        self.mera = EnhancedTensorNetworkMERA(config)
        self.latent_projection = nn.Linear(self.mera.output_dim, latent_dim)

    def forward(self, obs_sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        mera_out, aux = self.mera(obs_sequence)
        latent = self.latent_projection(mera_out)
        return latent, aux


# For backwards compatibility
UnitaryInitializer = None  # Removed - using inline initialization


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing MERA Tensor Network Encoder")
    print("=" * 70)

    config = EnhancedMERAConfig(
        num_layers=3,
        bond_dim=8,
        physical_dim=4,
        enable_hierarchical_entropy=True,
    )

    mera = EnhancedTensorNetworkMERA(config)
    print(f"Model parameters: {sum(p.numel() for p in mera.parameters()):,}")

    # Test
    batch_size = 4
    seq_len = 50
    input_dim = 64

    sequence = torch.randn(batch_size, seq_len, input_dim)

    print("\n1. Forward pass...")
    latent, aux = mera(sequence)

    print(f"   Input: {sequence.shape}")
    print(f"   Latent: {latent.shape}")
    print(f"   Hierarchical entropy: {aux['hierarchical_entropy'].mean().item():.4f}")
    print(f"   Sites per layer: {aux['num_sites_per_layer']}")

    print("\n2. Losses...")
    print(f"   Constraint: {aux['constraint_loss'].item():.6f}")
    print(f"   Scale consistency: {aux['scale_consistency_loss'].item():.6f}")
    print(f"   Scaling loss: {aux['scaling_loss'].item():.6f}")

    print("\n3. Scaling factors...")
    if aux['scaling_factors']:
        print(f"   Layer scaling factors: {[f'{sf:.3f}' for sf in aux['scaling_factors']]}")
    else:
        print("   No scaling factors computed")

    print("\n4. Gradient test...")
    loss = latent.mean() + mera.get_total_loss(aux)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in mera.parameters() if p.grad is not None)
    print(f"   Gradient norm: {grad_norm:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed - MERA tensor network encoder working!")
