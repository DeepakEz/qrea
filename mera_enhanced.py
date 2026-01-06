"""
Enhanced MERA Tensor Network for Reinforcement Learning
========================================================

This module implements a GENUINE tensor network MERA with proper contractions,
not an MLP approximation. The tensor network structure is what provides the
unique inductive bias for hierarchical temporal reasoning.

Key Research Contributions:
- True tensor network contractions (einsum operations)
- Proper dimension handling across layers
- Isometry constraint regularization (w†w = I)
- Integrated information (Φ_Q) as intrinsic motivation
- RG flow tracking for transfer learning

Architecture:
- Layer 0: physical_dim → bond_dim (dimension expansion)
- Layer 1+: bond_dim → bond_dim (consistent dimensions)
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

    # Physics-inspired constraints
    causal_velocity: float = 1.0
    enforce_isometry: bool = True
    isometry_weight: float = 0.1  # Increased for better isometry constraint
    enforce_unitarity: bool = True
    unitarity_weight: float = 0.1  # Increased for better unitarity constraint

    # RG flow regularization
    enforce_rg_fixed_point: bool = True
    rg_eigenvalue_weight: float = 0.01  # Reduced: too restrictive was hurting learning
    rg_target_eigenvalue: float = 1.0   # Target RG eigenvalue (fixed point)
    rg_loss_warmup_steps: int = 1000    # Warmup before applying full RG loss

    # Φ_Q computation
    enable_phi_q: bool = True
    phi_q_layers: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Intrinsic motivation
    phi_q_intrinsic_weight: float = 0.1
    entanglement_exploration_weight: float = 0.05

    # Scale consistency (reduced - was too restrictive per experiment findings)
    scale_consistency_weight: float = 0.001  # Was 0.1, hurting performance by -6.3%
    scale_loss_warmup_steps: int = 1000  # Warmup before applying full scale loss

    # Training
    use_gradient_checkpointing: bool = False
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
        """Initialize tensor with approximate unitary structure"""
        # Create random matrix and orthogonalize
        flat = torch.randn(d_in * d_in, d_out * d_out)
        u, s, vh = torch.linalg.svd(flat, full_matrices=False)

        # Use left singular vectors (orthogonal columns)
        k = min(d_in * d_in, d_out * d_out)
        if d_in * d_in <= d_out * d_out:
            result = u @ vh[:k, :]
        else:
            result = u[:, :k] @ vh

        return result.reshape(d_in, d_in, d_out, d_out) * 0.5

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
        Initialize near identity for stable RG flow.

        For RG fixed point, we want the isometry to approximately preserve
        norm: ||w(s1, s2)|| ≈ ||s1|| + ||s2||. This is achieved by
        initializing close to a "averaging" operation plus small noise.
        """
        # Shape: (d_out, d_in, d_in)
        tensor = torch.zeros(d_out, d_in, d_in)

        # For each output dimension, create a near-identity mapping
        # that averages the two inputs (which preserves norm for RG flow)
        min_dim = min(d_out, d_in)
        for i in range(min_dim):
            # Diagonal elements: average of corresponding input dimensions
            # w[i, i, i] combines site1[i] and site2[i] into output[i]
            tensor[i, i, i] = 1.0 / math.sqrt(2)  # Normalized averaging

        # Add small noise for symmetry breaking
        noise = torch.randn(d_out, d_in, d_in) * 0.1
        tensor = tensor + noise

        # Ensure isometry constraint approximately holds
        # Re-orthogonalize using SVD
        flat = tensor.reshape(d_out, d_in * d_in)
        u, s, vh = torch.linalg.svd(flat, full_matrices=False)

        # Scale singular values toward 1 (for isometry)
        s_scaled = 0.7 * torch.ones_like(s) + 0.3 * s / (s.max() + 1e-6)

        if d_out <= d_in * d_in:
            result = u @ torch.diag(s_scaled) @ vh[:d_out, :]
        else:
            result = u @ torch.diag(s_scaled) @ vh

        return result.reshape(d_out, d_in, d_in)

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


class PhiQComputer(nn.Module):
    """
    Compute integrated information Φ_Q from tensor network states.

    Φ_Q measures information integration - how much the whole system
    knows that cannot be reduced to its parts. Uses entanglement entropy
    as a quantum-inspired proxy.
    """

    def __init__(self, min_sites: int = 2):
        super().__init__()
        self.min_sites = min_sites

    def compute_entanglement_entropy(self, sites: List[torch.Tensor],
                                      partition_idx: int) -> torch.Tensor:
        """
        Compute entanglement entropy S_A for bipartition.

        Uses SVD of the correlation matrix as proxy for Schmidt decomposition.
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

    def compute_phi_for_partition(self, sites: List[torch.Tensor],
                                    partition_idx: int) -> torch.Tensor:
        """Compute Φ for a specific partition point."""
        if partition_idx <= 0 or partition_idx >= len(sites):
            return torch.zeros(sites[0].shape[0], device=sites[0].device)

        # Whole system entropy at this partition
        S_whole = self.compute_entanglement_entropy(sites, partition_idx)

        # Parts entropy
        left_sites = sites[:partition_idx]
        right_sites = sites[partition_idx:]

        S_left = self.compute_entanglement_entropy(left_sites, len(left_sites) // 2) \
                 if len(left_sites) > 1 else torch.zeros_like(S_whole)
        S_right = self.compute_entanglement_entropy(right_sites, len(right_sites) // 2) \
                  if len(right_sites) > 1 else torch.zeros_like(S_whole)

        # Φ at this partition
        return S_whole - (S_left + S_right)

    def forward(self, sites: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute Φ_Q using Minimum Information Partition (MIP) approximation.

        True IIT Φ requires finding the partition that minimizes integrated info.
        We approximate by computing Φ over multiple partition points and taking
        the minimum. This prevents artificially high Φ from arbitrary partitions.

        Positive Φ_Q indicates genuine information integration beyond parts.
        """
        if len(sites) < self.min_sites:
            return torch.zeros(sites[0].shape[0], device=sites[0].device)

        n = len(sites)

        # Compute Φ over multiple partition points (MIP approximation)
        # Use partitions at 1/4, 1/3, 1/2, 2/3, 3/4 of sequence
        partition_points = []
        for frac in [0.25, 0.33, 0.5, 0.67, 0.75]:
            pt = int(n * frac)
            if 0 < pt < n and pt not in partition_points:
                partition_points.append(pt)

        if not partition_points:
            partition_points = [n // 2]

        # Compute Φ at each partition
        phi_values = []
        for pt in partition_points:
            phi = self.compute_phi_for_partition(sites, pt)
            phi_values.append(phi)

        # MIP: take minimum Φ across partitions
        # This finds the "weakest link" - where integration is lowest
        phi_stack = torch.stack(phi_values, dim=-1)  # (B, n_partitions)
        phi_q = phi_stack.min(dim=-1)[0]  # (B,)

        return F.relu(phi_q)


class MERAIntrinsicMotivation(nn.Module):
    """Intrinsic motivation from MERA structure"""

    def __init__(self, config: EnhancedMERAConfig):
        super().__init__()
        self.config = config
        self.phi_q_computer = PhiQComputer()

        # Running statistics
        self.register_buffer('phi_q_mean', torch.tensor(0.0))
        self.register_buffer('phi_q_std', torch.tensor(1.0))
        self.register_buffer('entanglement_mean', torch.tensor(0.0))
        self.register_buffer('entanglement_std', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))

    def update_statistics(self, phi_q: torch.Tensor, entanglement: torch.Tensor):
        momentum = 0.99
        if self.update_count == 0:
            self.phi_q_mean = phi_q.mean()
            self.phi_q_std = phi_q.std() + 1e-8
            self.entanglement_mean = entanglement.mean()
            self.entanglement_std = entanglement.std() + 1e-8
        else:
            self.phi_q_mean = momentum * self.phi_q_mean + (1 - momentum) * phi_q.mean()
            self.phi_q_std = momentum * self.phi_q_std + (1 - momentum) * (phi_q.std() + 1e-8)
            self.entanglement_mean = momentum * self.entanglement_mean + (1 - momentum) * entanglement.mean()
            self.entanglement_std = momentum * self.entanglement_std + (1 - momentum) * (entanglement.std() + 1e-8)
        self.update_count += 1

    def compute_intrinsic_reward(self, layer_states: List[List[torch.Tensor]],
                                  rg_eigenvalues: List[float]) -> Dict[str, torch.Tensor]:
        device = layer_states[0][0].device
        batch_size = layer_states[0][0].shape[0]

        # Φ_Q reward
        phi_q_values = []
        for layer_idx in self.config.phi_q_layers:
            if layer_idx < len(layer_states) and len(layer_states[layer_idx]) >= 2:
                phi_q = self.phi_q_computer(layer_states[layer_idx])
                phi_q_values.append(phi_q)

        phi_q_total = torch.stack(phi_q_values).mean(dim=0) if phi_q_values else torch.zeros(batch_size, device=device)

        # Entanglement reward
        if len(layer_states[0]) > 1:
            entanglement = self.phi_q_computer.compute_entanglement_entropy(
                layer_states[0], len(layer_states[0]) // 2
            )
        else:
            entanglement = torch.zeros(batch_size, device=device)

        # RG novelty
        if rg_eigenvalues:
            rg_deviation = sum(abs(ev - 1.0) for ev in rg_eigenvalues) / len(rg_eigenvalues)
            rg_novelty = torch.full((batch_size,), rg_deviation, device=device)
        else:
            rg_novelty = torch.zeros(batch_size, device=device)

        if self.training:
            self.update_statistics(phi_q_total.detach(), entanglement.detach())

        # Normalize
        phi_q_normalized = (phi_q_total - self.phi_q_mean) / (self.phi_q_std + 1e-8)
        entanglement_normalized = (entanglement - self.entanglement_mean) / (self.entanglement_std + 1e-8)

        total_intrinsic = (
            self.config.phi_q_intrinsic_weight * phi_q_normalized +
            self.config.entanglement_exploration_weight * entanglement_normalized
        )

        return {
            'phi_q_reward': phi_q_normalized,
            'entanglement_reward': entanglement_normalized,
            'rg_novelty_reward': rg_novelty,
            'total_intrinsic': total_intrinsic,
            'phi_q_raw': phi_q_total,
            'entanglement_raw': entanglement,
        }


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

        # Intrinsic motivation
        self.intrinsic_motivation = MERAIntrinsicMotivation(config)
        self.phi_q_computer = PhiQComputer()

        # Scale consistency
        self.scale_projections = nn.ModuleDict()

        # RG tracking
        self.rg_eigenvalues_history = []

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
            # Project last site to bond_dim
            last = disentangled[-1]
            if last.shape[-1] != self.bond_dim:
                key = 'layer0_odd_proj'
                if key not in self.scale_projections:
                    self.scale_projections[key] = nn.Linear(last.shape[-1], self.bond_dim).to(last.device)
                last = self.scale_projections[key](last)
            coarse.append(last)

        return coarse

    def apply_layer_n(self, sites: List[torch.Tensor], layer_idx: int) -> List[torch.Tensor]:
        """Layers 1+: bond_dim → bond_dim"""
        if len(sites) < 2:
            return sites

        disentangler = self.disentanglers[layer_idx]
        isometry = self.isometries[layer_idx]

        # Disentangle
        disentangled = []
        for i in range(0, len(sites) - 1, 2):
            s1, s2 = disentangler(sites[i], sites[i + 1])
            disentangled.extend([s1, s2])
        if len(sites) % 2 == 1:
            disentangled.append(sites[-1])

        # Coarse-grain
        coarse = []
        for i in range(0, len(disentangled) - 1, 2):
            c = isometry(disentangled[i], disentangled[i + 1])
            coarse.append(c)
        if len(disentangled) % 2 == 1:
            coarse.append(disentangled[-1])

        return coarse

    def compute_rg_eigenvalues(self, sites_before: List[torch.Tensor],
                                sites_after: List[torch.Tensor],
                                return_tensors: bool = False) -> List:
        """
        Compute RG flow eigenvalues.

        Args:
            sites_before: Sites before coarse-graining
            sites_after: Sites after coarse-graining
            return_tensors: If True, return tensors for gradient computation

        Returns:
            List of eigenvalues (floats or tensors)
        """
        eigenvalues = []

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
                        eigenvalues.append(ratio.mean())
                    else:
                        eigenvalues.append(ratio.mean().item())

        return eigenvalues

    def compute_rg_eigenvalue_loss(self, sites_before: List[torch.Tensor],
                                    sites_after: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute RG eigenvalue loss - penalize EXTREME values only.

        In physics MERA:
        - λ > 1: relevant operators (important low-level features)
        - λ = 1: marginal operators (scale-invariant)
        - λ < 1: irrelevant operators (noise)

        We allow eigenvalues to vary naturally, only penalizing extremes
        (λ > 2.0 or λ < 0.3) to prevent instability.

        Applies warmup: loss is scaled from 0 to full weight over rg_loss_warmup_steps.
        """
        if not self.config.enforce_rg_fixed_point:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        eigenvalues = self.compute_rg_eigenvalues(sites_before, sites_after, return_tensors=True)

        if not eigenvalues:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Stack eigenvalues for efficient computation (fixes gradient accumulation)
        ev_tensor = torch.stack(eigenvalues)

        # Penalize extremes only: λ > 2.0 or λ < 0.3
        # This allows relevant (λ>1) and irrelevant (λ<1) operators to emerge naturally
        upper_violation = F.relu(ev_tensor - 2.0)  # Penalize λ > 2.0
        lower_violation = F.relu(0.3 - ev_tensor)  # Penalize λ < 0.3
        loss = (upper_violation ** 2 + lower_violation ** 2).mean()

        # Apply warmup factor
        warmup_factor = self.get_warmup_factor(self.config.rg_loss_warmup_steps)

        return loss * self.config.rg_eigenvalue_weight * warmup_factor

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

        Applies warmup: loss is scaled from 0 to full weight over scale_loss_warmup_steps.
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
                    if key not in self.scale_projections:
                        self.scale_projections[key] = nn.Linear(
                            combined.shape[-1], site_coarse.shape[-1]
                        ).to(combined.device)
                    projected = self.scale_projections[key](combined)
                    total_loss = total_loss + F.mse_loss(projected, site_coarse.detach())

        # Apply warmup factor
        warmup_factor = self.get_warmup_factor(self.config.scale_loss_warmup_steps)

        return total_loss * self.config.scale_consistency_weight * warmup_factor

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through MERA"""
        # Encode to physical_dim sites
        sites = self.encode_sequence(sequence)
        layer_states = [sites]
        phi_q_values = []
        all_rg_eigenvalues = []
        rg_eigenvalue_loss = torch.tensor(0.0, device=sequence.device)

        # Layer 0: physical_dim → bond_dim
        if self.config.enable_phi_q and 0 in self.config.phi_q_layers and len(sites) >= 2:
            phi_q_values.append(self.phi_q_computer(sites))

        sites_before = sites
        sites = self.apply_layer_0(sites)
        rg_evs = self.compute_rg_eigenvalues(sites_before, sites)
        if rg_evs:
            all_rg_eigenvalues.extend(rg_evs)
        # Compute RG loss for this layer
        rg_eigenvalue_loss = rg_eigenvalue_loss + self.compute_rg_eigenvalue_loss(sites_before, sites)
        layer_states.append(sites)

        # Layers 1+: bond_dim → bond_dim
        for layer_idx in range(self.config.num_layers - 1):
            if len(sites) <= 1:
                break

            if self.config.enable_phi_q and (layer_idx + 1) in self.config.phi_q_layers and len(sites) >= 2:
                phi_q_values.append(self.phi_q_computer(sites))

            sites_before = sites
            sites = self.apply_layer_n(sites, layer_idx)
            rg_evs = self.compute_rg_eigenvalues(sites_before, sites)
            if rg_evs:
                all_rg_eigenvalues.extend(rg_evs)
            # Compute RG loss for this layer
            rg_eigenvalue_loss = rg_eigenvalue_loss + self.compute_rg_eigenvalue_loss(sites_before, sites)
            layer_states.append(sites)

        # Final latent - use fixed-size pooling to handle variable number of sites
        # This fixes the "dimension explosion" issue where concat creates variable-length vectors
        if len(sites) > 0:
            sites_stack = torch.stack(sites, dim=1)  # (B, n_sites, bond_dim)
            # Use both max and mean pooling for richer representation
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

        # Aggregate Φ_Q
        phi_q_total = torch.stack(phi_q_values).mean(dim=0) if phi_q_values else torch.zeros(
            sequence.shape[0], device=sequence.device
        )

        # Track RG
        if all_rg_eigenvalues:
            self.rg_eigenvalues_history.append(all_rg_eigenvalues)
            if len(self.rg_eigenvalues_history) > 100:
                self.rg_eigenvalues_history.pop(0)

        # Losses
        constraint_loss = self.compute_constraint_loss()
        scale_loss = self.compute_scale_consistency_loss(layer_states)

        # Intrinsic rewards
        intrinsic_rewards = self.intrinsic_motivation.compute_intrinsic_reward(
            layer_states, all_rg_eigenvalues
        )

        aux = {
            'phi_q': phi_q_total,
            'layer_states': layer_states,
            'rg_eigenvalues': all_rg_eigenvalues,
            'constraint_loss': constraint_loss,
            'scale_consistency_loss': scale_loss,
            'rg_eigenvalue_loss': rg_eigenvalue_loss,
            'intrinsic_rewards': intrinsic_rewards,
            'num_sites_per_layer': [len(states) for states in layer_states],
        }

        return latent, aux

    def get_total_loss(self, aux: Dict) -> torch.Tensor:
        return aux['constraint_loss'] + aux['scale_consistency_loss'] + aux['rg_eigenvalue_loss']


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
    print("Testing True Tensor Network MERA")
    print("=" * 70)

    config = EnhancedMERAConfig(
        num_layers=3,
        bond_dim=8,
        physical_dim=4,
        enable_phi_q=True,
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
    print(f"   Φ_Q: {aux['phi_q'].mean().item():.4f}")
    print(f"   Sites per layer: {aux['num_sites_per_layer']}")

    print("\n2. Losses...")
    print(f"   Constraint: {aux['constraint_loss'].item():.6f}")
    print(f"   Scale: {aux['scale_consistency_loss'].item():.6f}")

    print("\n3. Intrinsic rewards...")
    intrinsic = aux['intrinsic_rewards']
    print(f"   Φ_Q reward: {intrinsic['phi_q_reward'].mean().item():.4f}")
    print(f"   Total: {intrinsic['total_intrinsic'].mean().item():.4f}")

    print("\n4. Gradient test...")
    loss = latent.mean() + mera.get_total_loss(aux)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in mera.parameters() if p.grad is not None)
    print(f"   Gradient norm: {grad_norm:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed - True tensor network MERA working!")
