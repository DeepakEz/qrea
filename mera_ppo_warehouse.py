"""
MERA-PPO Warehouse Training (Paper Version)
=============================================

Vanilla PPO + MERA encoder for ICLR submission.
MERA provides hierarchical temporal encoding; Φ_Q is measured as a probe ONLY.

Key Research Questions:
1. Does MERA's hierarchical structure improve sample efficiency vs baselines?
2. Does higher Φ_Q correlate with better coordination (emergent property)?
3. Can MERA representations transfer across robot counts?

Encoders:
- mera: MERA tensor network (hierarchical, physics-inspired)
- mera_uprt: MERA + UPRT spatial fields
- gru: GRU baseline
- transformer: Transformer baseline
- mlp: MLP baseline (no temporal structure)

Usage:
    python mera_ppo_warehouse.py --epochs 100 --encoder mera
    python mera_ppo_warehouse.py --encoder mlp  # Baseline comparison
    python mera_ppo_warehouse.py --robots 16    # Scaling study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import yaml
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict, field
from collections import deque

# Import existing infrastructure
from warehouse_env import WarehouseEnv
from warehouse_uprt import UPRTField
from mera_enhanced import EnhancedMERAConfig, EnhancedTensorNetworkMERA, PhiQComputer
# Note: mera_rl_integration contains MERAEnhancedWorldModel (unused in paper version)

# Paper-core: vanilla PPO only - no GRFE, Evolution, or Safety modules


# =============================================================================
# Data Structures
# =============================================================================

class Transition(NamedTuple):
    """Single transition for PPO"""
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    value: float
    log_prob: float
    phi_q: float
    robot_position: Optional[np.ndarray] = None  # For mera_uprt encoder


class TrajectoryRecorder:
    """Records complete world state trajectories for holographic training.

    Used by:
    - Cosmic Egg: World model pre-training (s, a, s' tuples)
    - Genome Distillation: Extract agent behavior patterns
    - Zero-latency: State prediction validation

    Records global_state snapshots at each step when enabled.
    """

    def __init__(self, capacity: int = 10000, record_every: int = 1):
        self.capacity = capacity
        self.record_every = record_every
        self.buffer: deque = deque(maxlen=capacity)
        self.step_counter = 0
        self.enabled = False

    def enable(self, record_every: int = 1):
        """Enable recording with specified frequency"""
        self.enabled = True
        self.record_every = record_every

    def disable(self):
        """Disable recording"""
        self.enabled = False

    def record(self, global_state: dict, actions: dict, rewards: dict,
               phi_q: float, done: bool):
        """Record a single timestep.

        Args:
            global_state: From env.get_global_state_snapshot()
            actions: Dict of robot_id -> action array
            rewards: Dict of robot_id -> reward
            phi_q: Current Φ_Q value
            done: Episode termination flag
        """
        if not self.enabled:
            return

        self.step_counter += 1
        if self.step_counter % self.record_every != 0:
            return

        # Convert actions/rewards to arrays for efficient storage
        num_robots = len(actions)
        actions_arr = np.array([actions[i] for i in range(num_robots)], dtype=np.float32)
        rewards_arr = np.array([rewards[i] for i in range(num_robots)], dtype=np.float32)

        record = {
            'state': global_state,
            'actions': actions_arr,
            'rewards': rewards_arr,
            'phi_q': phi_q,
            'done': done,
            'step': self.step_counter,
        }
        self.buffer.append(record)

    def get_trajectories(self, num_samples: Optional[int] = None) -> List[dict]:
        """Get recorded trajectories.

        Args:
            num_samples: If specified, return random sample of this size

        Returns:
            List of trajectory records
        """
        if num_samples is None or num_samples >= len(self.buffer):
            return list(self.buffer)

        indices = np.random.choice(len(self.buffer), num_samples, replace=False)
        return [self.buffer[i] for i in indices]

    def get_state_action_pairs(self) -> Tuple[List[dict], np.ndarray]:
        """Get (state, action) pairs for world model training.

        Returns:
            states: List of global state dicts
            actions: (N, num_robots, action_dim) array
        """
        states = [r['state'] for r in self.buffer]
        actions = np.stack([r['actions'] for r in self.buffer])
        return states, actions

    def save(self, path: str):
        """Save trajectories to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        """Load trajectories from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.capacity)

    def clear(self):
        """Clear recorded trajectories"""
        self.buffer.clear()
        self.step_counter = 0

    def __len__(self):
        return len(self.buffer)


@dataclass
class CoordinationMetrics:
    """Metrics for multi-agent coordination quality.

    FIX: All metrics are now normalized by num_robots for fair scaling studies.
    """
    total_collisions: int = 0
    packages_delivered: int = 0
    packages_picked_up: int = 0
    total_distance: float = 0.0
    total_energy: float = 0.0
    episode_length: int = 0
    throughput: float = 0.0
    avg_waiting_time: float = 0.0
    num_robots: int = 1  # Added for normalization

    @property
    def collision_rate(self) -> float:
        """Collisions per step (not normalized by robots - collisions are O(N²))"""
        return self.total_collisions / max(self.episode_length, 1)

    @property
    def collision_rate_per_robot(self) -> float:
        """Collisions per robot per step - normalized for scaling studies"""
        # Collisions are pairwise events, scale as O(N²) potential pairs
        # Normalize by N*(N-1)/2 potential collision pairs
        potential_pairs = max(self.num_robots * (self.num_robots - 1) / 2, 1)
        return self.total_collisions / (potential_pairs * max(self.episode_length, 1))

    @property
    def efficiency(self) -> float:
        return self.packages_delivered / max(self.total_energy, 0.001)

    @property
    def efficiency_per_robot(self) -> float:
        """Per-robot efficiency - normalized for scaling studies"""
        return self.efficiency / max(self.num_robots, 1)

    @property
    def throughput_per_robot(self) -> float:
        """Packages delivered per robot per hour"""
        return self.throughput / max(self.num_robots, 1)

    @property
    def synergy_score(self) -> float:
        """Higher when robots work together well.

        FIX: Now uses normalized collision rate so score is comparable across robot counts.
        """
        # Use per-robot-pair collision rate for fair scaling
        collision_penalty = 1.0 / (1.0 + self.collision_rate_per_robot * 100)
        # Use per-robot throughput
        throughput_bonus = min(self.throughput_per_robot / 10.0, 1.0)
        return (collision_penalty + throughput_bonus) / 2.0


# =============================================================================
# Neural Network Components
# =============================================================================

def compute_von_neumann_entropy(psi: torch.Tensor, partition_idx: int) -> torch.Tensor:
    """
    Compute Von Neumann entanglement entropy for a bipartition.

    This is the REAL physics metric used in tensor network literature.

    Algorithm:
    1. Reshape state as matrix ψ_{a,b}
    2. Compute reduced density matrix ρ_A = ψ @ ψ†
    3. Compute S(ρ_A) = -Tr(ρ_A log ρ_A)

    References:
    - Vidal et al., Phys. Rev. Lett. 90, 227902 (2003)
    - Calabrese & Cardy, J. Stat. Mech. P06002 (2004)
    """
    batch_size = psi.shape[0]
    total_dim = psi.shape[1]

    d_A = partition_idx
    d_B = total_dim // d_A if d_A > 0 else total_dim

    # Ensure dimensions work
    if d_A * d_B != total_dim:
        new_total = d_A * d_B
        if new_total > total_dim:
            psi = F.pad(psi, (0, new_total - total_dim))
        else:
            psi = psi[:, :new_total]

    psi_matrix = psi.reshape(batch_size, d_A, d_B)
    rho_A = torch.bmm(psi_matrix, psi_matrix.transpose(1, 2))

    # Normalize
    trace = torch.diagonal(rho_A, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    rho_A = rho_A / (trace + 1e-10)

    try:
        eigenvalues = torch.linalg.eigvalsh(rho_A)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        S_vN = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)
    except RuntimeError:
        S_vN = torch.zeros(batch_size, device=psi.device)

    return S_vN


def compute_geometric_phi(X: torch.Tensor, partition_idx: int) -> torch.Tensor:
    """
    Compute Geometric Integrated Information (Φ_G).

    Based on Barrett & Seth (2011) "Practical Measures of Integrated Information".

    Φ_G = 0.5 * log(det(Σ) / (det(Σ_A) * det(Σ_B)))

    This is the Gaussian approximation to integrated information.

    References:
    - Barrett & Seth, PLoS Comput Biol 7(1): e1001052 (2011)
    - Oizumi et al., PLoS Comput Biol 12(3): e1004654 (2016)
    """
    batch_size, dim = X.shape
    if partition_idx <= 0 or partition_idx >= dim:
        return torch.zeros(batch_size, device=X.device)

    X_A = X[:, :partition_idx]
    X_B = X[:, partition_idx:]

    X_centered = X - X.mean(dim=0, keepdim=True)
    X_A_centered = X_A - X_A.mean(dim=0, keepdim=True)
    X_B_centered = X_B - X_B.mean(dim=0, keepdim=True)

    reg = 1e-6 * torch.eye(dim, device=X.device)
    reg_A = 1e-6 * torch.eye(partition_idx, device=X.device)
    reg_B = 1e-6 * torch.eye(dim - partition_idx, device=X.device)

    Sigma = X_centered.T @ X_centered / batch_size + reg
    Sigma_A = X_A_centered.T @ X_A_centered / batch_size + reg_A
    Sigma_B = X_B_centered.T @ X_B_centered / batch_size + reg_B

    try:
        log_det_Sigma = torch.linalg.slogdet(Sigma)[1]
        log_det_Sigma_A = torch.linalg.slogdet(Sigma_A)[1]
        log_det_Sigma_B = torch.linalg.slogdet(Sigma_B)[1]

        phi_g = 0.5 * (log_det_Sigma_A + log_det_Sigma_B - log_det_Sigma)
        phi_g = torch.clamp(phi_g, min=0.0)
        return phi_g.expand(batch_size)
    except RuntimeError:
        return torch.zeros(batch_size, device=X.device)


def compute_representation_coherence(representations: torch.Tensor) -> torch.Tensor:
    """
    Compute RESEARCH-GRADE integration metric across all encoder types.

    Combines two rigorous measures:
    1. Von Neumann Entanglement Entropy (S_vN) - physics metric
    2. Geometric Integrated Information (Φ_G) - IIT approximation

    This ensures fair, scientifically meaningful comparison across encoders.

    References:
    - Vidal et al., Phys. Rev. Lett. 90, 227902 (2003) - S_vN
    - Barrett & Seth, PLoS Comput Biol 7(1): e1001052 (2011) - Φ_G
    - Oizumi et al., PLoS Comput Biol 12(3): e1004654 (2016) - IIT

    Args:
        representations: (batch, dim) or (batch, seq_len, dim) tensor

    Returns:
        coherence: (batch,) tensor - combined S_vN and Φ_G
    """
    with torch.no_grad():
        # Flatten 3D to 2D
        if representations.dim() == 3:
            B, T, D = representations.shape
            X = representations.reshape(B, T * D)
        else:
            X = representations

        B, dim = X.shape

        # Normalize to unit norm (pure state condition for S_vN)
        psi = F.normalize(X, dim=-1)

        # Compute metrics at multiple partitions
        S_vN_values = []
        phi_g_values = []

        for frac in [0.25, 0.33, 0.5]:
            partition_idx = max(1, min(dim - 1, int(dim * frac)))

            S_vN = compute_von_neumann_entropy(psi, partition_idx)
            phi_g = compute_geometric_phi(X, partition_idx)

            S_vN_values.append(S_vN)
            phi_g_values.append(phi_g)

        # Average across partitions
        S_vN_avg = torch.stack(S_vN_values, dim=-1).mean(dim=-1)
        phi_g_avg = torch.stack(phi_g_values, dim=-1).mean(dim=-1)

        # Normalize to [0, 1] range
        max_entropy = math.log(dim // 2 + 1)
        S_vN_normalized = S_vN_avg / (max_entropy + 1e-8)
        phi_g_normalized = torch.clamp(phi_g_avg, 0, 10) / 10.0

        # Combined metric
        coherence = (S_vN_normalized + phi_g_normalized) / 2.0
        coherence = torch.clamp(coherence, 0.0, 1.0)

        return coherence


class MLPEncoder(nn.Module):
    """Baseline MLP encoder for comparison.

    NOTE: MLP flattens temporal history, losing structure.
    Uses unified coherence metric for fair comparison with other encoders.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        latent = self.net(x)

        # FIX: Use unified coherence metric for fair comparison across encoders
        coherence = compute_representation_coherence(latent)

        return latent, {'phi_q': coherence}


class GRUEncoder(nn.Module):
    """GRU-based encoder baseline for temporal comparison.

    Standard recurrent baseline to compare against MERA's hierarchical structure.
    """

    def __init__(self, obs_dim: int, history_len: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.hidden_dim = hidden_dim

        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            obs_history: (batch, history_len, obs_dim)
        Returns:
            latent: (batch, hidden_dim), aux: Dict
        """
        # GRU forward
        gru_out, h_n = self.gru(obs_history)  # gru_out: (B, T, H), h_n: (num_layers, B, H)

        # Use last hidden state
        latent = h_n[-1]  # (B, H)
        latent = self.output_projection(latent)

        # FIX: Use unified coherence metric for fair comparison across encoders
        # Pass gru_out (B, T, H) for temporal coherence computation
        coherence = compute_representation_coherence(gru_out)

        return latent, {'phi_q': coherence}


class TransformerEncoder(nn.Module):
    """Transformer-based encoder baseline for temporal comparison.

    Modern attention-based baseline to compare against MERA.
    Tests whether MERA's hierarchical structure provides benefits over attention.
    """

    def __init__(self, obs_dim: int, history_len: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(obs_dim, d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, history_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.output_dim = d_model

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            obs_history: (batch, history_len, obs_dim)
        Returns:
            latent: (batch, d_model), aux: Dict
        """
        batch_size, seq_len, _ = obs_history.shape

        # Project input
        x = self.input_projection(obs_history)  # (B, T, d_model)

        # Add positional encoding (handle variable sequence lengths)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Transformer forward
        x = self.transformer(x)  # (B, T, d_model)

        # Mean pool over sequence (could also use CLS token)
        latent = x.mean(dim=1)  # (B, d_model)
        latent = self.output_projection(latent)

        # FIX: Use unified coherence metric for fair comparison across encoders
        # Pass transformer output (B, T, d_model) for temporal coherence computation
        coherence = compute_representation_coherence(x)

        return latent, {'phi_q': coherence}


class MERAEncoder(nn.Module):
    """MERA-based encoder that processes observation history"""

    def __init__(self, obs_dim: int, history_len: int, mera_config: EnhancedMERAConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len

        self.mera = EnhancedTensorNetworkMERA(mera_config)
        self.output_dim = self.mera.output_dim

        # Project observation to MERA input dim
        self.obs_projection = nn.Sequential(
            nn.Linear(obs_dim, mera_config.physical_dim * 4),
            nn.ReLU(),
            nn.Linear(mera_config.physical_dim * 4, mera_config.physical_dim * 2),
        )

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            obs_history: (batch, history_len, obs_dim)
        Returns:
            latent: (batch, output_dim), aux: Dict
        """
        projected = self.obs_projection(obs_history)
        latent, aux = self.mera(projected)
        return latent, aux

    def set_step(self, step: int):
        """Update step counter for warmup scheduling"""
        self.mera.set_step(step)


class SpatioTemporalEncoder(nn.Module):
    """Combined UPRT (spatial) + MERA (temporal) encoder.

    Fixes the "split brain" problem by connecting:
    - UPRT fields: spatial activity accumulators (activity, interaction, memory)
    - MERA: temporal hierarchical encoding

    The spatial features from UPRT fields are combined with MERA's temporal
    latent to create a unified spatio-temporal representation.
    """

    def __init__(self, obs_dim: int, history_len: int, mera_config: EnhancedMERAConfig,
                 config: dict, device: str = 'cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.device = device

        # Store world grid size from config (for position normalization)
        # Register as buffer for proper device handling
        self.register_buffer('world_grid_size', torch.tensor(
            config['environment']['grid_size'], dtype=torch.float32
        ))

        # Temporal encoder (MERA)
        self.mera_encoder = MERAEncoder(obs_dim, history_len, mera_config)
        mera_dim = self.mera_encoder.output_dim

        # Spatial encoder (UPRT field sampling)
        self.uprt_field = UPRTField(config)
        # UPRT fields: activity(16) + interaction(16) + memory(32) = 64 channels
        spatial_dim = 64

        # Spatial feature projection
        self.spatial_projection = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        spatial_out_dim = 64

        # Fusion layer: combine spatial + temporal
        self.fusion = nn.Sequential(
            nn.Linear(mera_dim + spatial_out_dim, mera_dim),
            nn.ReLU(),
            nn.Linear(mera_dim, mera_dim),
        )

        self.output_dim = mera_dim

    def sample_field_at_position(self, position: torch.Tensor) -> torch.Tensor:
        """Sample UPRT field values at robot position.

        Args:
            position: (batch, 2) robot positions in world coordinates

        Returns:
            field_features: (batch, 64) concatenated field values
        """
        batch_size = position.shape[0]

        # Normalize position to field grid coordinates using actual world grid size
        grid_size = self.uprt_field.grid_resolution
        world_size = self.world_grid_size.to(position.device)
        normalized = position / world_size  # Use config grid_size, not hardcoded 50.0
        grid_x = (normalized[:, 0] * (grid_size[0] - 1)).long().clamp(0, grid_size[0] - 1)
        grid_y = (normalized[:, 1] * (grid_size[1] - 1)).long().clamp(0, grid_size[1] - 1)

        # Sample from each field
        activity = self.uprt_field.activity_field[grid_x, grid_y]  # (batch, 16)
        interaction = self.uprt_field.interaction_field[grid_x, grid_y]  # (batch, 16)
        memory = self.uprt_field.memory_field[grid_x, grid_y]  # (batch, 32)

        return torch.cat([activity, interaction, memory], dim=-1)  # (batch, 64)

    def forward(self, obs_history: torch.Tensor,
                robot_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            obs_history: (batch, history_len, obs_dim)
            robot_positions: (batch, 2) current robot positions for field sampling

        Returns:
            latent: (batch, output_dim), aux: Dict
        """
        # Temporal encoding via MERA
        mera_latent, mera_aux = self.mera_encoder(obs_history)

        # Spatial encoding via UPRT field sampling
        if robot_positions is not None:
            spatial_features = self.sample_field_at_position(robot_positions)
            spatial_features = self.spatial_projection(spatial_features)

            # Fuse spatial + temporal
            combined = torch.cat([mera_latent, spatial_features], dim=-1)
            latent = self.fusion(combined)
        else:
            latent = mera_latent

        # Add spatial info to aux
        mera_aux['spatial_integrated'] = robot_positions is not None

        return latent, mera_aux

    def set_step(self, step: int):
        """Update step counter for warmup scheduling"""
        self.mera_encoder.set_step(step)

    def update_uprt_fields(self, robot_positions: torch.Tensor,
                           robot_activities: torch.Tensor, dt: float = 0.1):
        """Update UPRT field dynamics based on robot activity"""
        self.uprt_field.update_fields(robot_positions, robot_activities, dt)


class PPOActorCritic(nn.Module):
    """PPO Actor-Critic with multiple encoder options.

    Supported encoders:
    - mera: MERA tensor network (hierarchical, physics-inspired)
    - mera_uprt: MERA + UPRT spatial fields (fixes split-brain problem)
    - gru: GRU recurrent baseline (standard temporal)
    - transformer: Transformer baseline (attention-based)
    - mlp: MLP baseline (no temporal structure)
    """

    def __init__(self, obs_dim: int, action_dim: int, history_len: int,
                 encoder_type: str = "mera", mera_config: Optional[EnhancedMERAConfig] = None,
                 config: Optional[dict] = None, device: str = 'cpu'):
        super().__init__()
        self.encoder_type = encoder_type
        self.action_dim = action_dim
        self.history_len = history_len

        # Create encoder based on type
        if encoder_type == "mera":
            if mera_config is None:
                mera_config = EnhancedMERAConfig(
                    num_layers=3, bond_dim=8, physical_dim=4,
                    enable_hierarchical_entropy=True, use_identity_init=True,
                    enforce_scaling_bounds=True
                )
            self.encoder = MERAEncoder(obs_dim, history_len, mera_config)
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "mera_uprt":
            # Combined spatio-temporal encoder: MERA (temporal) + UPRT (spatial)
            if mera_config is None:
                mera_config = EnhancedMERAConfig(
                    num_layers=3, bond_dim=8, physical_dim=4,
                    enable_hierarchical_entropy=True, use_identity_init=True,
                    enforce_scaling_bounds=True
                )
            if config is None:
                raise ValueError("config required for mera_uprt encoder")
            self.encoder = SpatioTemporalEncoder(obs_dim, history_len, mera_config, config, device)
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "gru":
            self.encoder = GRUEncoder(obs_dim, history_len, hidden_dim=256, num_layers=2)
            encoder_dim = self.encoder.output_dim
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(obs_dim, history_len, d_model=128, nhead=4, num_layers=2)
            encoder_dim = self.encoder.output_dim
        else:  # mlp
            flat_dim = obs_dim * history_len
            self.encoder = MLPEncoder(flat_dim, 256)
            encoder_dim = 256

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(encoder_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(encoder_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Φ_Q integration for value
        if encoder_type == "mera":
            self.phi_q_value_weight = nn.Parameter(torch.tensor(0.1))

        # Action scaling bounds (from warehouse_env action space)
        # [linear_vel, angular_vel, gripper]
        self.action_low = torch.tensor([-2.0, -1.57, 0.0])
        self.action_high = torch.tensor([2.0, 1.57, 1.0])

    def forward(self, obs_history: torch.Tensor,
                robot_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if self.encoder_type == "mlp":
            batch_size = obs_history.shape[0]
            obs_flat = obs_history.reshape(batch_size, -1)
            latent, aux = self.encoder(obs_flat)
        elif self.encoder_type == "mera_uprt":
            # Pass robot_positions to SpatioTemporalEncoder for UPRT field sampling
            latent, aux = self.encoder(obs_history, robot_positions)
        else:
            latent, aux = self.encoder(obs_history)

        actor_features = self.actor(latent)
        action_mean = self.actor_mean(actor_features)
        value = self.critic(latent).squeeze(-1)

        # NOTE: phi_q is now diagnostic-only, NOT used in value function
        # This ensures fair comparison between MERA and baselines
        # Previous code added phi_q to value for MERA only, creating unfair advantage
        # if self.encoder_type == "mera" and aux['phi_q'] is not None:
        #     value = value + self.phi_q_value_weight * aux['phi_q']

        return action_mean, value, aux

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] (tanh output) to action space bounds"""
        low = self.action_low.to(action.device)
        high = self.action_high.to(action.device)
        # tanh bounds to [-1, 1], then scale to [low, high]
        scaled = low + (torch.tanh(action) + 1.0) / 2.0 * (high - low)
        return scaled

    def get_action(self, obs_history: torch.Tensor, deterministic: bool = False,
                   robot_positions: Optional[torch.Tensor] = None):
        action_mean, value, aux = self(obs_history, robot_positions)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        raw_action = action_mean if deterministic else dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)

        # Scale action to environment bounds
        scaled_action = self.scale_action(raw_action)

        phi_q = aux['phi_q'].mean().item() if aux['phi_q'] is not None else 0.0

        # Return both: scaled for env, raw for PPO updates
        return scaled_action, raw_action, value, log_prob, phi_q, aux

    def evaluate_actions(self, obs_history: torch.Tensor, actions: torch.Tensor,
                        robot_positions: Optional[torch.Tensor] = None):
        action_mean, value, aux = self(obs_history, robot_positions)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return value, log_prob, entropy, aux

    def set_step(self, step: int):
        """Update step counter for warmup scheduling (MERA only)"""
        if self.encoder_type == "mera" and hasattr(self.encoder, 'set_step'):
            self.encoder.set_step(step)

    def update_uprt_fields(self, robot_positions: torch.Tensor,
                           robot_activities: torch.Tensor, dt: float = 0.1):
        """Update UPRT field dynamics based on robot activity (mera_uprt only)"""
        if self.encoder_type == "mera_uprt" and hasattr(self.encoder, 'update_uprt_fields'):
            self.encoder.update_uprt_fields(robot_positions, robot_activities, dt)


# =============================================================================
# PPO Trainer with Warehouse Integration
# =============================================================================

class MERAWarehousePPO:
    """PPO trainer using existing WarehouseEnv and MERA integration"""

    def __init__(self, config_path: str = "config.yaml", encoder_type: str = "mera",
                 num_epochs: int = 100, device: str = None, config_override: dict = None):
        # Load config (or use override)
        if config_override is not None:
            self.config = config_override
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder_type = encoder_type
        self.num_epochs = num_epochs

        # Override max_steps for training (original is 5000)
        # Increased from 500 to 1000 to give agents more time to complete pickup-delivery
        self.config['environment']['max_episode_steps'] = 1000

        # Create environment
        self.env = WarehouseEnv(self.config)
        self.num_robots = self.config['environment']['num_robots']

        # Get dimensions from environment
        sample_obs = self.env.reset()
        self.obs_dim = len(sample_obs[0])
        self.action_dim = self.env.action_space.shape[0]

        # PPO hyperparameters from config
        learning_config = self.config['learning']
        self.gamma = learning_config['actor_critic']['discount']
        self.gae_lambda = learning_config['actor_critic']['lambda_gae']
        self.lr = learning_config['learning_rate']
        self.grad_clip = learning_config['grad_clip']

        # Training params
        # Increased history_len from 16 to 32 for better temporal context (helps Φ_Q)
        self.history_len = 32
        # steps_per_epoch must be > max_steps * num_robots to complete at least 1 episode
        # With max_steps=1000 and 8 robots: need > 8000 steps
        self.steps_per_epoch = max(8192, self.env.max_steps * self.num_robots * 2)
        self.num_minibatches = 8
        self.update_epochs = 4
        self.clip_epsilon = 0.2
        self.entropy_coef = learning_config['actor_critic']['entropy_weight']
        self.value_coef = 0.5

        # MERA config
        mera_config = EnhancedMERAConfig(
            num_layers=3,
            bond_dim=self.config['agent']['world_model']['latent_dim'] // 32,
            physical_dim=4,
            enable_hierarchical_entropy=True,
            use_identity_init=True,
            enforce_scaling_bounds=True,
            scaling_weight=0.01,  # Reduced per experiment findings
        )

        # Create network (pass config for mera_uprt encoder)
        self.network = PPOActorCritic(
            self.obs_dim, self.action_dim, self.history_len,
            encoder_type=encoder_type, mera_config=mera_config,
            config=self.config, device=str(self.device)
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Observation history per robot
        self.obs_history = {i: deque(maxlen=self.history_len) for i in range(self.num_robots)}

        # Tracking
        self.global_step = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.coordination_history = []
        self.phi_q_history = []
        self.phi_q_vs_coordination = []

        # Incremental epoch tracking
        self.epoch_reward = 0.0
        self.epoch_phi_q = []
        self.epoch_steps = 0

        # Trajectory recording for holographic training (Cosmic Egg, Genome Distillation)
        self.trajectory_recorder = TrajectoryRecorder(capacity=50000, record_every=1)

        # Output
        self.output_dir = Path(f"./results_{encoder_type}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Environment: max_steps={self.env.max_steps}, robots={self.num_robots}")
        print(f"  Steps per epoch: {self.steps_per_epoch} (~{self.steps_per_epoch // (self.env.max_steps * self.num_robots)} episodes/epoch)")

    def _reset_obs_history(self, observations: Dict[int, np.ndarray]):
        for robot_id, obs in observations.items():
            self.obs_history[robot_id].clear()
            for _ in range(self.history_len):
                self.obs_history[robot_id].append(obs)

    def _get_obs_tensor(self, robot_id: int) -> torch.Tensor:
        obs_list = list(self.obs_history[robot_id])
        obs_array = np.stack(obs_list, axis=0)
        return torch.from_numpy(obs_array).float().unsqueeze(0).to(self.device)

    def _get_all_obs_tensors(self) -> torch.Tensor:
        tensors = [self._get_obs_tensor(i) for i in range(self.num_robots)]
        return torch.cat(tensors, dim=0)

    def _get_robot_positions(self) -> torch.Tensor:
        """Get current robot positions for UPRT field sampling"""
        positions = np.array([robot.position for robot in self.env.robots])
        return torch.from_numpy(positions).float().to(self.device)

    def collect_rollout(self) -> Tuple[Dict[int, List[Transition]], Dict]:
        """Collect experience from environment"""
        observations = self.env.reset()
        self._reset_obs_history(observations)

        robot_transitions = {i: [] for i in range(self.num_robots)}
        episode_phi_q = []
        episode_reward = 0.0
        episode_steps = 0

        # Track epoch-level metrics
        epoch_total_reward = 0.0
        epoch_phi_q_values = []
        epoch_env_stats = None

        max_steps = self.steps_per_epoch // self.num_robots

        for step in range(max_steps):
            obs_batch = self._get_all_obs_tensors()
            robot_positions = self._get_robot_positions() if self.encoder_type == 'mera_uprt' else None

            with torch.no_grad():
                scaled_actions, raw_actions, values, log_probs, phi_q, aux = self.network.get_action(
                    obs_batch, robot_positions=robot_positions
                )

            scaled_actions_np = scaled_actions.cpu().numpy()  # For env
            raw_actions_np = raw_actions.cpu().numpy()  # For PPO updates
            values_np = values.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            episode_phi_q.append(phi_q)
            epoch_phi_q_values.append(phi_q)

            # Execute in environment with SCALED actions
            actions_dict = {i: scaled_actions_np[i] for i in range(self.num_robots)}
            next_obs, rewards, dones, info = self.env.step(actions_dict)

            # Record trajectory for holographic training (if enabled)
            if self.trajectory_recorder.enabled:
                global_state = self.env.get_global_state_snapshot()
                self.trajectory_recorder.record(
                    global_state=global_state,
                    actions=actions_dict,
                    rewards=rewards,
                    phi_q=phi_q,
                    done=dones.get('__all__', False)
                )

            # === UPDATE UPRT FIELDS (fix split-brain problem) ===
            if self.encoder_type == 'mera_uprt':
                uprt_positions = []
                uprt_activities = []
                for i in range(self.num_robots):
                    if i in info:
                        uprt_positions.append(info[i]['position'])
                        # Create activity vector [velocity, carrying, noise...]
                        act = np.zeros(64)
                        vel = np.array(info[i].get('velocity', [0, 0]))
                        act[0] = np.linalg.norm(vel)
                        act[1] = 1.0 if info[i].get('carrying', False) else 0.0
                        uprt_activities.append(act)

                if uprt_positions:
                    pos_tensor = torch.tensor(np.array(uprt_positions), device=self.device).float()
                    act_tensor = torch.tensor(np.array(uprt_activities), device=self.device).float()
                    self.network.update_uprt_fields(pos_tensor, act_tensor)
            # =====================================================

            # Store transitions (use RAW actions for PPO updates)
            # Get positions BEFORE step was taken (same as used for action selection)
            positions_np = robot_positions.cpu().numpy() if robot_positions is not None else None
            for i in range(self.num_robots):
                robot_transitions[i].append(Transition(
                    obs=np.stack(list(self.obs_history[i]), axis=0),
                    action=raw_actions_np[i],  # Store raw for evaluate_actions
                    reward=rewards[i],
                    done=dones[i],
                    value=values_np[i],
                    log_prob=log_probs_np[i],
                    phi_q=phi_q,
                    robot_position=positions_np[i] if positions_np is not None else None,
                ))
                self.obs_history[i].append(next_obs[i])
                episode_reward += rewards[i]
                epoch_total_reward += rewards[i]

            episode_steps += 1
            self.global_step += self.num_robots

            # Update MERA step counter for warmup scheduling
            self.network.set_step(self.global_step)

            # Check for episode end
            if dones.get('__all__', False):
                self.episode_count += 1

                # Get coordination metrics from environment
                env_stats = self.env.get_statistics()
                epoch_env_stats = env_stats  # Keep last stats

                metrics = CoordinationMetrics(
                    total_collisions=env_stats['collisions'],
                    packages_delivered=env_stats['packages_delivered'],
                    packages_picked_up=env_stats.get('packages_picked_up', 0),
                    total_distance=env_stats.get('total_distance', 0),
                    total_energy=env_stats.get('total_energy', 0.001),
                    episode_length=episode_steps,
                    throughput=env_stats['throughput'],
                    avg_waiting_time=env_stats.get('avg_waiting_time', 0),
                    num_robots=self.num_robots,  # Added for normalized metrics
                )

                self.coordination_history.append(asdict(metrics))

                # Verbose episode logging for diagnostics
                min_del_dist = getattr(self.env, '_min_delivery_dist', float('inf'))
                print(f"  [Episode {self.episode_count}] "
                      f"Pickups: {env_stats.get('packages_picked_up', 0)}, "
                      f"Delivered: {env_stats['packages_delivered']}, "
                      f"Collisions: {env_stats['collisions']}, "
                      f"Steps: {episode_steps}, "
                      f"MinDelDist: {min_del_dist:.1f}m")
                # Reset min delivery distance for next episode
                self.env._min_delivery_dist = float('inf')

                if episode_phi_q:
                    avg_phi_q = np.mean(episode_phi_q)
                    self.phi_q_history.append(avg_phi_q)
                    self.phi_q_vs_coordination.append((avg_phi_q, metrics.synergy_score))

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_steps)

                # Reset for next episode
                observations = self.env.reset()
                self._reset_obs_history(observations)
                episode_phi_q = []
                episode_reward = 0.0
                episode_steps = 0

        # If no episodes completed, still record epoch-level stats
        if not self.coordination_history or epoch_env_stats is None:
            # Get current stats even if episode didn't complete
            env_stats = self.env.get_statistics()
            epoch_env_stats = env_stats

        # Store epoch-level phi_q for tracking
        if epoch_phi_q_values:
            self.epoch_phi_q = epoch_phi_q_values

        return robot_transitions, {'env_stats': epoch_env_stats, 'epoch_reward': epoch_total_reward}

    def compute_returns(self, transitions: List[Transition]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages.

        FIX: Proper bootstrap value handling for multi-episode rollouts:
        - If last transition is terminal (done=True): next_value = 0
        - If last transition is NOT terminal (truncated): next_value = V(s_last)

        This ensures we don't treat truncated rollouts as having zero future value.
        """
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]

        returns, advantages = [], []
        gae = 0.0

        for t in reversed(range(len(transitions))):
            if t == len(transitions) - 1:
                # FIX: For last transition, check if it was terminal or truncated
                # If truncated (not done), bootstrap from last value estimate
                # If terminal (done), next_value = 0
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # FIX: Normalize both advantages AND returns for stable value learning
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns, advantages

    def update(self, robot_transitions: Dict[int, List[Transition]]):
        """PPO update step with value clipping.

        FIX: Added PPO value clipping to prevent large value function updates.
        This is standard in PPO implementations (OpenAI, CleanRL, etc.)
        """
        all_transitions = []
        for i in range(self.num_robots):
            all_transitions.extend(robot_transitions[i])

        if not all_transitions:
            return

        obs = torch.tensor(np.stack([t.obs for t in all_transitions]),
                          dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([t.action for t in all_transitions]),
                              dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in all_transitions],
                                     dtype=torch.float32, device=self.device)
        # FIX: Store old values for value clipping
        old_values = torch.tensor([t.value for t in all_transitions],
                                  dtype=torch.float32, device=self.device)

        # Extract robot positions for mera_uprt encoder
        if self.encoder_type == 'mera_uprt' and all_transitions[0].robot_position is not None:
            robot_positions = torch.tensor(
                np.stack([t.robot_position for t in all_transitions]),
                dtype=torch.float32, device=self.device
            )
        else:
            robot_positions = None

        all_returns, all_advantages = [], []
        for i in range(self.num_robots):
            if robot_transitions[i]:
                returns, advantages = self.compute_returns(robot_transitions[i])
                all_returns.append(returns)
                all_advantages.append(advantages)

        if not all_returns:
            return

        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)

        # Fix: Ensure batch_size is at least 1 to prevent empty batches
        batch_size = max(1, len(all_transitions) // self.num_minibatches)

        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(all_transitions))

            for start in range(0, len(all_transitions), batch_size):
                end = min(start + batch_size, len(all_transitions))
                batch_idx = indices[start:end]

                # Skip empty batches (shouldn't happen with batch_size >= 1, but be safe)
                if len(batch_idx) == 0:
                    continue

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_old_values = old_values[batch_idx]  # FIX: Get old values
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_positions = robot_positions[batch_idx] if robot_positions is not None else None

                values, log_probs, entropy, aux = self.network.evaluate_actions(
                    batch_obs, batch_actions, batch_positions
                )

                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # FIX: Value loss with clipping (standard PPO technique)
                # Prevents large value function updates that can destabilize training
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (values - batch_returns) ** 2
                value_loss_clipped = (values_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()

                # MERA constraint losses (isometry, scale consistency)
                # Note: Φ_Q is tracked as a probe only - does NOT affect optimization
                if self.encoder_type in ["mera", "mera_uprt"]:
                    mera_loss = self.network.encoder.mera.get_total_loss(aux) if hasattr(self.network.encoder, 'mera') else \
                               self.network.encoder.mera_encoder.mera.get_total_loss(aux)
                else:
                    mera_loss = 0.0

                # Vanilla PPO loss: policy + value + entropy + MERA constraints
                loss = (policy_loss + self.value_coef * value_loss +
                       self.entropy_coef * entropy_loss + mera_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                self.optimizer.step()

    def compute_phi_q_correlation(self) -> Dict:
        """Compute correlation between Φ_Q and coordination"""
        num_samples = len(self.phi_q_vs_coordination)

        if num_samples < 2:
            return {
                'correlation': 0.0,
                'phi_q_mean': 0.0,
                'synergy_mean': 0.0,
                'num_samples': num_samples,
                # Backward/forward compatible aliases
                'phi_q': 0.0,
                'synergy': 0.0,
            }

        phi_q_vals = np.array([x[0] for x in self.phi_q_vs_coordination])
        synergy_vals = np.array([x[1] for x in self.phi_q_vs_coordination])

        phi_q_mean = float(phi_q_vals.mean())
        synergy_mean = float(synergy_vals.mean())

        # Pearson correlation (need enough samples for meaningful correlation)
        if num_samples < 10:
            correlation = 0.0
        else:
            numerator = np.sum((phi_q_vals - phi_q_mean) * (synergy_vals - synergy_mean))
            denominator = (np.sqrt(np.sum((phi_q_vals - phi_q_mean) ** 2)) *
                          np.sqrt(np.sum((synergy_vals - synergy_mean) ** 2)) + 1e-8)
            correlation = float(numerator / denominator)

        return {
            'correlation': correlation,
            'phi_q_mean': phi_q_mean,
            'synergy_mean': synergy_mean,
            'num_samples': num_samples,
            # Backward/forward compatible aliases
            'phi_q': phi_q_mean,
            'synergy': synergy_mean,
        }

    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("MERA-PPO Warehouse Training")
        print("=" * 70)
        print(f"Encoder: {self.encoder_type}")
        print(f"Robots: {self.num_robots}")
        print(f"Obs dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            robot_transitions, epoch_info = self.collect_rollout()
            self.update(robot_transitions)

            epoch_time = time.time() - epoch_start

            if epoch % 10 == 0 or epoch == 1:
                # Use epoch-level info for current stats
                env_stats = epoch_info.get('env_stats', {})
                epoch_reward = epoch_info.get('epoch_reward', 0)

                # Episode-based metrics (if episodes completed)
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else epoch_reward

                # Phi_Q from epoch or episodes
                if hasattr(self, 'epoch_phi_q') and self.epoch_phi_q:
                    avg_phi_q = np.mean(self.epoch_phi_q)
                elif self.phi_q_history:
                    avg_phi_q = np.mean(self.phi_q_history[-10:])
                else:
                    avg_phi_q = 0

                # Coordination metrics from env_stats or history
                if env_stats:
                    avg_collisions = env_stats.get('collisions', 0)
                    avg_delivered = env_stats.get('packages_delivered', 0)
                    avg_pickups = env_stats.get('packages_picked_up', 0)
                    avg_throughput = env_stats.get('throughput', 0)
                elif self.coordination_history:
                    recent = self.coordination_history[-10:]
                    avg_collisions = np.mean([m['total_collisions'] for m in recent])
                    avg_delivered = np.mean([m['packages_delivered'] for m in recent])
                    avg_pickups = np.mean([m.get('packages_picked_up', 0) for m in recent])
                    avg_throughput = np.mean([m['throughput'] for m in recent])
                else:
                    avg_collisions = avg_delivered = avg_pickups = avg_throughput = 0

                correlation = self.compute_phi_q_correlation()

                print(f"\nEpoch {epoch}/{self.num_epochs} ({epoch_time:.1f}s) [Episodes: {self.episode_count}]")
                print(f"  Reward:      {avg_reward:.2f}")
                print(f"  Φ_Q:         {avg_phi_q:.4f}")
                print(f"  Collisions:  {avg_collisions:.1f}")
                print(f"  Pickups:     {avg_pickups:.0f}")
                print(f"  Delivered:   {avg_delivered:.1f}")
                print(f"  Throughput:  {avg_throughput:.1f}/hr")
                print(f"  Φ_Q↔Synergy: r={correlation['correlation']:.3f}")

            if epoch % 25 == 0:
                self.save_checkpoint(epoch)

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Training Complete! Time: {total_time/60:.1f} min")
        print("=" * 70)

        self.final_analysis()
        return self.get_results()

    def save_checkpoint(self, epoch: int):
        path = self.output_dir / f"checkpoint_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'phi_q_history': self.phi_q_history,
                'coordination_history': self.coordination_history,
            }
        }, path)
        print(f"  Saved: {path}")

    def final_analysis(self):
        """Analyze Φ_Q vs coordination correlation"""
        print("\n" + "=" * 70)
        print("Φ_Q vs Coordination Analysis")
        print("=" * 70)

        corr = self.compute_phi_q_correlation()

        # Safe extraction with fallbacks (bulletproof against schema changes)
        phi_q_mean = float(corr.get('phi_q_mean', corr.get('phi_q', 0.0)))
        synergy_mean = float(corr.get('synergy_mean', corr.get('synergy', 0.0)))
        num_samples = int(corr.get('num_samples', 0))
        r = float(corr.get('correlation', 0.0))

        print(f"\nSamples: {num_samples}")
        print(f"Φ_Q mean: {phi_q_mean:.4f}")
        print(f"Synergy mean: {synergy_mean:.3f}")
        print(f"Pearson r: {r:.3f}")

        if r > 0.3:
            print("\n→ POSITIVE: Higher Φ_Q correlates with better coordination!")
        elif r < -0.3:
            print("\n→ NEGATIVE: Higher Φ_Q correlates with worse coordination.")
        else:
            print("\n→ WEAK: Φ_Q doesn't strongly predict coordination.")

        # Save analysis
        with open(self.output_dir / "analysis.json", 'w') as f:
            json.dump({'correlation': corr, 'encoder': self.encoder_type}, f, indent=2)

        # Save results.json for run_experiments.py compatibility
        results_data = {
            'final_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'avg_reward': float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0,
            'packages_delivered': sum(m.get('packages_delivered', 0) for m in self.coordination_history[-10:]) if self.coordination_history else 0,
            'collisions': sum(m.get('total_collisions', 0) for m in self.coordination_history[-10:]) if self.coordination_history else 0,
            'throughput': float(np.mean([m.get('throughput', 0) for m in self.coordination_history[-10:]])) if self.coordination_history else 0,
            'phi_q': phi_q_mean,  # Use safe extracted value
            'episode_steps': len(self.episode_rewards) * self.steps_per_epoch,
        }
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to {self.output_dir / 'results.json'}")

    def get_results(self) -> Dict:
        return {
            'episode_rewards': self.episode_rewards,
            'phi_q_history': self.phi_q_history,
            'coordination_history': self.coordination_history,
            'correlation': self.compute_phi_q_correlation(),
        }

    # =========================================================================
    # Trajectory Recording for Holographic Training
    # =========================================================================

    def enable_trajectory_recording(self, record_every: int = 1):
        """Enable trajectory recording for Cosmic Egg / Genome Distillation.

        Args:
            record_every: Record every N steps (1 = all, 10 = every 10th)
        """
        self.trajectory_recorder.enable(record_every)
        print(f"  Trajectory recording enabled (every {record_every} steps)")

    def disable_trajectory_recording(self):
        """Disable trajectory recording"""
        self.trajectory_recorder.disable()

    def get_recorded_trajectories(self, num_samples: Optional[int] = None) -> List[dict]:
        """Get recorded trajectories for training world models.

        Args:
            num_samples: Optional limit on returned samples

        Returns:
            List of trajectory records with state, actions, rewards, phi_q
        """
        return self.trajectory_recorder.get_trajectories(num_samples)

    def save_trajectories(self, path: Optional[str] = None):
        """Save recorded trajectories to disk.

        Args:
            path: Save path (default: output_dir/trajectories.pkl)
        """
        if path is None:
            path = str(self.output_dir / "trajectories.pkl")
        self.trajectory_recorder.save(path)
        print(f"  Saved {len(self.trajectory_recorder)} trajectories to {path}")

    def get_trajectory_stats(self) -> dict:
        """Get statistics about recorded trajectories"""
        n = len(self.trajectory_recorder)
        if n == 0:
            return {'count': 0}

        trajectories = self.trajectory_recorder.get_trajectories()
        phi_q_vals = [t['phi_q'] for t in trajectories]
        rewards = np.concatenate([t['rewards'] for t in trajectories])

        return {
            'count': n,
            'phi_q_mean': float(np.mean(phi_q_vals)),
            'phi_q_std': float(np.std(phi_q_vals)),
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'episodes_captured': sum(1 for t in trajectories if t['done']),
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MERA-PPO Warehouse Training")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--encoder', type=str, default='mera',
                        choices=['mera', 'mera_uprt', 'gru', 'transformer', 'mlp'],
                        help='Encoder type: mera (temporal), mera_uprt (spatial+temporal), gru, transformer, mlp')
    parser.add_argument('--robots', type=int, default=None,
                        help='Override number of robots (for scaling study: 4, 8, 16, 32)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--sparse_rewards', action='store_true',
                        help='Use sparse rewards (delivery only, no shaping)')
    parser.add_argument('--quick_test', action='store_true')
    args = parser.parse_args()

    # Set seed if provided for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Enable deterministic operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: torch.use_deterministic_algorithms(True) may cause errors with some operations
        print(f"Random seed: {args.seed} (deterministic mode enabled)")

    epochs = 5 if args.quick_test else args.epochs

    # Load and potentially modify config
    config_override = None
    if args.robots is not None or args.sparse_rewards:
        with open(args.config, 'r') as f:
            config_override = yaml.safe_load(f)

        if args.robots is not None:
            config_override['environment']['num_robots'] = args.robots
            print(f"Overriding num_robots to {args.robots}")

        if args.sparse_rewards:
            config_override['environment']['sparse_rewards'] = True
            print("Using sparse rewards")

    trainer = MERAWarehousePPO(
        config_path=args.config,
        encoder_type=args.encoder,
        num_epochs=epochs,
        config_override=config_override,
    )

    results = trainer.train()

    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    if results['episode_rewards']:
        print(f"Avg Reward (last 10): {np.mean(results['episode_rewards'][-10:]):.2f}")
    print(f"Φ_Q↔Synergy r: {results['correlation']['correlation']:.3f}")


if __name__ == "__main__":
    main()
