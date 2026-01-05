"""
MERA-RL Integration Module
==========================

Integrates Enhanced MERA with the QREA training pipeline:

1. MERAEnhancedTrainer: Training with MERA encoder + Φ_Q intrinsic motivation
2. MERAWorldModel: MERA-based world model for model-based RL
3. Experiment configurations for ablation studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import copy

try:
    from mera_enhanced import (
        EnhancedMERAConfig,
        EnhancedTensorNetworkMERA,
        MERAWorldModelEncoder,
        PhiQComputer
    )
except ImportError:
    # Handle case where running from different directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from mera_enhanced import (
        EnhancedMERAConfig,
        EnhancedTensorNetworkMERA,
        MERAWorldModelEncoder,
        PhiQComputer
    )


@dataclass
class MERATrainingConfig:
    """Configuration for MERA-enhanced training"""
    # MERA architecture
    mera_num_layers: int = 3
    mera_bond_dim: int = 8
    mera_physical_dim: int = 4

    # Intrinsic motivation weights
    phi_q_weight: float = 0.1
    entanglement_weight: float = 0.05
    scale_consistency_weight: float = 0.001  # Reduced: was hurting performance
    constraint_weight: float = 0.1  # Increased for isometry enforcement

    # Warmup for scale consistency (start at 0, ramp to full weight)
    scale_loss_warmup_steps: int = 1000  # Steps before full scale loss

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 32
    sequence_length: int = 50
    grad_clip: float = 10.0

    # Ablation flags
    use_phi_q_intrinsic: bool = True
    use_entanglement_intrinsic: bool = True
    use_constraint_loss: bool = True
    use_scale_loss: bool = True

    # Comparison baselines
    baseline_encoder: str = "mera"  # "mera", "mlp", "transformer"


class MERAEnhancedWorldModel(nn.Module):
    """
    World model with MERA encoder for hierarchical temporal representations.

    Components:
    1. MERA Encoder: Temporal sequence → hierarchical latent
    2. Dynamics Model: Predicts next latent state
    3. Reward Predictor: Predicts reward from latent
    4. Decoder: Reconstructs observations
    """

    def __init__(self, obs_dim: int, action_dim: int, config: MERATrainingConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Latent dimensions
        self.latent_dim = 256
        self.stochastic_dim = 32
        self.deterministic_dim = 256

        # MERA config
        mera_config = EnhancedMERAConfig(
            num_layers=config.mera_num_layers,
            bond_dim=config.mera_bond_dim,
            physical_dim=config.mera_physical_dim,
            temporal_window=config.sequence_length,
            enable_phi_q=config.use_phi_q_intrinsic,
            enforce_isometry=config.use_constraint_loss,
            enforce_unitarity=config.use_constraint_loss,
            phi_q_intrinsic_weight=config.phi_q_weight,
            entanglement_exploration_weight=config.entanglement_weight,
        )

        # MERA encoder
        self.mera_encoder = EnhancedTensorNetworkMERA(mera_config)

        # Project MERA output to latent space
        self.mera_to_latent = nn.Sequential(
            nn.Linear(self.mera_encoder.output_dim, 512),
            nn.ELU(),
            nn.Linear(512, self.latent_dim),
        )

        # Recurrent dynamics (GRU)
        self.dynamics_rnn = nn.GRUCell(
            self.stochastic_dim + action_dim,
            self.deterministic_dim
        )

        # Stochastic state (posterior: given observation)
        self.posterior = nn.Sequential(
            nn.Linear(self.deterministic_dim + self.latent_dim, 512),
            nn.ELU(),
            nn.Linear(512, 2 * self.stochastic_dim)  # mean, std
        )

        # Stochastic state (prior: without observation)
        self.prior = nn.Sequential(
            nn.Linear(self.deterministic_dim, 512),
            nn.ELU(),
            nn.Linear(512, 2 * self.stochastic_dim)
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.stochastic_dim + self.deterministic_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

        # Observation decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.stochastic_dim + self.deterministic_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, obs_dim)
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get initial RNN state"""
        return {
            'deter': torch.zeros(batch_size, self.deterministic_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stochastic_dim, device=device),
        }

    def encode_sequence(self, obs_sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Encode observation sequence through MERA.

        Args:
            obs_sequence: (batch, seq_len, obs_dim)

        Returns:
            latent: (batch, latent_dim) MERA-encoded representation
            mera_aux: Auxiliary outputs (Φ_Q, etc.)
        """
        mera_out, mera_aux = self.mera_encoder(obs_sequence)
        latent = self.mera_to_latent(mera_out)
        return latent, mera_aux

    def observe(self, obs: torch.Tensor, action: torch.Tensor,
                prev_state: Dict[str, torch.Tensor],
                mera_latent: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Update state given observation (posterior).

        Args:
            obs: (batch, obs_dim) current observation
            action: (batch, action_dim) previous action
            prev_state: Previous state dict
            mera_latent: Optional pre-computed MERA latent

        Returns:
            New state dict with posterior stochastic state
        """
        # Deterministic update
        x = torch.cat([prev_state['stoch'], action], dim=-1)
        deter = self.dynamics_rnn(x, prev_state['deter'])

        # Get MERA latent if not provided
        if mera_latent is None:
            # Use obs directly (single timestep fallback)
            obs_embed = obs  # Would need embedding here
        else:
            obs_embed = mera_latent

        # Posterior distribution
        post_input = torch.cat([deter, obs_embed], dim=-1)
        post_params = self.posterior(post_input)
        mean, log_std = torch.chunk(post_params, 2, dim=-1)
        std = F.softplus(log_std) + 0.1

        # Sample stochastic state
        stoch = mean + std * torch.randn_like(std)

        return {
            'deter': deter,
            'stoch': stoch,
            'mean': mean,
            'std': std,
        }

    def imagine(self, action: torch.Tensor,
                prev_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict next state without observation (prior).

        Args:
            action: (batch, action_dim)
            prev_state: Previous state

        Returns:
            Predicted next state
        """
        # Deterministic update
        x = torch.cat([prev_state['stoch'], action], dim=-1)
        deter = self.dynamics_rnn(x, prev_state['deter'])

        # Prior distribution
        prior_params = self.prior(deter)
        mean, log_std = torch.chunk(prior_params, 2, dim=-1)
        std = F.softplus(log_std) + 0.1

        stoch = mean + std * torch.randn_like(std)

        return {
            'deter': deter,
            'stoch': stoch,
            'mean': mean,
            'std': std,
        }

    def predict_reward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict reward from state"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        return self.reward_predictor(latent)

    def decode_obs(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode observation from state"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        return self.decoder(latent)

    def get_mera_losses(self, mera_aux: Dict) -> Dict[str, torch.Tensor]:
        """Extract MERA-specific losses"""
        losses = {}

        if self.config.use_constraint_loss:
            losses['constraint'] = mera_aux['constraint_loss'] * self.config.constraint_weight

        if self.config.use_scale_loss:
            losses['scale_consistency'] = mera_aux['scale_consistency_loss'] * self.config.scale_consistency_weight

        return losses

    def get_intrinsic_reward(self, mera_aux: Dict) -> torch.Tensor:
        """Get MERA-based intrinsic reward"""
        intrinsic = mera_aux['intrinsic_rewards']
        total = torch.zeros_like(intrinsic['total_intrinsic'])

        if self.config.use_phi_q_intrinsic:
            total = total + self.config.phi_q_weight * intrinsic['phi_q_reward']

        if self.config.use_entanglement_intrinsic:
            total = total + self.config.entanglement_weight * intrinsic['entanglement_reward']

        return total


class MERAEnhancedTrainer:
    """
    Trainer with MERA-enhanced world model and intrinsic motivation.

    Key features:
    1. MERA encoder for hierarchical temporal representations
    2. Φ_Q intrinsic motivation for exploration
    3. Scale consistency regularization
    4. Support for ablation studies
    """

    def __init__(self, world_model: MERAEnhancedWorldModel,
                 policy: nn.Module, value: nn.Module,
                 config: MERATrainingConfig, device: torch.device):
        self.world_model = world_model
        self.policy = policy
        self.value = value
        self.config = config
        self.device = device

        # Optimizers
        self.world_model_optimizer = optim.Adam(
            world_model.parameters(), lr=config.learning_rate
        )
        self.actor_optimizer = optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            value.parameters(), lr=config.learning_rate
        )

        # Replay buffer
        self.buffer = deque(maxlen=100000)

        # Metrics tracking
        self.metrics_history = {
            'world_model_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'phi_q_mean': [],
            'intrinsic_reward_mean': [],
            'constraint_loss': [],
            'scale_loss': [],
        }

    def add_experience(self, obs: np.ndarray, action: np.ndarray,
                       reward: float, next_obs: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
        })

    def sample_sequences(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample sequences from buffer"""
        if len(self.buffer) < self.config.sequence_length + 1:
            return None

        # Sample random starting indices
        max_start = len(self.buffer) - self.config.sequence_length
        if max_start <= 0:
            return None

        starts = np.random.randint(0, max_start, size=batch_size)

        obs_seqs = []
        action_seqs = []
        reward_seqs = []

        for start in starts:
            end = start + self.config.sequence_length
            obs_seqs.append([self.buffer[i]['obs'] for i in range(start, end)])
            action_seqs.append([self.buffer[i]['action'] for i in range(start, end)])
            reward_seqs.append([self.buffer[i]['reward'] for i in range(start, end)])

        return {
            'obs': torch.FloatTensor(np.array(obs_seqs)).to(self.device),
            'action': torch.FloatTensor(np.array(action_seqs)).to(self.device),
            'reward': torch.FloatTensor(np.array(reward_seqs)).unsqueeze(-1).to(self.device),
        }

    def train_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train world model with MERA encoder"""
        obs_seq = batch['obs']  # (B, T, obs_dim)
        action_seq = batch['action']  # (B, T, action_dim)
        reward_seq = batch['reward']  # (B, T, 1)

        batch_size, seq_len = obs_seq.shape[:2]

        # Encode full sequence through MERA
        mera_latent, mera_aux = self.world_model.encode_sequence(obs_seq)

        # Initialize state
        state = self.world_model.initial_state(batch_size, self.device)

        # Forward through sequence
        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0

        for t in range(seq_len - 1):
            # Observe current step
            state = self.world_model.observe(
                obs_seq[:, t],
                action_seq[:, t],
                state,
                mera_latent
            )

            # Predict and compute losses
            pred_obs = self.world_model.decode_obs(state)
            pred_reward = self.world_model.predict_reward(state)

            recon_loss += F.mse_loss(pred_obs, obs_seq[:, t + 1])
            reward_loss += F.mse_loss(pred_reward, reward_seq[:, t + 1])

            # KL divergence (posterior vs prior)
            prior_state = self.world_model.imagine(action_seq[:, t], state)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(state['mean'], state['std']),
                torch.distributions.Normal(prior_state['mean'], prior_state['std'])
            ).sum(dim=-1).mean()
            kl_loss += kl

        recon_loss /= (seq_len - 1)
        reward_loss /= (seq_len - 1)
        kl_loss /= (seq_len - 1)

        # MERA-specific losses
        mera_losses = self.world_model.get_mera_losses(mera_aux)
        constraint_loss = mera_losses.get('constraint', torch.tensor(0.0))
        scale_loss = mera_losses.get('scale_consistency', torch.tensor(0.0))

        # Total loss
        total_loss = recon_loss + reward_loss + 0.1 * kl_loss + constraint_loss + scale_loss

        # Optimize
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.world_model_optimizer.step()

        # Compute intrinsic reward for logging
        intrinsic_reward = self.world_model.get_intrinsic_reward(mera_aux)

        metrics = {
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
            'constraint_loss': constraint_loss.item() if torch.is_tensor(constraint_loss) else 0.0,
            'scale_loss': scale_loss.item() if torch.is_tensor(scale_loss) else 0.0,
            'phi_q_mean': mera_aux['phi_q'].mean().item() if mera_aux['phi_q'] is not None else 0.0,
            'intrinsic_reward_mean': intrinsic_reward.mean().item(),
            'total_loss': total_loss.item(),
        }

        return metrics

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step"""
        batch = self.sample_sequences(self.config.batch_size)
        if batch is None:
            return None

        metrics = self.train_world_model(batch)

        # Track metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        return metrics


# =============================================================================
# Ablation Study Configurations
# =============================================================================

def get_ablation_configs() -> Dict[str, MERATrainingConfig]:
    """Get configurations for ablation studies"""
    configs = {}

    # Full MERA (all features)
    configs['full_mera'] = MERATrainingConfig(
        use_phi_q_intrinsic=True,
        use_entanglement_intrinsic=True,
        use_constraint_loss=True,
        use_scale_loss=True,
    )

    # No Φ_Q intrinsic motivation
    configs['no_phi_q'] = MERATrainingConfig(
        use_phi_q_intrinsic=False,
        use_entanglement_intrinsic=True,
        use_constraint_loss=True,
        use_scale_loss=True,
    )

    # No entanglement intrinsic
    configs['no_entanglement'] = MERATrainingConfig(
        use_phi_q_intrinsic=True,
        use_entanglement_intrinsic=False,
        use_constraint_loss=True,
        use_scale_loss=True,
    )

    # No constraint regularization
    configs['no_constraints'] = MERATrainingConfig(
        use_phi_q_intrinsic=True,
        use_entanglement_intrinsic=True,
        use_constraint_loss=False,
        use_scale_loss=True,
    )

    # No scale consistency
    configs['no_scale_loss'] = MERATrainingConfig(
        use_phi_q_intrinsic=True,
        use_entanglement_intrinsic=True,
        use_constraint_loss=True,
        use_scale_loss=False,
    )

    # No MERA features (just architecture)
    configs['mera_only'] = MERATrainingConfig(
        use_phi_q_intrinsic=False,
        use_entanglement_intrinsic=False,
        use_constraint_loss=False,
        use_scale_loss=False,
    )

    return configs


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing MERA-RL Integration")
    print("=" * 70)

    # Configuration
    config = MERATrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create world model
    obs_dim = 64
    action_dim = 3

    print("\n1. Creating MERA-enhanced world model...")
    world_model = MERAEnhancedWorldModel(obs_dim, action_dim, config).to(device)
    print(f"   World model parameters: {sum(p.numel() for p in world_model.parameters()):,}")

    # Test forward pass
    print("\n2. Testing sequence encoding...")
    batch_size = 4
    seq_len = config.sequence_length

    obs_seq = torch.randn(batch_size, seq_len, obs_dim).to(device)
    action_seq = torch.randn(batch_size, seq_len, action_dim).to(device)

    mera_latent, mera_aux = world_model.encode_sequence(obs_seq)
    print(f"   MERA latent shape: {mera_latent.shape}")
    print(f"   Φ_Q: {mera_aux['phi_q'].mean().item():.4f}")

    # Test dynamics
    print("\n3. Testing dynamics model...")
    state = world_model.initial_state(batch_size, device)
    next_state = world_model.observe(obs_seq[:, 0], action_seq[:, 0], state, mera_latent)
    print(f"   State stoch shape: {next_state['stoch'].shape}")
    print(f"   State deter shape: {next_state['deter'].shape}")

    # Test imagined rollout
    print("\n4. Testing imagination...")
    imagined_state = world_model.imagine(action_seq[:, 0], next_state)
    pred_reward = world_model.predict_reward(imagined_state)
    print(f"   Predicted reward shape: {pred_reward.shape}")

    # Test intrinsic reward
    print("\n5. Testing intrinsic reward...")
    intrinsic = world_model.get_intrinsic_reward(mera_aux)
    print(f"   Intrinsic reward: {intrinsic.mean().item():.4f}")

    # Test MERA losses
    print("\n6. Testing MERA losses...")
    mera_losses = world_model.get_mera_losses(mera_aux)
    for name, loss in mera_losses.items():
        print(f"   {name}: {loss.item():.6f}")

    # Test training step
    print("\n7. Testing training...")
    policy = nn.Sequential(
        nn.Linear(world_model.stochastic_dim + world_model.deterministic_dim, 256),
        nn.ELU(),
        nn.Linear(256, action_dim)
    ).to(device)

    value = nn.Sequential(
        nn.Linear(world_model.stochastic_dim + world_model.deterministic_dim, 256),
        nn.ELU(),
        nn.Linear(256, 1)
    ).to(device)

    trainer = MERAEnhancedTrainer(world_model, policy, value, config, device)

    # Add some fake experience
    for _ in range(200):
        trainer.add_experience(
            obs=np.random.randn(obs_dim),
            action=np.random.randn(action_dim),
            reward=np.random.randn(),
            next_obs=np.random.randn(obs_dim),
            done=False
        )

    metrics = trainer.train_step()
    if metrics:
        print("   Training metrics:")
        for key, value in metrics.items():
            print(f"      {key}: {value:.4f}")

    # Show ablation configs
    print("\n8. Ablation configurations:")
    for name, cfg in get_ablation_configs().items():
        features = []
        if cfg.use_phi_q_intrinsic: features.append("Φ_Q")
        if cfg.use_entanglement_intrinsic: features.append("Ent")
        if cfg.use_constraint_loss: features.append("Const")
        if cfg.use_scale_loss: features.append("Scale")
        print(f"   {name}: {', '.join(features) if features else 'None'}")

    print("\n" + "=" * 70)
    print("All integration tests passed!")
