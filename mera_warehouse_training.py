"""
MERA-Enhanced Warehouse RL Training
====================================

End-to-end training pipeline integrating:
1. MERA tensor network encoder with Φ_Q intrinsic motivation
2. Warehouse multi-robot environment
3. World model-based RL (Dreamer-style)
4. Configurable long training with checkpoints

Usage:
    python mera_warehouse_training.py --epochs 100 --save-interval 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import copy

# Local imports
from mera_enhanced import EnhancedMERAConfig, EnhancedTensorNetworkMERA
from mera_rl_integration import MERATrainingConfig, MERAEnhancedWorldModel, MERAEnhancedTrainer


@dataclass
class WarehouseTrainingConfig:
    """Full training configuration"""
    # Training duration
    num_epochs: int = 100
    steps_per_epoch: int = 1000
    eval_interval: int = 10
    save_interval: int = 25

    # Environment
    num_envs: int = 4  # Parallel environments
    max_episode_steps: int = 500

    # MERA config
    mera_num_layers: int = 3
    mera_bond_dim: int = 8
    mera_physical_dim: int = 4

    # World model training
    world_model_batch_size: int = 32
    sequence_length: int = 50
    imagination_horizon: int = 15

    # Actor-Critic
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    world_model_lr: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    entropy_coef: float = 0.01

    # Intrinsic motivation
    phi_q_weight: float = 0.1
    entanglement_weight: float = 0.05
    intrinsic_reward_scale: float = 0.1

    # Constraint losses (tuned)
    isometry_weight: float = 0.1
    scale_consistency_weight: float = 0.01

    # Misc
    grad_clip: float = 10.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


class SimplePolicy(nn.Module):
    """Actor network for warehouse robots"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)


class ValueNetwork(nn.Module):
    """Critic network"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class SimpleWarehouseEnv:
    """
    Simplified warehouse environment for MERA training.
    Matches the interface expected by the training loop.
    """

    def __init__(self, config: WarehouseTrainingConfig):
        self.config = config
        self.grid_size = np.array([50.0, 50.0])
        self.num_robots = 4
        self.obs_dim = 64  # Simplified observation
        self.action_dim = 3  # [linear_vel, angular_vel, gripper]

        # State
        self.robot_positions = None
        self.robot_velocities = None
        self.packages = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        """Reset and return initial observation"""
        self.step_count = 0

        # Random robot positions
        self.robot_positions = np.random.uniform(5, 45, (self.num_robots, 2))
        self.robot_velocities = np.zeros((self.num_robots, 2))

        # Random packages
        self.packages = np.random.uniform(10, 40, (8, 2))
        self.package_picked = np.zeros(8, dtype=bool)
        self.package_delivered = np.zeros(8, dtype=bool)

        # Destinations
        self.destinations = np.array([
            [45, 45], [45, 5], [5, 45], [5, 5]
        ], dtype=np.float32)

        self.robot_carrying = np.full(self.num_robots, -1)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get observation vector"""
        obs = []

        # Robot states (4 robots * 6 = 24)
        for i in range(self.num_robots):
            obs.extend([
                self.robot_positions[i, 0] / self.grid_size[0],
                self.robot_positions[i, 1] / self.grid_size[1],
                self.robot_velocities[i, 0] / 2.0,
                self.robot_velocities[i, 1] / 2.0,
                1.0 if self.robot_carrying[i] >= 0 else 0.0,
                float(self.robot_carrying[i]) / 8.0 if self.robot_carrying[i] >= 0 else 0.0,
            ])

        # Package states (8 packages * 4 = 32)
        for i in range(8):
            if self.package_delivered[i]:
                obs.extend([0.0, 0.0, 0.0, 1.0])
            elif self.package_picked[i]:
                obs.extend([0.0, 0.0, 1.0, 0.0])
            else:
                obs.extend([
                    self.packages[i, 0] / self.grid_size[0],
                    self.packages[i, 1] / self.grid_size[1],
                    0.0, 0.0
                ])

        # Pad to obs_dim
        while len(obs) < self.obs_dim:
            obs.append(0.0)

        return np.array(obs[:self.obs_dim], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action for robot 0 (simplified single-agent view).

        Args:
            action: [linear_vel, angular_vel, gripper]

        Returns:
            obs, reward, done, info
        """
        self.step_count += 1

        # Apply action to robot 0
        linear_vel = np.clip(action[0], -2.0, 2.0)
        angle = action[1] * np.pi  # Map to [-pi, pi]
        gripper = action[2] > 0.5

        # Update velocity
        direction = np.array([np.cos(angle), np.sin(angle)])
        self.robot_velocities[0] = direction * linear_vel * 0.5

        # Update position
        new_pos = self.robot_positions[0] + self.robot_velocities[0]
        new_pos = np.clip(new_pos, 1, self.grid_size - 1)
        self.robot_positions[0] = new_pos

        # Simple AI for other robots
        for i in range(1, self.num_robots):
            self.robot_velocities[i] = np.random.randn(2) * 0.1
            self.robot_positions[i] = np.clip(
                self.robot_positions[i] + self.robot_velocities[i],
                1, self.grid_size - 1
            )

        # Calculate reward
        reward = 0.0

        # Pickup logic
        if gripper and self.robot_carrying[0] < 0:
            for i in range(8):
                if not self.package_picked[i] and not self.package_delivered[i]:
                    dist = np.linalg.norm(self.packages[i] - self.robot_positions[0])
                    if dist < 2.0:
                        self.package_picked[i] = True
                        self.robot_carrying[0] = i
                        reward += 5.0  # Pickup reward
                        break

        # Delivery logic
        if self.robot_carrying[0] >= 0:
            pkg_idx = self.robot_carrying[0]
            # Update package position
            self.packages[pkg_idx] = self.robot_positions[0].copy()

            # Check delivery
            for dest in self.destinations:
                dist = np.linalg.norm(self.robot_positions[0] - dest)
                if dist < 3.0:
                    self.package_delivered[pkg_idx] = True
                    self.robot_carrying[0] = -1
                    reward += 20.0  # Delivery reward
                    break

        # Proximity reward (encourage exploring towards packages)
        if self.robot_carrying[0] < 0:
            for i in range(8):
                if not self.package_picked[i] and not self.package_delivered[i]:
                    dist = np.linalg.norm(self.packages[i] - self.robot_positions[0])
                    if dist < 5.0:
                        reward += 0.1 * (5.0 - dist) / 5.0

        # Small step penalty
        reward -= 0.01

        # Check done
        done = self.step_count >= self.config.max_episode_steps
        done = done or all(self.package_delivered)

        info = {
            'packages_delivered': sum(self.package_delivered),
            'packages_picked': sum(self.package_picked),
        }

        return self._get_obs(), reward, done, info


class ReplayBuffer:
    """Experience replay buffer with sequence sampling"""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seq_len: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequences(self, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        if self.size < self.seq_len + 1:
            return None

        # Find valid starting points (not crossing episode boundaries)
        valid_starts = []
        for i in range(self.size - self.seq_len):
            # Check no done in the middle of sequence
            if not any(self.dones[i:i + self.seq_len - 1]):
                valid_starts.append(i)

        if len(valid_starts) < batch_size:
            return None

        # Sample starts
        starts = np.random.choice(valid_starts, size=batch_size, replace=False)

        # Build sequences
        obs_seq = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        action_seq = np.zeros((batch_size, self.seq_len, self.action_dim), dtype=np.float32)
        reward_seq = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        for i, start in enumerate(starts):
            obs_seq[i] = self.obs[start:start + self.seq_len]
            action_seq[i] = self.actions[start:start + self.seq_len]
            reward_seq[i] = self.rewards[start:start + self.seq_len]

        return {
            'obs': torch.from_numpy(obs_seq).to(device),
            'action': torch.from_numpy(action_seq).to(device),
            'reward': torch.from_numpy(reward_seq).to(device),
        }


class MERAWarehouseTrainer:
    """
    End-to-end trainer for MERA-enhanced warehouse RL.

    Training loop:
    1. Collect experience from environment
    2. Train world model on sequences (with MERA encoder)
    3. Imagine trajectories using world model
    4. Train actor-critic on imagined trajectories
    5. Compute and log MERA-specific metrics (Φ_Q, RG flow)
    """

    def __init__(self, config: WarehouseTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create environment
        self.env = SimpleWarehouseEnv(config)
        self.obs_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim

        # MERA training config
        self.mera_config = MERATrainingConfig(
            mera_num_layers=config.mera_num_layers,
            mera_bond_dim=config.mera_bond_dim,
            mera_physical_dim=config.mera_physical_dim,
            phi_q_weight=config.phi_q_weight,
            entanglement_weight=config.entanglement_weight,
            scale_consistency_weight=config.scale_consistency_weight,
            constraint_weight=config.isometry_weight,
            sequence_length=config.sequence_length,
            learning_rate=config.world_model_lr,
            batch_size=config.world_model_batch_size,
        )

        # Create world model
        self.world_model = MERAEnhancedWorldModel(
            self.obs_dim, self.action_dim, self.mera_config
        ).to(self.device)

        # State dimension for actor-critic
        state_dim = self.world_model.stochastic_dim + self.world_model.deterministic_dim

        # Create actor and critic
        self.actor = SimplePolicy(state_dim, self.action_dim).to(self.device)
        self.critic = ValueNetwork(state_dim).to(self.device)

        # Optimizers
        self.world_model_opt = optim.Adam(
            self.world_model.parameters(), lr=config.world_model_lr
        )
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=100000,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            seq_len=config.sequence_length,
        )

        # Metrics
        self.metrics = {
            'epoch': [],
            'world_model_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'phi_q': [],
            'constraint_loss': [],
            'scale_loss': [],
            'episode_reward': [],
            'packages_delivered': [],
            'rg_eigenvalues': [],
        }

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def collect_experience(self, num_steps: int) -> Dict[str, float]:
        """Collect experience from environment"""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_rewards = []
        packages_delivered = []

        for _ in range(num_steps):
            # Get action from actor (using random policy initially)
            if self.buffer.size < 1000:
                action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    # Encode through MERA
                    obs_seq = obs_t.unsqueeze(1).expand(-1, self.config.sequence_length, -1)
                    mera_latent, _ = self.world_model.encode_sequence(obs_seq)

                    # Get state
                    state = self.world_model.initial_state(1, self.device)
                    state = self.world_model.observe(
                        obs_t, torch.zeros(1, self.action_dim, device=self.device),
                        state, mera_latent
                    )

                    # Sample action
                    full_state = torch.cat([state['stoch'], state['deter']], dim=-1)
                    action, _ = self.actor.sample(full_state)
                    action = action.squeeze(0).cpu().numpy()

            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward

            # Store experience
            self.buffer.add(obs, action, reward, done)

            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                packages_delivered.append(info['packages_delivered'])
                obs = self.env.reset()
                episode_reward = 0.0

        return {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_packages_delivered': np.mean(packages_delivered) if packages_delivered else 0.0,
        }

    def train_world_model(self, num_batches: int) -> Dict[str, float]:
        """Train world model on sequences"""
        total_loss = 0.0
        total_constraint = 0.0
        total_scale = 0.0
        total_phi_q = 0.0
        rg_eigenvalues = []

        for _ in range(num_batches):
            batch = self.buffer.sample_sequences(
                self.config.world_model_batch_size, self.device
            )
            if batch is None:
                continue

            obs_seq = batch['obs']
            action_seq = batch['action']
            reward_seq = batch['reward']

            batch_size, seq_len = obs_seq.shape[:2]

            # Encode through MERA
            mera_latent, mera_aux = self.world_model.encode_sequence(obs_seq)

            # Forward through dynamics
            state = self.world_model.initial_state(batch_size, self.device)

            recon_loss = 0.0
            reward_loss = 0.0
            kl_loss = 0.0

            for t in range(seq_len - 1):
                state = self.world_model.observe(
                    obs_seq[:, t], action_seq[:, t], state, mera_latent
                )

                pred_obs = self.world_model.decode_obs(state)
                pred_reward = self.world_model.predict_reward(state)

                recon_loss += F.mse_loss(pred_obs, obs_seq[:, t + 1])
                reward_loss += F.mse_loss(pred_reward.squeeze(-1), reward_seq[:, t + 1])

                # KL
                prior_state = self.world_model.imagine(action_seq[:, t], state)
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(state['mean'], state['std']),
                    torch.distributions.Normal(prior_state['mean'], prior_state['std'])
                ).sum(-1).mean()
                kl_loss += kl

            recon_loss /= (seq_len - 1)
            reward_loss /= (seq_len - 1)
            kl_loss /= (seq_len - 1)

            # MERA losses
            mera_losses = self.world_model.get_mera_losses(mera_aux)
            constraint_loss = mera_losses.get('constraint', torch.tensor(0.0, device=self.device))
            scale_loss = mera_losses.get('scale_consistency', torch.tensor(0.0, device=self.device))

            # Total loss
            loss = recon_loss + reward_loss + 0.1 * kl_loss + constraint_loss + scale_loss

            # Optimize
            self.world_model_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
            self.world_model_opt.step()

            total_loss += loss.item()
            total_constraint += constraint_loss.item() if torch.is_tensor(constraint_loss) else 0.0
            total_scale += scale_loss.item() if torch.is_tensor(scale_loss) else 0.0
            total_phi_q += mera_aux['phi_q'].mean().item() if mera_aux['phi_q'] is not None else 0.0

            if mera_aux['rg_eigenvalues']:
                rg_eigenvalues.extend(mera_aux['rg_eigenvalues'])

        return {
            'world_model_loss': total_loss / max(num_batches, 1),
            'constraint_loss': total_constraint / max(num_batches, 1),
            'scale_loss': total_scale / max(num_batches, 1),
            'phi_q': total_phi_q / max(num_batches, 1),
            'rg_eigenvalues': rg_eigenvalues,
        }

    def train_actor_critic(self, num_batches: int) -> Dict[str, float]:
        """Train actor-critic using imagined rollouts"""
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(num_batches):
            batch = self.buffer.sample_sequences(
                self.config.world_model_batch_size, self.device
            )
            if batch is None:
                continue

            obs_seq = batch['obs']
            batch_size = obs_seq.shape[0]

            # Get initial state from world model
            with torch.no_grad():
                mera_latent, mera_aux = self.world_model.encode_sequence(obs_seq)
                state = self.world_model.initial_state(batch_size, self.device)
                state = self.world_model.observe(
                    obs_seq[:, 0],
                    torch.zeros(batch_size, self.action_dim, device=self.device),
                    state, mera_latent
                )

            # Imagine rollouts
            imagined_states = []
            imagined_actions = []
            imagined_rewards = []
            imagined_values = []

            for _ in range(self.config.imagination_horizon):
                full_state = torch.cat([state['stoch'], state['deter']], dim=-1)
                imagined_states.append(full_state)

                # Sample action
                action, log_prob = self.actor.sample(full_state)
                imagined_actions.append(action)

                # Predict reward and value
                with torch.no_grad():
                    reward = self.world_model.predict_reward(state)
                    # Add intrinsic reward
                    intrinsic = self.world_model.get_intrinsic_reward(mera_aux)
                    total_reward = reward.squeeze(-1) + self.config.intrinsic_reward_scale * intrinsic

                imagined_rewards.append(total_reward)
                imagined_values.append(self.critic(full_state))

                # Imagine next state
                state = self.world_model.imagine(action, state)

            # Compute returns with GAE
            imagined_states = torch.stack(imagined_states, dim=1)  # (B, H, state_dim)
            imagined_rewards = torch.stack(imagined_rewards, dim=1)  # (B, H)
            imagined_values = torch.stack(imagined_values, dim=1)  # (B, H)

            # Bootstrap value
            with torch.no_grad():
                final_state = torch.cat([state['stoch'], state['deter']], dim=-1)
                bootstrap_value = self.critic(final_state)

            # GAE
            advantages = torch.zeros_like(imagined_rewards)
            returns = torch.zeros_like(imagined_rewards)
            gae = 0.0

            for t in reversed(range(self.config.imagination_horizon)):
                if t == self.config.imagination_horizon - 1:
                    next_value = bootstrap_value
                else:
                    next_value = imagined_values[:, t + 1]

                delta = imagined_rewards[:, t] + self.config.gamma * next_value - imagined_values[:, t]
                gae = delta + self.config.gamma * self.config.lambda_gae * gae
                advantages[:, t] = gae
                returns[:, t] = advantages[:, t] + imagined_values[:, t]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Actor loss
            flat_states = imagined_states.reshape(-1, imagined_states.shape[-1])
            flat_actions = torch.stack(imagined_actions, dim=1).reshape(-1, self.action_dim)
            flat_advantages = advantages.reshape(-1)

            log_probs = self.actor.log_prob(flat_states, flat_actions)
            entropy = -log_probs.mean()
            actor_loss = -(log_probs * flat_advantages.detach()).mean() - self.config.entropy_coef * entropy

            # Critic loss
            flat_values = self.critic(flat_states)
            flat_returns = returns.reshape(-1)
            critic_loss = F.mse_loss(flat_values, flat_returns.detach())

            # Optimize actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
            self.actor_opt.step()

            # Optimize critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
            self.critic_opt.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            'actor_loss': total_actor_loss / max(num_batches, 1),
            'critic_loss': total_critic_loss / max(num_batches, 1),
        }

    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'world_model_state': self.world_model.state_dict(),
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'world_model_opt': self.world_model_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'config': asdict(self.config),
            'metrics': self.metrics,
        }

        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.world_model.load_state_dict(checkpoint['world_model_state'])
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
        self.world_model_opt.load_state_dict(checkpoint['world_model_opt'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt'])
        self.metrics = checkpoint['metrics']

        return checkpoint['epoch']

    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("MERA-Enhanced Warehouse RL Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Steps per epoch: {self.config.steps_per_epoch}")
        print(f"MERA layers: {self.config.mera_num_layers}")
        print(f"Bond dimension: {self.config.mera_bond_dim}")
        print(f"Isometry weight: {self.config.isometry_weight}")
        print(f"Scale consistency weight: {self.config.scale_consistency_weight}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # Collect experience
            collect_metrics = self.collect_experience(self.config.steps_per_epoch)

            # Train world model
            wm_metrics = self.train_world_model(num_batches=50)

            # Train actor-critic
            ac_metrics = self.train_actor_critic(num_batches=50)

            # Log metrics
            self.metrics['epoch'].append(epoch)
            self.metrics['world_model_loss'].append(wm_metrics['world_model_loss'])
            self.metrics['actor_loss'].append(ac_metrics['actor_loss'])
            self.metrics['critic_loss'].append(ac_metrics['critic_loss'])
            self.metrics['phi_q'].append(wm_metrics['phi_q'])
            self.metrics['constraint_loss'].append(wm_metrics['constraint_loss'])
            self.metrics['scale_loss'].append(wm_metrics['scale_loss'])
            self.metrics['episode_reward'].append(collect_metrics['mean_episode_reward'])
            self.metrics['packages_delivered'].append(collect_metrics['mean_packages_delivered'])

            # RG analysis
            rg_evs = wm_metrics['rg_eigenvalues']
            if rg_evs:
                mean_rg = np.mean(rg_evs)
                near_fixed = sum(1 for ev in rg_evs if 0.9 < ev < 1.1) / len(rg_evs) * 100
            else:
                mean_rg = 0.0
                near_fixed = 0.0
            self.metrics['rg_eigenvalues'].append({'mean': mean_rg, 'near_fixed_pct': near_fixed})

            epoch_time = time.time() - epoch_start

            # Print progress
            print(f"\nEpoch {epoch}/{self.config.num_epochs} ({epoch_time:.1f}s)")
            print(f"  World Model Loss: {wm_metrics['world_model_loss']:.4f}")
            print(f"  Constraint Loss:  {wm_metrics['constraint_loss']:.6f}")
            print(f"  Scale Loss:       {wm_metrics['scale_loss']:.6f}")
            print(f"  Φ_Q:              {wm_metrics['phi_q']:.4f}")
            print(f"  Actor Loss:       {ac_metrics['actor_loss']:.4f}")
            print(f"  Critic Loss:      {ac_metrics['critic_loss']:.4f}")
            print(f"  Episode Reward:   {collect_metrics['mean_episode_reward']:.2f}")
            print(f"  Packages/Episode: {collect_metrics['mean_packages_delivered']:.2f}")
            print(f"  RG Mean:          {mean_rg:.4f} ({near_fixed:.1f}% near fixed point)")

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

            # Save metrics log
            if epoch % 10 == 0:
                log_path = Path(self.config.log_dir) / "training_log.json"
                with open(log_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Training complete! Total time: {total_time/60:.1f} minutes")
        print("=" * 70)

        # Final save
        self.save_checkpoint(self.config.num_epochs)

        return self.metrics


def analyze_training_results(metrics: Dict) -> Dict:
    """Analyze training results"""
    analysis = {}

    # Learning curves
    if metrics['world_model_loss']:
        initial_wm = np.mean(metrics['world_model_loss'][:5])
        final_wm = np.mean(metrics['world_model_loss'][-5:])
        analysis['world_model_improvement'] = (initial_wm - final_wm) / initial_wm * 100

    # Φ_Q trends
    if metrics['phi_q']:
        analysis['phi_q_mean'] = np.mean(metrics['phi_q'])
        analysis['phi_q_std'] = np.std(metrics['phi_q'])
        analysis['phi_q_trend'] = np.polyfit(range(len(metrics['phi_q'])), metrics['phi_q'], 1)[0]

    # RG convergence
    if metrics['rg_eigenvalues']:
        near_fixed_pcts = [m['near_fixed_pct'] for m in metrics['rg_eigenvalues']]
        analysis['rg_convergence'] = np.mean(near_fixed_pcts[-10:]) if len(near_fixed_pcts) >= 10 else np.mean(near_fixed_pcts)

    # Task performance
    if metrics['episode_reward']:
        analysis['final_reward'] = np.mean(metrics['episode_reward'][-10:])
        analysis['reward_improvement'] = np.mean(metrics['episode_reward'][-10:]) - np.mean(metrics['episode_reward'][:10])

    if metrics['packages_delivered']:
        analysis['final_packages'] = np.mean(metrics['packages_delivered'][-10:])

    return analysis


def main():
    parser = argparse.ArgumentParser(description='MERA Warehouse Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=1000, help='Environment steps per epoch')
    parser.add_argument('--save-interval', type=int, default=25, help='Checkpoint save interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (5 epochs)')
    args = parser.parse_args()

    # Create config
    config = WarehouseTrainingConfig(
        num_epochs=5 if args.quick_test else args.epochs,
        steps_per_epoch=100 if args.quick_test else args.steps_per_epoch,
        save_interval=args.save_interval,
        seed=args.seed,
    )

    # Create trainer
    trainer = MERAWarehouseTrainer(config)

    # Resume if specified
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # Train
    metrics = trainer.train()

    # Analyze
    print("\n" + "=" * 70)
    print("Training Analysis")
    print("=" * 70)

    analysis = analyze_training_results(metrics)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Training complete!")


if __name__ == "__main__":
    main()
