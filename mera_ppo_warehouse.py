"""
MERA-PPO Warehouse Training
============================

Connects MERA tensor network (brain) with warehouse environment (body)
to test if Φ_Q correlates with multi-agent coordination quality.

Key Research Questions:
1. Does higher Φ_Q during training correlate with better coordination?
2. Do MERA-encoded policies learn faster than MLP baselines?
3. Does the hierarchical structure help with multi-agent credit assignment?

Architecture:
- MERA Encoder: Processes observation history → hierarchical latent
- Shared Value Network: Estimates state value with Φ_Q features
- Per-Agent Policy: Outputs actions conditioned on MERA latent
- Coordination Tracker: Measures collisions, throughput, synergy

Usage:
    python mera_ppo_warehouse.py --epochs 100 --num_robots 4
    python mera_ppo_warehouse.py --baseline mlp  # Compare with MLP encoder
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
import copy

# Local imports
from mera_enhanced import EnhancedMERAConfig, EnhancedTensorNetworkMERA, PhiQComputer


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
    phi_q: float  # Track Φ_Q for correlation analysis


@dataclass
class CoordinationMetrics:
    """Metrics for multi-agent coordination quality"""
    total_collisions: int = 0
    packages_delivered: int = 0
    total_distance: float = 0.0
    total_energy: float = 0.0
    episode_length: int = 0

    # Coordination-specific
    near_misses: int = 0  # Close calls that didn't collide
    simultaneous_deliveries: int = 0  # Robots delivering at same time
    idle_time: float = 0.0  # Time robots spent idle

    @property
    def collision_rate(self) -> float:
        return self.total_collisions / max(self.episode_length, 1)

    @property
    def throughput(self) -> float:
        # Packages per 1000 steps
        return self.packages_delivered / max(self.episode_length, 1) * 1000

    @property
    def efficiency(self) -> float:
        # Packages per energy unit
        return self.packages_delivered / max(self.total_energy, 1)

    @property
    def synergy_score(self) -> float:
        """Higher when robots work together well"""
        # Low collisions + high throughput + low idle time
        collision_penalty = 1.0 / (1.0 + self.total_collisions)
        throughput_bonus = min(self.throughput, 1.0)
        active_ratio = 1.0 - (self.idle_time / max(self.episode_length, 1))
        return (collision_penalty + throughput_bonus + active_ratio) / 3.0


@dataclass
class TrainingConfig:
    """Full training configuration"""
    # Environment
    num_robots: int = 4
    grid_size: Tuple[int, int] = (50, 50)
    max_episode_steps: int = 500

    # MERA config
    mera_num_layers: int = 3
    mera_bond_dim: int = 8
    mera_physical_dim: int = 4
    observation_history: int = 16  # Frames to feed into MERA

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training schedule
    num_epochs: int = 100
    steps_per_epoch: int = 2048
    num_minibatches: int = 4
    update_epochs: int = 4

    # Φ_Q intrinsic motivation
    phi_q_intrinsic_weight: float = 0.1
    track_phi_q_correlation: bool = True

    # Comparison
    encoder_type: str = "mera"  # "mera" or "mlp"

    # Logging
    log_interval: int = 10
    save_interval: int = 25
    output_dir: str = "./mera_ppo_results"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# =============================================================================
# Neural Network Components
# =============================================================================

class MLPEncoder(nn.Module):
    """Baseline MLP encoder for comparison"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Returns latent and empty aux dict for compatibility"""
        return self.net(x), {'phi_q': torch.zeros(x.shape[0], device=x.device)}


class MERAEncoder(nn.Module):
    """MERA-based encoder that processes observation history"""

    def __init__(self, obs_dim: int, config: TrainingConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.config = config

        # MERA config
        mera_config = EnhancedMERAConfig(
            num_layers=config.mera_num_layers,
            bond_dim=config.mera_bond_dim,
            physical_dim=config.mera_physical_dim,
            temporal_window=config.observation_history,
            enable_phi_q=True,
            use_identity_init=True,
            enforce_rg_fixed_point=True,
            rg_eigenvalue_weight=0.05,
        )

        self.mera = EnhancedTensorNetworkMERA(mera_config)
        self.output_dim = self.mera.output_dim

        # Project observation to MERA input dim
        self.obs_projection = nn.Sequential(
            nn.Linear(obs_dim, config.mera_physical_dim * 4),
            nn.ReLU(),
            nn.Linear(config.mera_physical_dim * 4, config.mera_physical_dim * 2),
        )

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process observation history through MERA.

        Args:
            obs_history: (batch, history_len, obs_dim)

        Returns:
            latent: (batch, output_dim)
            aux: Dict with phi_q, rg_eigenvalues, etc.
        """
        # Project observations
        projected = self.obs_projection(obs_history)  # (B, T, mera_input_dim)

        # Pass through MERA
        latent, aux = self.mera(projected)

        return latent, aux


class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic with MERA or MLP encoder.

    The encoder processes observation history and outputs:
    - Policy: Gaussian distribution over continuous actions
    - Value: Scalar state value estimate
    - Φ_Q: Integrated information metric (for MERA)
    """

    def __init__(self, obs_dim: int, action_dim: int, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.action_dim = action_dim

        # Create encoder based on type
        if config.encoder_type == "mera":
            self.encoder = MERAEncoder(obs_dim, config)
            encoder_output_dim = self.encoder.output_dim
        else:
            # MLP baseline
            flat_dim = obs_dim * config.observation_history
            self.encoder = MLPEncoder(flat_dim, 256)
            encoder_output_dim = 256

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(encoder_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(encoder_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Φ_Q feature integration for value estimation
        if config.encoder_type == "mera":
            self.phi_q_value_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            obs_history: (batch, history_len, obs_dim)

        Returns:
            action_mean, value, aux_dict
        """
        # Flatten for MLP baseline
        if self.config.encoder_type == "mlp":
            batch_size = obs_history.shape[0]
            obs_flat = obs_history.reshape(batch_size, -1)
            latent, aux = self.encoder(obs_flat)
        else:
            latent, aux = self.encoder(obs_history)

        # Actor output
        actor_features = self.actor(latent)
        action_mean = self.actor_mean(actor_features)

        # Critic output
        value = self.critic(latent).squeeze(-1)

        # Add Φ_Q to value estimate (MERA only)
        if self.config.encoder_type == "mera" and aux['phi_q'] is not None:
            value = value + self.phi_q_value_weight * aux['phi_q']

        return action_mean, value, aux

    def get_action(self, obs_history: torch.Tensor, deterministic: bool = False):
        """Sample action from policy"""
        action_mean, value, aux = self(obs_history)
        action_std = self.actor_log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)

        phi_q = aux['phi_q'].mean().item() if aux['phi_q'] is not None else 0.0

        return action, value, log_prob, phi_q

    def evaluate_actions(self, obs_history: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        action_mean, value, aux = self(obs_history)
        action_std = self.actor_log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return value, log_prob, entropy, aux


# =============================================================================
# Simple Warehouse Environment (Self-Contained)
# =============================================================================

class SimpleMultiRobotWarehouse:
    """
    Simplified multi-robot warehouse for MERA-PPO training.

    Features:
    - Multiple robots with continuous control
    - Package pickup and delivery
    - Collision detection and tracking
    - Coordination metrics
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.grid_size = np.array(config.grid_size, dtype=np.float32)
        self.num_robots = config.num_robots
        self.max_steps = config.max_episode_steps

        # Robot parameters
        self.robot_radius = 0.5
        self.max_speed = 2.0
        self.dt = 0.1

        # Observation and action dimensions (per robot)
        self.obs_dim = 10 + 8 * (self.num_robots - 1) + 16  # self + others + packages
        self.action_dim = 3  # [velocity, angular_velocity, gripper]

        # State
        self.robot_positions = None
        self.robot_velocities = None
        self.robot_orientations = None
        self.robot_carrying = None
        self.packages = None
        self.package_destinations = None
        self.package_picked = None
        self.package_delivered = None

        self.step_count = 0
        self.metrics = CoordinationMetrics()

    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return observations for all robots"""
        self.step_count = 0
        self.metrics = CoordinationMetrics()

        # Initialize robots in spread-out positions
        self.robot_positions = np.zeros((self.num_robots, 2), dtype=np.float32)
        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            radius = min(self.grid_size) * 0.3
            center = self.grid_size / 2
            self.robot_positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])

        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_orientations = np.random.uniform(0, 2*np.pi, self.num_robots).astype(np.float32)
        self.robot_carrying = np.full(self.num_robots, -1, dtype=np.int32)

        # Initialize packages
        num_packages = 8
        self.packages = np.random.uniform(5, self.grid_size - 5, (num_packages, 2)).astype(np.float32)
        self.package_destinations = np.array([
            [self.grid_size[0] - 5, self.grid_size[1] - 5],
            [self.grid_size[0] - 5, 5],
            [5, self.grid_size[1] - 5],
            [5, 5],
        ] * 2, dtype=np.float32)[:num_packages]
        self.package_picked = np.zeros(num_packages, dtype=bool)
        self.package_delivered = np.zeros(num_packages, dtype=bool)

        return self._get_all_observations()

    def _get_all_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all robots"""
        return {i: self._get_observation(i) for i in range(self.num_robots)}

    def _get_observation(self, robot_id: int) -> np.ndarray:
        """Get observation for a single robot"""
        obs = []

        # Self state (10)
        pos = self.robot_positions[robot_id]
        vel = self.robot_velocities[robot_id]
        ori = self.robot_orientations[robot_id]
        carrying = self.robot_carrying[robot_id]

        obs.extend([
            pos[0] / self.grid_size[0],
            pos[1] / self.grid_size[1],
            vel[0] / self.max_speed,
            vel[1] / self.max_speed,
            np.cos(ori),
            np.sin(ori),
            1.0 if carrying >= 0 else 0.0,
            float(carrying) / 8.0 if carrying >= 0 else 0.0,
            np.linalg.norm(vel) / self.max_speed,
            0.0,  # Padding
        ])

        # Other robots (8 features per robot)
        for other_id in range(self.num_robots):
            if other_id == robot_id:
                continue

            other_pos = self.robot_positions[other_id]
            other_vel = self.robot_velocities[other_id]
            rel_pos = other_pos - pos
            rel_vel = other_vel - vel
            dist = np.linalg.norm(rel_pos)

            obs.extend([
                rel_pos[0] / self.grid_size[0],
                rel_pos[1] / self.grid_size[1],
                rel_vel[0] / self.max_speed,
                rel_vel[1] / self.max_speed,
                dist / np.linalg.norm(self.grid_size),
                np.cos(self.robot_orientations[other_id] - ori),
                np.sin(self.robot_orientations[other_id] - ori),
                1.0 if self.robot_carrying[other_id] >= 0 else 0.0,
            ])

        # Package info (closest 2 packages, 8 features each = 16)
        available_packages = [
            (i, np.linalg.norm(self.packages[i] - pos))
            for i in range(len(self.packages))
            if not self.package_delivered[i] and not self.package_picked[i]
        ]
        available_packages.sort(key=lambda x: x[1])

        for i in range(2):
            if i < len(available_packages):
                pkg_idx, pkg_dist = available_packages[i]
                pkg_pos = self.packages[pkg_idx]
                dest_pos = self.package_destinations[pkg_idx]

                obs.extend([
                    (pkg_pos[0] - pos[0]) / self.grid_size[0],
                    (pkg_pos[1] - pos[1]) / self.grid_size[1],
                    pkg_dist / np.linalg.norm(self.grid_size),
                    (dest_pos[0] - pos[0]) / self.grid_size[0],
                    (dest_pos[1] - pos[1]) / self.grid_size[1],
                    np.linalg.norm(dest_pos - pos) / np.linalg.norm(self.grid_size),
                    1.0,  # Available flag
                    0.0,
                ])
            else:
                obs.extend([0.0] * 8)

        return np.array(obs, dtype=np.float32)

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[int, bool],        # dones
        Dict[str, any],         # info
    ]:
        """Execute one environment step"""
        self.step_count += 1
        self.metrics.episode_length = self.step_count

        # Apply actions
        for robot_id, action in actions.items():
            self._apply_action(robot_id, action)

        # Update physics
        self._update_physics()

        # Handle packages
        self._handle_packages()

        # Check collisions
        collision_count = self._check_collisions()
        self.metrics.total_collisions += collision_count

        # Calculate rewards
        rewards = self._calculate_rewards(collision_count)

        # Check done
        done = self.step_count >= self.max_steps or all(self.package_delivered)
        dones = {i: done for i in range(self.num_robots)}
        dones['__all__'] = done

        # Get observations
        observations = self._get_all_observations()

        # Info
        info = {
            'metrics': asdict(self.metrics),
            'packages_delivered': int(sum(self.package_delivered)),
            'collision_count': collision_count,
        }

        return observations, rewards, dones, info

    def _apply_action(self, robot_id: int, action: np.ndarray):
        """Apply action to robot"""
        velocity_cmd = np.clip(action[0], -1, 1) * self.max_speed
        angular_cmd = np.clip(action[1], -1, 1) * np.pi
        gripper_cmd = action[2] > 0.0

        # Update velocity
        direction = np.array([
            np.cos(self.robot_orientations[robot_id]),
            np.sin(self.robot_orientations[robot_id])
        ])
        target_vel = direction * velocity_cmd

        # Smooth velocity update
        alpha = 0.3
        self.robot_velocities[robot_id] = (
            alpha * target_vel + (1 - alpha) * self.robot_velocities[robot_id]
        )

        # Update orientation
        self.robot_orientations[robot_id] += angular_cmd * self.dt
        self.robot_orientations[robot_id] %= (2 * np.pi)

        # Handle gripper
        if gripper_cmd:
            self._try_pickup_or_deliver(robot_id)

    def _update_physics(self):
        """Update positions"""
        for i in range(self.num_robots):
            new_pos = self.robot_positions[i] + self.robot_velocities[i] * self.dt
            new_pos = np.clip(new_pos, self.robot_radius, self.grid_size - self.robot_radius)

            self.metrics.total_distance += np.linalg.norm(new_pos - self.robot_positions[i])
            self.metrics.total_energy += np.linalg.norm(self.robot_velocities[i]) * 0.1

            self.robot_positions[i] = new_pos

            # Update carried package position
            if self.robot_carrying[i] >= 0:
                self.packages[self.robot_carrying[i]] = new_pos.copy()

    def _try_pickup_or_deliver(self, robot_id: int):
        """Try to pickup or deliver package"""
        pos = self.robot_positions[robot_id]

        # Already carrying - check delivery
        if self.robot_carrying[robot_id] >= 0:
            pkg_idx = self.robot_carrying[robot_id]
            dest = self.package_destinations[pkg_idx]
            if np.linalg.norm(pos - dest) < 3.0:
                self.package_delivered[pkg_idx] = True
                self.robot_carrying[robot_id] = -1
                self.metrics.packages_delivered += 1
            return

        # Try pickup
        for i in range(len(self.packages)):
            if self.package_picked[i] or self.package_delivered[i]:
                continue
            if np.linalg.norm(pos - self.packages[i]) < 2.0:
                self.package_picked[i] = True
                self.robot_carrying[robot_id] = i
                break

    def _handle_packages(self):
        """Handle package logic"""
        # Check for simultaneous deliveries
        delivering_count = 0
        for i in range(self.num_robots):
            if self.robot_carrying[i] >= 0:
                pkg_idx = self.robot_carrying[i]
                dest = self.package_destinations[pkg_idx]
                if np.linalg.norm(self.robot_positions[i] - dest) < 5.0:
                    delivering_count += 1

        if delivering_count > 1:
            self.metrics.simultaneous_deliveries += 1

    def _check_collisions(self) -> int:
        """Check robot-robot collisions"""
        collision_count = 0
        min_dist = 2 * self.robot_radius

        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = np.linalg.norm(self.robot_positions[i] - self.robot_positions[j])

                if dist < min_dist:
                    collision_count += 1

                    # Push apart
                    direction = self.robot_positions[j] - self.robot_positions[i]
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    overlap = min_dist - dist
                    self.robot_positions[i] -= direction * overlap * 0.5
                    self.robot_positions[j] += direction * overlap * 0.5

                elif dist < min_dist * 2:
                    self.metrics.near_misses += 1

        return collision_count

    def _calculate_rewards(self, collision_count: int) -> Dict[int, float]:
        """Calculate rewards for all robots"""
        rewards = {}

        for i in range(self.num_robots):
            reward = 0.0

            # Delivery reward
            if self.robot_carrying[i] >= 0:
                pkg_idx = self.robot_carrying[i]
                if self.package_delivered[pkg_idx]:
                    reward += 20.0

            # Progress toward package/destination
            if self.robot_carrying[i] >= 0:
                # Progress to destination
                pkg_idx = self.robot_carrying[i]
                dest = self.package_destinations[pkg_idx]
                dist = np.linalg.norm(self.robot_positions[i] - dest)
                reward += 0.1 * (1.0 - dist / np.linalg.norm(self.grid_size))
            else:
                # Progress to nearest package
                for pkg_idx in range(len(self.packages)):
                    if not self.package_picked[pkg_idx] and not self.package_delivered[pkg_idx]:
                        dist = np.linalg.norm(self.robot_positions[i] - self.packages[pkg_idx])
                        if dist < 5.0:
                            reward += 0.05 * (5.0 - dist) / 5.0
                        break

            # Collision penalty (shared among colliders)
            if collision_count > 0:
                reward -= 5.0 * collision_count / self.num_robots

            # Small step penalty
            reward -= 0.01

            rewards[i] = reward

        return rewards


# =============================================================================
# PPO Trainer
# =============================================================================

class MERAPPO:
    """PPO trainer with MERA encoder"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Environment
        self.env = SimpleMultiRobotWarehouse(config)

        # Network
        self.network = PPOActorCritic(
            self.env.obs_dim,
            self.env.action_dim,
            config
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Observation history buffer (for MERA temporal processing)
        self.obs_history = {
            i: deque(maxlen=config.observation_history)
            for i in range(config.num_robots)
        }

        # Training state
        self.global_step = 0
        self.episode_count = 0

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.coordination_history = []
        self.phi_q_history = []

        # Φ_Q correlation tracking
        self.phi_q_vs_coordination = []  # List of (phi_q, synergy_score) tuples

        # Output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _reset_obs_history(self, observations: Dict[int, np.ndarray]):
        """Reset observation history with initial observations"""
        for robot_id, obs in observations.items():
            self.obs_history[robot_id].clear()
            for _ in range(self.config.observation_history):
                self.obs_history[robot_id].append(obs)

    def _get_obs_tensor(self, robot_id: int) -> torch.Tensor:
        """Get observation history as tensor"""
        obs_list = list(self.obs_history[robot_id])
        obs_array = np.stack(obs_list, axis=0)  # (history, obs_dim)
        return torch.from_numpy(obs_array).float().unsqueeze(0).to(self.device)

    def _get_all_obs_tensors(self) -> torch.Tensor:
        """Get batched observation tensors for all robots"""
        tensors = [self._get_obs_tensor(i) for i in range(self.config.num_robots)]
        return torch.cat(tensors, dim=0)  # (num_robots, history, obs_dim)

    def collect_rollout(self) -> Tuple[Dict[int, List[Transition]], Dict]:
        """Collect rollout data"""
        observations = self.env.reset()
        self._reset_obs_history(observations)

        robot_transitions = {i: [] for i in range(self.config.num_robots)}
        episode_phi_q = []

        for step in range(self.config.steps_per_epoch // self.config.num_robots):
            # Get actions for all robots
            obs_batch = self._get_all_obs_tensors()

            with torch.no_grad():
                actions, values, log_probs, phi_q = self.network.get_action(obs_batch)

            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            episode_phi_q.append(phi_q)

            # Execute actions
            actions_dict = {i: actions_np[i] for i in range(self.config.num_robots)}
            next_obs, rewards, dones, info = self.env.step(actions_dict)

            # Store transitions
            for i in range(self.config.num_robots):
                robot_transitions[i].append(Transition(
                    obs=np.stack(list(self.obs_history[i]), axis=0),
                    action=actions_np[i],
                    reward=rewards[i],
                    done=dones[i],
                    value=values_np[i],
                    log_prob=log_probs_np[i],
                    phi_q=phi_q,
                ))

                # Update observation history
                self.obs_history[i].append(next_obs[i])

            self.global_step += self.config.num_robots

            if dones['__all__']:
                self.episode_count += 1

                # Track metrics
                metrics = info['metrics']
                coord_metrics = CoordinationMetrics(**metrics)
                self.coordination_history.append(metrics)
                self.phi_q_history.append(np.mean(episode_phi_q))

                # Track Φ_Q vs coordination correlation
                self.phi_q_vs_coordination.append((
                    np.mean(episode_phi_q),
                    coord_metrics.synergy_score
                ))

                total_reward = sum(sum(t.reward for t in robot_transitions[i])
                                   for i in range(self.config.num_robots))
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(step + 1)

                # Reset
                observations = self.env.reset()
                self._reset_obs_history(observations)
                episode_phi_q = []

        return robot_transitions, info

    def compute_returns(self, transitions: List[Transition]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages"""
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]

        returns = []
        advantages = []
        gae = 0.0

        for t in reversed(range(len(transitions))):
            if t == len(transitions) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae

            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, robot_transitions: Dict[int, List[Transition]]):
        """PPO update"""
        # Flatten transitions from all robots
        all_transitions = []
        for i in range(self.config.num_robots):
            all_transitions.extend(robot_transitions[i])

        # Prepare data
        obs = torch.tensor(np.stack([t.obs for t in all_transitions]),
                          dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([t.action for t in all_transitions]),
                              dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in all_transitions],
                                     dtype=torch.float32, device=self.device)

        # Compute returns for each robot separately
        all_returns = []
        all_advantages = []
        for i in range(self.config.num_robots):
            returns, advantages = self.compute_returns(robot_transitions[i])
            all_returns.append(returns)
            all_advantages.append(advantages)

        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)

        # PPO update epochs
        batch_size = len(all_transitions) // self.config.num_minibatches

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(len(all_transitions))

            for start in range(0, len(all_transitions), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                values, log_probs, entropy, aux = self.network.evaluate_actions(
                    batch_obs, batch_actions
                )

                # PPO loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                    1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # MERA constraint loss
                if self.config.encoder_type == "mera":
                    mera_loss = self.network.encoder.mera.get_total_loss(aux)

                    # Φ_Q intrinsic motivation bonus
                    phi_q_bonus = -self.config.phi_q_intrinsic_weight * aux['phi_q'].mean()
                else:
                    mera_loss = 0.0
                    phi_q_bonus = 0.0

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                    + mera_loss
                    + phi_q_bonus
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

    def compute_phi_q_correlation(self) -> Dict:
        """Compute correlation between Φ_Q and coordination metrics"""
        if len(self.phi_q_vs_coordination) < 10:
            return {'correlation': 0.0, 'num_samples': len(self.phi_q_vs_coordination)}

        phi_q_values = [x[0] for x in self.phi_q_vs_coordination]
        synergy_values = [x[1] for x in self.phi_q_vs_coordination]

        # Pearson correlation
        phi_q_mean = np.mean(phi_q_values)
        synergy_mean = np.mean(synergy_values)

        numerator = sum((p - phi_q_mean) * (s - synergy_mean)
                       for p, s in self.phi_q_vs_coordination)
        denominator = (
            np.sqrt(sum((p - phi_q_mean) ** 2 for p in phi_q_values)) *
            np.sqrt(sum((s - synergy_mean) ** 2 for s in synergy_values))
        )

        correlation = numerator / (denominator + 1e-8)

        return {
            'correlation': float(correlation),
            'phi_q_mean': float(phi_q_mean),
            'synergy_mean': float(synergy_mean),
            'num_samples': len(self.phi_q_vs_coordination),
        }

    def train(self):
        """Main training loop"""
        print("=" * 70)
        print(f"MERA-PPO Warehouse Training")
        print("=" * 70)
        print(f"Encoder: {self.config.encoder_type}")
        print(f"Robots: {self.config.num_robots}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Device: {self.device}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # Collect rollout
            robot_transitions, info = self.collect_rollout()

            # Update policy
            self.update(robot_transitions)

            epoch_time = time.time() - epoch_start

            # Logging
            if epoch % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
                avg_phi_q = np.mean(self.phi_q_history[-10:]) if self.phi_q_history else 0

                # Coordination metrics
                if self.coordination_history:
                    recent = self.coordination_history[-10:]
                    avg_collisions = np.mean([m['total_collisions'] for m in recent])
                    avg_delivered = np.mean([m['packages_delivered'] for m in recent])
                    avg_synergy = np.mean([
                        CoordinationMetrics(**m).synergy_score for m in recent
                    ])
                else:
                    avg_collisions = 0
                    avg_delivered = 0
                    avg_synergy = 0

                # Φ_Q correlation
                correlation = self.compute_phi_q_correlation()

                print(f"\nEpoch {epoch}/{self.config.num_epochs} ({epoch_time:.1f}s)")
                print(f"  Reward:      {avg_reward:.2f}")
                print(f"  Episode Len: {avg_length:.0f}")
                print(f"  Φ_Q:         {avg_phi_q:.4f}")
                print(f"  Collisions:  {avg_collisions:.1f}")
                print(f"  Delivered:   {avg_delivered:.1f}")
                print(f"  Synergy:     {avg_synergy:.3f}")
                print(f"  Φ_Q↔Synergy: r={correlation['correlation']:.3f}")

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print(f"Training Complete! Total time: {total_time/60:.1f} minutes")
        print("=" * 70)

        # Final analysis
        self.final_analysis()

        return self.get_results()

    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'coordination_history': self.coordination_history,
            'phi_q_history': self.phi_q_history,
            'phi_q_vs_coordination': self.phi_q_vs_coordination,
            'config': asdict(self.config),
        }

        path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")

    def final_analysis(self):
        """Final analysis of Φ_Q vs coordination correlation"""
        print("\n" + "=" * 70)
        print("Φ_Q vs Coordination Analysis")
        print("=" * 70)

        correlation = self.compute_phi_q_correlation()

        print(f"\nCorrelation Analysis (n={correlation['num_samples']}):")
        print(f"  Φ_Q mean:     {correlation['phi_q_mean']:.4f}")
        print(f"  Synergy mean: {correlation['synergy_mean']:.3f}")
        print(f"  Pearson r:    {correlation['correlation']:.3f}")

        if correlation['correlation'] > 0.3:
            print("\n  → POSITIVE correlation: Higher Φ_Q associates with better coordination!")
        elif correlation['correlation'] < -0.3:
            print("\n  → NEGATIVE correlation: Higher Φ_Q associates with worse coordination.")
        else:
            print("\n  → WEAK correlation: Φ_Q doesn't strongly predict coordination.")

        # Save analysis
        analysis = {
            'correlation': correlation,
            'final_metrics': {
                'avg_reward': float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0,
                'avg_synergy': float(np.mean([
                    CoordinationMetrics(**m).synergy_score
                    for m in self.coordination_history[-50:]
                ])) if self.coordination_history else 0,
                'total_episodes': len(self.episode_rewards),
            }
        }

        path = Path(self.config.output_dir) / "phi_q_analysis.json"
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {path}")

    def get_results(self) -> Dict:
        """Get training results"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'coordination_history': self.coordination_history,
            'phi_q_history': self.phi_q_history,
            'phi_q_correlation': self.compute_phi_q_correlation(),
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MERA-PPO Warehouse Training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_robots', type=int, default=4)
    parser.add_argument('--encoder', type=str, default='mera',
                       choices=['mera', 'mlp'])
    parser.add_argument('--steps_per_epoch', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./mera_ppo_results')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with 5 epochs')

    args = parser.parse_args()

    # Config
    config = TrainingConfig(
        num_epochs=5 if args.quick_test else args.epochs,
        num_robots=args.num_robots,
        encoder_type=args.encoder,
        steps_per_epoch=512 if args.quick_test else args.steps_per_epoch,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Train
    trainer = MERAPPO(config)
    results = trainer.train()

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Final Reward: {np.mean(results['episode_rewards'][-10:]):.2f}")
    print(f"Φ_Q↔Synergy Correlation: {results['phi_q_correlation']['correlation']:.3f}")


if __name__ == "__main__":
    main()
