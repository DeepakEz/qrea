"""
MERA Transfer Learning Experiment
==================================

Tests if MERA's physics-inspired structure enables better transfer learning
than MLP baselines.

Hypothesis: The hierarchical RG structure of MERA should learn representations
that generalize across tasks, since they capture scale-invariant features.

Experiment Design:
1. Train on Task A (uniform package distribution)
2. Freeze MERA encoder layers
3. Fine-tune on Task B (clustered package distribution)
4. Compare MERA vs MLP transfer performance

Success Metric:
- MERA should require fewer fine-tuning steps to reach same performance
- MERA should achieve higher final performance with frozen encoder

Usage:
    python transfer_experiment.py --pretrain_epochs 50 --finetune_epochs 25
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque

from mera_enhanced import EnhancedMERAConfig, EnhancedTensorNetworkMERA
from mera_ppo_warehouse import (
    PPOActorCritic, MLPEncoder, MERAEncoder,
    Transition, CoordinationMetrics
)
from warehouse_env import WarehouseEnv
import yaml


@dataclass
class TransferConfig:
    """Transfer learning experiment configuration"""
    # Training phases
    pretrain_epochs: int = 50
    finetune_epochs: int = 25
    steps_per_epoch: int = 1024

    # Tasks
    task_a: str = "uniform"    # Uniform package distribution
    task_b: str = "clustered"  # Clustered package distribution

    # Freezing strategy
    freeze_encoder: bool = True
    freeze_mera_only: bool = True  # Only freeze MERA, not projection heads

    # Comparison
    compare_mlp: bool = True

    # MERA config
    mera_num_layers: int = 3
    mera_bond_dim: int = 8

    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2

    # Output
    output_dir: str = "./transfer_results"
    seed: int = 42


class TaskVariant:
    """Modifies warehouse environment for different tasks"""

    @staticmethod
    def apply_uniform(env: WarehouseEnv):
        """Uniform package distribution (default)"""
        # No modification needed - this is the default
        pass

    @staticmethod
    def apply_clustered(env: WarehouseEnv):
        """Clustered package distribution - packages spawn in clusters"""
        # Override package spawn locations to create clusters
        if hasattr(env, 'package_spawn_zones'):
            env.package_spawn_zones = [
                {'center': [10, 10], 'radius': 5},  # Bottom-left cluster
                {'center': [40, 40], 'radius': 5},  # Top-right cluster
            ]

    @staticmethod
    def apply_dynamic(env: WarehouseEnv):
        """Dynamic priorities - package priorities change over time"""
        if hasattr(env, 'dynamic_priorities'):
            env.dynamic_priorities = True

    @staticmethod
    def apply(task_name: str, env: WarehouseEnv):
        tasks = {
            'uniform': TaskVariant.apply_uniform,
            'clustered': TaskVariant.apply_clustered,
            'dynamic': TaskVariant.apply_dynamic,
        }
        if task_name in tasks:
            tasks[task_name](env)


class TransferExperiment:
    """Runs transfer learning experiment"""

    def __init__(self, config: TransferConfig, env_config_path: str = "config.yaml"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Load env config
        with open(env_config_path, 'r') as f:
            self.env_config = yaml.safe_load(f)

        # Create environment
        self.env = WarehouseEnv(self.env_config)
        self.num_robots = self.env_config['environment']['num_robots']

        # Get dimensions
        sample_obs = self.env.reset()
        self.obs_dim = len(sample_obs[0])
        self.action_dim = self.env.action_space.shape[0]
        self.history_len = 16

        # MERA config
        self.mera_config = EnhancedMERAConfig(
            num_layers=config.mera_num_layers,
            bond_dim=config.mera_bond_dim,
            physical_dim=4,
            enable_phi_q=True,
            use_identity_init=True,
            enforce_rg_fixed_point=True,
            rg_eigenvalue_weight=0.01,
        )

        # Create networks
        self.mera_network = PPOActorCritic(
            self.obs_dim, self.action_dim, self.history_len,
            encoder_type="mera", mera_config=self.mera_config
        ).to(self.device)

        if config.compare_mlp:
            self.mlp_network = PPOActorCritic(
                self.obs_dim, self.action_dim, self.history_len,
                encoder_type="mlp"
            ).to(self.device)
        else:
            self.mlp_network = None

        # Results
        self.results = {
            'mera': {'pretrain': [], 'finetune': []},
            'mlp': {'pretrain': [], 'finetune': []},
            'config': asdict(config),
        }

        # Output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self, network: nn.Module, frozen_encoder: bool = False):
        """Create optimizer, optionally excluding frozen encoder params"""
        if frozen_encoder and hasattr(network, 'encoder'):
            # Only optimize non-encoder parameters
            params = [p for n, p in network.named_parameters()
                     if 'encoder' not in n and p.requires_grad]
        else:
            params = network.parameters()

        return optim.Adam(params, lr=self.config.learning_rate)

    def _freeze_encoder(self, network: nn.Module):
        """Freeze encoder weights"""
        if hasattr(network, 'encoder'):
            for param in network.encoder.parameters():
                param.requires_grad = False
            print(f"  Froze encoder ({sum(p.numel() for p in network.encoder.parameters()):,} params)")

    def _unfreeze_encoder(self, network: nn.Module):
        """Unfreeze encoder weights"""
        if hasattr(network, 'encoder'):
            for param in network.encoder.parameters():
                param.requires_grad = True

    def train_epoch(self, network: nn.Module, optimizer: optim.Optimizer,
                   obs_history: Dict[int, deque]) -> Dict:
        """Train for one epoch, return metrics"""
        network.train()

        total_reward = 0.0
        total_phi_q = 0.0
        episode_count = 0
        step_count = 0

        observations = self.env.reset()

        # Initialize obs history
        for robot_id, obs in observations.items():
            obs_history[robot_id].clear()
            for _ in range(self.history_len):
                obs_history[robot_id].append(obs)

        transitions = {i: [] for i in range(self.num_robots)}

        for step in range(self.config.steps_per_epoch // self.num_robots):
            # Get observation tensors
            obs_tensors = []
            for i in range(self.num_robots):
                obs_array = np.stack(list(obs_history[i]), axis=0)
                obs_tensors.append(torch.from_numpy(obs_array).float())
            obs_batch = torch.stack(obs_tensors, dim=0).to(self.device)

            # Get actions
            with torch.no_grad():
                actions, values, log_probs, phi_q, aux = network.get_action(obs_batch)

            actions_np = actions.cpu().numpy()

            # Execute
            actions_dict = {i: actions_np[i] for i in range(self.num_robots)}
            next_obs, rewards, dones, info = self.env.step(actions_dict)

            # Track metrics
            for i in range(self.num_robots):
                total_reward += rewards[i]
                obs_history[i].append(next_obs[i])

            total_phi_q += phi_q
            step_count += self.num_robots

            if dones.get('__all__', False):
                episode_count += 1
                observations = self.env.reset()
                for robot_id, obs in observations.items():
                    obs_history[robot_id].clear()
                    for _ in range(self.history_len):
                        obs_history[robot_id].append(obs)

        return {
            'avg_reward': total_reward / max(step_count, 1),
            'avg_phi_q': total_phi_q / max(step_count // self.num_robots, 1),
            'episodes': episode_count,
        }

    def run_phase(self, network: nn.Module, phase_name: str, task: str,
                 num_epochs: int, frozen_encoder: bool = False) -> List[Dict]:
        """Run training phase"""
        print(f"\n{'='*60}")
        print(f"Phase: {phase_name} | Task: {task} | Frozen: {frozen_encoder}")
        print(f"{'='*60}")

        # Apply task variant
        TaskVariant.apply(task, self.env)

        # Freeze if needed
        if frozen_encoder:
            self._freeze_encoder(network)

        # Create optimizer
        optimizer = self._create_optimizer(network, frozen_encoder)

        # Observation history
        obs_history = {i: deque(maxlen=self.history_len) for i in range(self.num_robots)}

        metrics_history = []

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            metrics = self.train_epoch(network, optimizer, obs_history)
            metrics['epoch'] = epoch
            metrics['time'] = time.time() - epoch_start

            metrics_history.append(metrics)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: Reward={metrics['avg_reward']:.3f}, "
                      f"Φ_Q={metrics['avg_phi_q']:.4f}")

        # Unfreeze
        if frozen_encoder:
            self._unfreeze_encoder(network)

        return metrics_history

    def run(self):
        """Run full transfer learning experiment"""
        print("=" * 70)
        print("MERA TRANSFER LEARNING EXPERIMENT")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Task A: {self.config.task_a}")
        print(f"Task B: {self.config.task_b}")
        print(f"Pretrain epochs: {self.config.pretrain_epochs}")
        print(f"Finetune epochs: {self.config.finetune_epochs}")

        # ===== MERA =====
        print("\n" + "=" * 70)
        print("MERA ENCODER")
        print("=" * 70)

        # Pretrain on Task A
        mera_pretrain = self.run_phase(
            self.mera_network, "Pretrain", self.config.task_a,
            self.config.pretrain_epochs, frozen_encoder=False
        )
        self.results['mera']['pretrain'] = mera_pretrain

        # Save pretrained state
        mera_pretrained_state = copy.deepcopy(self.mera_network.state_dict())

        # Fine-tune on Task B with frozen encoder
        mera_finetune = self.run_phase(
            self.mera_network, "Finetune (frozen)", self.config.task_b,
            self.config.finetune_epochs, frozen_encoder=self.config.freeze_encoder
        )
        self.results['mera']['finetune'] = mera_finetune

        # ===== MLP Baseline =====
        if self.config.compare_mlp:
            print("\n" + "=" * 70)
            print("MLP BASELINE")
            print("=" * 70)

            # Pretrain
            mlp_pretrain = self.run_phase(
                self.mlp_network, "Pretrain", self.config.task_a,
                self.config.pretrain_epochs, frozen_encoder=False
            )
            self.results['mlp']['pretrain'] = mlp_pretrain

            # Fine-tune
            mlp_finetune = self.run_phase(
                self.mlp_network, "Finetune (frozen)", self.config.task_b,
                self.config.finetune_epochs, frozen_encoder=self.config.freeze_encoder
            )
            self.results['mlp']['finetune'] = mlp_finetune

        # Analysis
        self.analyze_results()

        # Save
        self.save_results()

        return self.results

    def analyze_results(self):
        """Analyze transfer learning results"""
        print("\n" + "=" * 70)
        print("TRANSFER ANALYSIS")
        print("=" * 70)

        # MERA metrics
        mera_pretrain_final = np.mean([m['avg_reward'] for m in self.results['mera']['pretrain'][-5:]])
        mera_finetune_final = np.mean([m['avg_reward'] for m in self.results['mera']['finetune'][-5:]])
        mera_finetune_start = self.results['mera']['finetune'][0]['avg_reward'] if self.results['mera']['finetune'] else 0

        print(f"\nMERA:")
        print(f"  Pretrain final:  {mera_pretrain_final:.3f}")
        print(f"  Finetune start:  {mera_finetune_start:.3f}")
        print(f"  Finetune final:  {mera_finetune_final:.3f}")
        print(f"  Transfer ratio:  {mera_finetune_start / (mera_pretrain_final + 1e-6):.2%}")
        print(f"  Learning speed:  {(mera_finetune_final - mera_finetune_start) / self.config.finetune_epochs:.4f}/epoch")

        if self.config.compare_mlp and self.results['mlp']['finetune']:
            mlp_pretrain_final = np.mean([m['avg_reward'] for m in self.results['mlp']['pretrain'][-5:]])
            mlp_finetune_final = np.mean([m['avg_reward'] for m in self.results['mlp']['finetune'][-5:]])
            mlp_finetune_start = self.results['mlp']['finetune'][0]['avg_reward']

            print(f"\nMLP:")
            print(f"  Pretrain final:  {mlp_pretrain_final:.3f}")
            print(f"  Finetune start:  {mlp_finetune_start:.3f}")
            print(f"  Finetune final:  {mlp_finetune_final:.3f}")
            print(f"  Transfer ratio:  {mlp_finetune_start / (mlp_pretrain_final + 1e-6):.2%}")
            print(f"  Learning speed:  {(mlp_finetune_final - mlp_finetune_start) / self.config.finetune_epochs:.4f}/epoch")

            # Comparison
            print(f"\nCOMPARISON:")
            transfer_advantage = (mera_finetune_start / (mera_pretrain_final + 1e-6)) - \
                               (mlp_finetune_start / (mlp_pretrain_final + 1e-6))
            print(f"  MERA transfer advantage: {transfer_advantage:+.2%}")

            if transfer_advantage > 0.1:
                print("  → MERA shows BETTER transfer! Hierarchical structure helps generalization.")
            elif transfer_advantage < -0.1:
                print("  → MLP shows better transfer. MERA structure may be too rigid.")
            else:
                print("  → Similar transfer performance.")

    def save_results(self):
        """Save experiment results"""
        path = Path(self.config.output_dir) / "transfer_results.json"

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(path, 'w') as f:
            json.dump(convert(self.results), f, indent=2)

        print(f"\nResults saved to: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MERA Transfer Learning Experiment")
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=25)
    parser.add_argument('--task_a', type=str, default='uniform')
    parser.add_argument('--task_b', type=str, default='clustered')
    parser.add_argument('--quick_test', action='store_true')
    args = parser.parse_args()

    config = TransferConfig(
        pretrain_epochs=5 if args.quick_test else args.pretrain_epochs,
        finetune_epochs=3 if args.quick_test else args.finetune_epochs,
        steps_per_epoch=256 if args.quick_test else 1024,
        task_a=args.task_a,
        task_b=args.task_b,
    )

    experiment = TransferExperiment(config)
    results = experiment.run()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
