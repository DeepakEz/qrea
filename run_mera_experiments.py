"""
MERA Tensor Network Experiments for RL
=======================================

This script runs benchmark experiments for the MERA-enhanced RL system:

1. Validation: Verify MERA components work correctly
2. Ablation: Test contribution of each component
3. Comparison: Compare against baseline encoders
4. Analysis: Visualize Φ_Q, RG flow, and intrinsic rewards

Usage:
    python run_mera_experiments.py --experiment validation
    python run_mera_experiments.py --experiment ablation
    python run_mera_experiments.py --experiment benchmark
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

# Import MERA modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mera_enhanced import (
    EnhancedMERAConfig,
    EnhancedTensorNetworkMERA,
    MERAWorldModelEncoder,
    PhiQComputer,
)
from mera_rl_integration import (
    MERATrainingConfig,
    MERAEnhancedWorldModel,
    MERAEnhancedTrainer,
    get_ablation_configs
)


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    name: str
    config: dict
    metrics: dict
    duration_seconds: float
    timestamp: str


class ExperimentRunner:
    """Runs MERA experiments and collects results"""

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results: List[ExperimentResult] = []

    def run_validation_experiments(self) -> Dict[str, bool]:
        """
        Validate all MERA components work correctly.

        Tests:
        1. Forward pass shapes
        2. Gradient flow
        3. Constraint losses
        4. Φ_Q computation
        5. Intrinsic rewards
        """
        print("\n" + "=" * 70)
        print("VALIDATION EXPERIMENTS")
        print("=" * 70)

        results = {}
        start_time = time.time()

        # Test 1: Basic forward pass
        print("\n1. Testing basic forward pass...")
        try:
            config = EnhancedMERAConfig(num_layers=3, bond_dim=8, physical_dim=4)
            mera = EnhancedTensorNetworkMERA(config).to(self.device)

            batch = torch.randn(4, 50, 64).to(self.device)
            latent, aux = mera(batch)

            assert latent.shape[0] == 4, "Batch size mismatch"
            assert not torch.isnan(latent).any(), "NaN in output"
            results['forward_pass'] = True
            print("   ✓ Forward pass OK")
        except Exception as e:
            results['forward_pass'] = False
            print(f"   ✗ Forward pass FAILED: {e}")

        # Test 2: Gradient flow
        print("\n2. Testing gradient flow...")
        try:
            loss = latent.mean() + mera.get_total_loss(aux)
            loss.backward()

            has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in mera.parameters())
            assert has_gradients, "No gradients computed"
            results['gradient_flow'] = True
            print("   ✓ Gradient flow OK")
        except Exception as e:
            results['gradient_flow'] = False
            print(f"   ✗ Gradient flow FAILED: {e}")

        # Test 3: Constraint losses
        print("\n3. Testing constraint losses...")
        try:
            constraint_loss = aux['constraint_loss']
            rg_loss = aux.get('rg_eigenvalue_loss', torch.tensor(0.0))
            assert constraint_loss.item() >= 0, "Negative constraint loss"
            assert not torch.isnan(constraint_loss), "NaN in constraint loss"
            results['constraint_loss'] = True
            print(f"   ✓ Constraint loss OK: {constraint_loss.item():.6f}")
            print(f"   ✓ RG eigenvalue loss: {rg_loss.item():.6f}")
        except Exception as e:
            results['constraint_loss'] = False
            print(f"   ✗ Constraint loss FAILED: {e}")

        # Test 4: Φ_Q computation
        print("\n4. Testing Φ_Q computation...")
        try:
            phi_q = aux['phi_q']
            assert phi_q is not None, "Φ_Q is None"
            assert (phi_q >= 0).all(), "Negative Φ_Q values"
            assert not torch.isnan(phi_q).any(), "NaN in Φ_Q"
            results['phi_q_computation'] = True
            print(f"   ✓ Φ_Q computation OK: mean={phi_q.mean().item():.4f}")
        except Exception as e:
            results['phi_q_computation'] = False
            print(f"   ✗ Φ_Q computation FAILED: {e}")

        # Test 5: Intrinsic rewards
        print("\n5. Testing intrinsic rewards...")
        try:
            intrinsic = aux['intrinsic_rewards']
            assert 'phi_q_reward' in intrinsic, "Missing phi_q_reward"
            assert 'total_intrinsic' in intrinsic, "Missing total_intrinsic"
            results['intrinsic_rewards'] = True
            print(f"   ✓ Intrinsic rewards OK: {intrinsic['total_intrinsic'].mean().item():.4f}")
        except Exception as e:
            results['intrinsic_rewards'] = False
            print(f"   ✗ Intrinsic rewards FAILED: {e}")

        # Test 6: World model integration
        print("\n6. Testing world model integration...")
        try:
            train_config = MERATrainingConfig()
            world_model = MERAEnhancedWorldModel(64, 3, train_config).to(self.device)

            obs_seq = torch.randn(4, 50, 64).to(self.device)
            mera_latent, mera_aux = world_model.encode_sequence(obs_seq)

            assert mera_latent.shape == (4, 256), f"Wrong latent shape: {mera_latent.shape}"
            results['world_model'] = True
            print("   ✓ World model integration OK")
        except Exception as e:
            results['world_model'] = False
            print(f"   ✗ World model integration FAILED: {e}")

        # Test 7: Training step
        print("\n7. Testing training step...")
        try:
            policy = nn.Linear(world_model.stochastic_dim + world_model.deterministic_dim, 3).to(self.device)
            value = nn.Linear(world_model.stochastic_dim + world_model.deterministic_dim, 1).to(self.device)

            trainer = MERAEnhancedTrainer(world_model, policy, value, train_config, self.device)

            # Add fake experience
            for _ in range(100):
                trainer.add_experience(
                    np.random.randn(64), np.random.randn(3),
                    np.random.randn(), np.random.randn(64), False
                )

            metrics = trainer.train_step()
            assert metrics is not None, "Training returned None"
            results['training_step'] = True
            print(f"   ✓ Training step OK: loss={metrics['total_loss']:.4f}")
        except Exception as e:
            results['training_step'] = False
            print(f"   ✗ Training step FAILED: {e}")

        duration = time.time() - start_time
        print(f"\nValidation completed in {duration:.2f}s")
        print(f"Passed: {sum(results.values())}/{len(results)}")

        # Save result
        self.results.append(ExperimentResult(
            name="validation",
            config={},
            metrics=results,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        ))

        return results

    def run_ablation_experiments(self, num_steps: int = 100) -> Dict[str, Dict]:
        """
        Run ablation study on MERA components.

        Tests contribution of:
        1. Φ_Q intrinsic motivation
        2. Entanglement exploration
        3. Constraint regularization
        4. Scale consistency loss
        """
        print("\n" + "=" * 70)
        print("ABLATION EXPERIMENTS")
        print("=" * 70)

        configs = get_ablation_configs()
        results = {}

        for name, config in configs.items():
            print(f"\n--- Running: {name} ---")
            start_time = time.time()

            try:
                # Create models
                world_model = MERAEnhancedWorldModel(64, 3, config).to(self.device)
                policy = nn.Linear(world_model.stochastic_dim + world_model.deterministic_dim, 3).to(self.device)
                value = nn.Linear(world_model.stochastic_dim + world_model.deterministic_dim, 1).to(self.device)

                trainer = MERAEnhancedTrainer(world_model, policy, value, config, self.device)

                # Fill buffer
                for _ in range(200):
                    trainer.add_experience(
                        np.random.randn(64), np.random.randn(3),
                        np.random.randn(), np.random.randn(64), False
                    )

                # Training loop
                losses = []
                phi_q_values = []
                intrinsic_values = []

                for step in range(num_steps):
                    metrics = trainer.train_step()
                    if metrics:
                        losses.append(metrics['total_loss'])
                        phi_q_values.append(metrics.get('phi_q_mean', 0))
                        intrinsic_values.append(metrics.get('intrinsic_reward_mean', 0))

                duration = time.time() - start_time

                results[name] = {
                    'final_loss': np.mean(losses[-10:]) if losses else float('inf'),
                    'loss_std': np.std(losses[-10:]) if losses else 0,
                    'avg_phi_q': np.mean(phi_q_values) if phi_q_values else 0,
                    'avg_intrinsic': np.mean(intrinsic_values) if intrinsic_values else 0,
                    'duration': duration,
                    'config': asdict(config) if hasattr(config, '__dataclass_fields__') else {},
                }

                print(f"   Loss: {results[name]['final_loss']:.4f} ± {results[name]['loss_std']:.4f}")
                print(f"   Φ_Q: {results[name]['avg_phi_q']:.4f}")
                print(f"   Duration: {duration:.2f}s")

            except Exception as e:
                print(f"   FAILED: {e}")
                results[name] = {'error': str(e)}

        # Save results
        self.results.append(ExperimentResult(
            name="ablation",
            config={'num_steps': num_steps},
            metrics=results,
            duration_seconds=sum(r.get('duration', 0) for r in results.values()),
            timestamp=datetime.now().isoformat()
        ))

        return results

    def run_phi_q_analysis(self) -> Dict:
        """
        Analyze Φ_Q behavior across different input patterns.

        Tests how Φ_Q responds to:
        1. Random noise
        2. Structured sequences
        3. Repeating patterns
        4. Correlated data
        """
        print("\n" + "=" * 70)
        print("Φ_Q ANALYSIS")
        print("=" * 70)

        config = EnhancedMERAConfig(num_layers=3, bond_dim=8, physical_dim=4, enable_phi_q=True)
        mera = EnhancedTensorNetworkMERA(config).to(self.device)
        mera.eval()

        results = {}

        # Test different input patterns
        patterns = {
            'random': lambda: torch.randn(8, 50, 64),
            'structured': lambda: torch.sin(torch.linspace(0, 10, 50)).unsqueeze(0).unsqueeze(-1).expand(8, 50, 64),
            'repeating': lambda: torch.randn(8, 1, 64).repeat(1, 50, 1),
            'correlated': lambda: torch.cumsum(torch.randn(8, 50, 64) * 0.1, dim=1),
        }

        for name, gen_fn in patterns.items():
            print(f"\n--- Pattern: {name} ---")

            phi_q_values = []
            entanglement_values = []

            for _ in range(10):
                data = gen_fn().to(self.device)
                with torch.no_grad():
                    _, aux = mera(data)

                if aux['phi_q'] is not None:
                    phi_q_values.append(aux['phi_q'].mean().item())

                intrinsic = aux['intrinsic_rewards']
                entanglement_values.append(intrinsic['entanglement_raw'].mean().item())

            results[name] = {
                'phi_q_mean': np.mean(phi_q_values),
                'phi_q_std': np.std(phi_q_values),
                'entanglement_mean': np.mean(entanglement_values),
                'entanglement_std': np.std(entanglement_values),
            }

            print(f"   Φ_Q: {results[name]['phi_q_mean']:.4f} ± {results[name]['phi_q_std']:.4f}")
            print(f"   Entanglement: {results[name]['entanglement_mean']:.4f} ± {results[name]['entanglement_std']:.4f}")

        # Save results
        self.results.append(ExperimentResult(
            name="phi_q_analysis",
            config={},
            metrics=results,
            duration_seconds=0,
            timestamp=datetime.now().isoformat()
        ))

        return results

    def run_rg_flow_analysis(self) -> Dict:
        """
        Analyze RG flow eigenvalues for scale consistency.

        Tests:
        1. Eigenvalue distribution across layers
        2. Fixed point detection (λ ≈ 1)
        3. Scaling behavior
        4. RG eigenvalue loss (new)
        """
        print("\n" + "=" * 70)
        print("RG FLOW ANALYSIS")
        print("=" * 70)

        # Test with identity-based initialization and RG regularization
        config = EnhancedMERAConfig(
            num_layers=4,
            bond_dim=8,
            physical_dim=4,
            use_identity_init=True,
            enforce_rg_fixed_point=True,
            rg_eigenvalue_weight=0.05,
        )
        mera = EnhancedTensorNetworkMERA(config).to(self.device)
        mera.eval()

        all_eigenvalues = []
        rg_losses = []

        print("\n   Testing with identity initialization + RG regularization...")

        for _ in range(50):
            data = torch.randn(8, 50, 64).to(self.device)
            with torch.no_grad():
                _, aux = mera(data)

            if aux['rg_eigenvalues']:
                all_eigenvalues.extend(aux['rg_eigenvalues'])

            if 'rg_eigenvalue_loss' in aux:
                rg_losses.append(aux['rg_eigenvalue_loss'].item())

        if all_eigenvalues:
            eigenvalues = np.array(all_eigenvalues)

            results = {
                'mean': float(np.mean(eigenvalues)),
                'std': float(np.std(eigenvalues)),
                'min': float(np.min(eigenvalues)),
                'max': float(np.max(eigenvalues)),
                'near_fixed_point': float(np.mean(np.abs(eigenvalues - 1.0) < 0.1)),
                'rg_loss_mean': float(np.mean(rg_losses)) if rg_losses else 0.0,
                'histogram': np.histogram(eigenvalues, bins=20)[0].tolist(),
                'identity_init': True,
                'rg_regularization': True,
            }

            print(f"   Mean eigenvalue: {results['mean']:.4f} ± {results['std']:.4f}")
            print(f"   Range: [{results['min']:.4f}, {results['max']:.4f}]")
            print(f"   Near fixed point (|λ-1| < 0.1): {results['near_fixed_point']*100:.1f}%")
            print(f"   RG eigenvalue loss: {results['rg_loss_mean']:.6f}")

            # Compare with old initialization
            print("\n   Comparing with random initialization (no RG loss)...")
            config_old = EnhancedMERAConfig(
                num_layers=4,
                bond_dim=8,
                physical_dim=4,
                use_identity_init=False,
                enforce_rg_fixed_point=False,
            )
            mera_old = EnhancedTensorNetworkMERA(config_old).to(self.device)
            mera_old.eval()

            old_eigenvalues = []
            for _ in range(50):
                data = torch.randn(8, 50, 64).to(self.device)
                with torch.no_grad():
                    _, aux = mera_old(data)
                if aux['rg_eigenvalues']:
                    old_eigenvalues.extend(aux['rg_eigenvalues'])

            if old_eigenvalues:
                old_evs = np.array(old_eigenvalues)
                old_near_fp = np.mean(np.abs(old_evs - 1.0) < 0.1) * 100
                print(f"   Old init mean: {np.mean(old_evs):.4f}, near FP: {old_near_fp:.1f}%")
                results['old_init_mean'] = float(np.mean(old_evs))
                results['old_init_near_fp'] = float(old_near_fp)
        else:
            results = {'error': 'No eigenvalues computed'}

        self.results.append(ExperimentResult(
            name="rg_flow_analysis",
            config={},
            metrics=results,
            duration_seconds=0,
            timestamp=datetime.now().isoformat()
        ))

        return results

    def save_results(self, filename: Optional[str] = None):
        """Save all experiment results to JSON"""
        if filename is None:
            filename = f"mera_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        data = {
            'experiments': [asdict(r) for r in self.results],
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")

    def generate_report(self) -> str:
        """Generate markdown report of experiments"""
        report = ["# MERA Tensor Network Experiments Report\n"]
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        report.append(f"Device: {self.device}\n\n")

        for result in self.results:
            report.append(f"## {result.name.replace('_', ' ').title()}\n")
            report.append(f"Duration: {result.duration_seconds:.2f}s\n\n")

            if isinstance(result.metrics, dict):
                report.append("### Results\n")
                report.append("| Metric | Value |\n")
                report.append("|--------|-------|\n")
                for key, value in result.metrics.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            report.append(f"| {key}/{k} | {v} |\n")
                    else:
                        report.append(f"| {key} | {value} |\n")
            report.append("\n")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Run MERA Experiments")
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['validation', 'ablation', 'phi_q', 'rg_flow', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--num_steps', type=int, default=500,
                       help='Number of training steps for ablation (default: 500)')
    parser.add_argument('--long_training', action='store_true',
                       help='Use extended training (1000+ steps) for better RG convergence')

    args = parser.parse_args()

    # Use longer training if specified
    if args.long_training:
        args.num_steps = max(args.num_steps, 1000)

    print("=" * 70)
    print("MERA TENSOR NETWORK EXPERIMENTS")
    print("=" * 70)

    runner = ExperimentRunner(args.output_dir)

    if args.experiment in ['validation', 'all']:
        runner.run_validation_experiments()

    if args.experiment in ['ablation', 'all']:
        runner.run_ablation_experiments(args.num_steps)

    if args.experiment in ['phi_q', 'all']:
        runner.run_phi_q_analysis()

    if args.experiment in ['rg_flow', 'all']:
        runner.run_rg_flow_analysis()

    # Save results
    runner.save_results()

    # Generate report
    report = runner.generate_report()
    report_path = runner.output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
