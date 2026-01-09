#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner with Statistical Analysis
======================================================

Runs experiments across multiple seeds for statistical rigor.
Reports mean +/- std for all metrics.

Usage:
    python run_experiments.py --encoder mera --seeds 5 --epochs 100
    python run_experiments.py --compare all --seeds 5  # Compare all encoders
    python run_experiments.py --scaling --seeds 3       # Scaling study (4,8,16,32 robots)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml


@dataclass
class ExperimentResult:
    """Results from a single experiment run (paper metrics only)"""
    seed: int
    encoder: str
    num_robots: int
    epochs: int

    # Performance metrics
    final_reward: float
    avg_reward: float

    # Coordination metrics (paper core)
    packages_delivered: int
    collision_rate: float  # collisions per episode step
    throughput: float

    # Φ_Q metric (probe - not optimized)
    final_phi_q: float = 0.0

    # Training time
    training_time_seconds: float = 0.0


@dataclass
class AggregatedResults:
    """Aggregated results across seeds (paper metrics)"""
    encoder: str
    num_robots: int
    num_seeds: int

    # Mean +/- std for metrics
    final_reward_mean: float
    final_reward_std: float

    # Coordination metrics
    packages_delivered_mean: float
    packages_delivered_std: float
    collision_rate_mean: float
    collision_rate_std: float
    throughput_mean: float
    throughput_std: float

    # Φ_Q (probe only)
    phi_q_mean: float = 0.0
    phi_q_std: float = 0.0

    training_time_mean: float = 0.0
    training_time_std: float = 0.0


def run_single_experiment(
    encoder: str,
    seed: int,
    epochs: int,
    num_robots: int = 8,
    config_path: str = "config.yaml",
    sparse_rewards: bool = False
) -> ExperimentResult:
    """Run a single experiment with given parameters"""

    # Create temporary config with modified robot count
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['environment']['num_robots'] = num_robots

    # Use unique output dir for this run
    output_dir = f"./results_{encoder}_robots{num_robots}_seed{seed}"

    # Write temporary config using tempfile to avoid collisions when running in parallel
    temp_fd, temp_config = tempfile.mkstemp(suffix='.yaml', prefix=f'config_{encoder}_seed{seed}_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(config, f)
        # Set seed via environment variable
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)

        # Run the training
        start_time = time.time()

        cmd = [
            sys.executable, "mera_ppo_warehouse.py",
            "--config", temp_config,
            "--encoder", encoder,
            "--epochs", str(epochs),
        ]

        if sparse_rewards:
            cmd.append("--sparse_rewards")

        # Capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600 * 4  # 4 hour timeout
        )

        training_time = time.time() - start_time

        # Parse results from output or result files
        results_file = Path(output_dir) / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
        else:
            # Parse from stdout
            data = parse_stdout(result.stdout)

        # Compute collision_rate from collisions and episode length
        collisions = data.get('collisions', 0)
        episode_steps = data.get('episode_steps', epochs * 500)  # estimate
        collision_rate = collisions / max(episode_steps, 1)

        return ExperimentResult(
            seed=seed,
            encoder=encoder,
            num_robots=num_robots,
            epochs=epochs,
            final_reward=data.get('final_reward', 0),
            avg_reward=data.get('avg_reward', 0),
            packages_delivered=data.get('packages_delivered', 0),
            collision_rate=collision_rate,
            throughput=data.get('throughput', 0),
            final_phi_q=data.get('phi_q', 0),
            training_time_seconds=training_time
        )

    finally:
        # Cleanup temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)


def parse_stdout(stdout: str) -> Dict:
    """Parse metrics from training stdout"""
    data = {
        'final_reward': 0,
        'avg_reward': 0,
        'max_reward': 0,
        'packages_delivered': 0,
        'collisions': 0,
        'throughput': 0,
        'phi_q': 0,
    }

    lines = stdout.split('\n')
    for line in lines:
        if 'Reward:' in line:
            try:
                # Parse "Reward: X.XX" pattern
                parts = line.split('Reward:')
                if len(parts) > 1:
                    val = float(parts[1].split()[0].strip(','))
                    data['final_reward'] = val
            except (ValueError, IndexError):
                pass
        if 'Delivered:' in line:
            try:
                parts = line.split('Delivered:')
                if len(parts) > 1:
                    val = int(parts[1].split()[0].strip(','))
                    data['packages_delivered'] = val
            except (ValueError, IndexError):
                pass
        if 'Throughput:' in line:
            try:
                parts = line.split('Throughput:')
                if len(parts) > 1:
                    val = float(parts[1].split()[0].strip(','))
                    data['throughput'] = val
            except (ValueError, IndexError):
                pass
        if 'Phi_Q:' in line or 'phi_q:' in line.lower():
            try:
                parts = line.lower().split('phi_q:')
                if len(parts) > 1:
                    val = float(parts[1].split()[0].strip(','))
                    data['phi_q'] = val
            except (ValueError, IndexError):
                pass

    return data


def aggregate_results(results: List[ExperimentResult]) -> AggregatedResults:
    """Aggregate results across seeds with mean +/- std"""
    if not results:
        raise ValueError("No results to aggregate")

    encoder = results[0].encoder
    num_robots = results[0].num_robots

    final_rewards = [r.final_reward for r in results]
    packages = [r.packages_delivered for r in results]
    collision_rates = [r.collision_rate for r in results]
    throughputs = [r.throughput for r in results]
    phi_qs = [r.final_phi_q for r in results]
    times = [r.training_time_seconds for r in results]

    return AggregatedResults(
        encoder=encoder,
        num_robots=num_robots,
        num_seeds=len(results),
        final_reward_mean=np.mean(final_rewards),
        final_reward_std=np.std(final_rewards),
        packages_delivered_mean=np.mean(packages),
        packages_delivered_std=np.std(packages),
        collision_rate_mean=np.mean(collision_rates),
        collision_rate_std=np.std(collision_rates),
        throughput_mean=np.mean(throughputs),
        throughput_std=np.std(throughputs),
        phi_q_mean=np.mean(phi_qs),
        phi_q_std=np.std(phi_qs),
        training_time_mean=np.mean(times),
        training_time_std=np.std(times),
    )


def print_results_table(all_results: Dict[str, AggregatedResults]):
    """Print nicely formatted results table (paper metrics)"""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS (mean +/- std across seeds)")
    print("=" * 100)

    # Header
    print(f"{'Encoder':<12} {'Robots':<7} {'Reward':<18} {'Delivered':<12} {'Coll.Rate':<12} {'Throughput':<12} {'Phi_Q':<12}")
    print("-" * 100)

    for key, agg in sorted(all_results.items()):
        reward_str = f"{agg.final_reward_mean:.1f}±{agg.final_reward_std:.1f}"
        delivered_str = f"{agg.packages_delivered_mean:.1f}±{agg.packages_delivered_std:.1f}"
        coll_str = f"{agg.collision_rate_mean:.4f}±{agg.collision_rate_std:.4f}"
        throughput_str = f"{agg.throughput_mean:.1f}±{agg.throughput_std:.1f}"
        phi_q_str = f"{agg.phi_q_mean:.4f}±{agg.phi_q_std:.4f}"

        print(f"{agg.encoder:<12} {agg.num_robots:<7} {reward_str:<18} {delivered_str:<12} {coll_str:<12} {throughput_str:<12} {phi_q_str:<12}")

    print("=" * 100)


def run_encoder_comparison(
    encoders: List[str],
    seeds: int,
    epochs: int,
    num_robots: int = 8,
    parallel: bool = True,
    sparse_rewards: bool = False
) -> Dict[str, AggregatedResults]:
    """Compare multiple encoders across seeds"""

    all_results = {}

    for encoder in encoders:
        print(f"\n>>> Running {encoder} encoder ({seeds} seeds, {epochs} epochs)")

        results = []
        if parallel and seeds > 1:
            with ProcessPoolExecutor(max_workers=min(seeds, 4)) as executor:
                futures = {
                    executor.submit(run_single_experiment, encoder, seed, epochs, num_robots, "config.yaml", sparse_rewards): seed
                    for seed in range(seeds)
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"  Seed {seed}: reward={result.final_reward:.2f}, delivered={result.packages_delivered}")
                    except Exception as e:
                        print(f"  Seed {seed} FAILED: {e}")
        else:
            for seed in range(seeds):
                try:
                    result = run_single_experiment(encoder, seed, epochs, num_robots, "config.yaml", sparse_rewards)
                    results.append(result)
                    print(f"  Seed {seed}: reward={result.final_reward:.2f}, delivered={result.packages_delivered}")
                except Exception as e:
                    print(f"  Seed {seed} FAILED: {e}")

        if results:
            all_results[encoder] = aggregate_results(results)

    return all_results


def run_scaling_study(
    encoder: str,
    robot_counts: List[int],
    seeds: int,
    epochs: int,
    sparse_rewards: bool = False
) -> Dict[str, AggregatedResults]:
    """Run scaling study across different robot counts"""

    all_results = {}

    for num_robots in robot_counts:
        print(f"\n>>> Scaling study: {num_robots} robots ({seeds} seeds)")

        results = []
        for seed in range(seeds):
            try:
                result = run_single_experiment(encoder, seed, epochs, num_robots, "config.yaml", sparse_rewards)
                results.append(result)
                print(f"  Seed {seed}: reward={result.final_reward:.2f}, throughput={result.throughput:.2f}")
            except Exception as e:
                print(f"  Seed {seed} FAILED: {e}")

        if results:
            all_results[f"{encoder}_{num_robots}robots"] = aggregate_results(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument('--encoder', type=str, default='mera',
                        choices=['mera', 'mera_uprt', 'gru', 'transformer', 'mlp'],
                        help='Encoder to test')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per run')
    parser.add_argument('--robots', type=int, default=8,
                        help='Number of robots (default: 8)')

    # Special modes
    parser.add_argument('--compare', type=str, default=None,
                        choices=['all', 'baselines', 'mera_variants'],
                        help='Compare multiple encoders')
    parser.add_argument('--scaling', action='store_true',
                        help='Run scaling study (4, 8, 16, 32 robots)')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Run seeds in parallel')
    parser.add_argument('--sparse_rewards', action='store_true',
                        help='Use sparse rewards (stops reward hacking)')
    parser.add_argument('--output', type=str, default='experiment_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    print("=" * 80)
    print("QREA Multi-Seed Experiment Runner")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")

    results = {}

    if args.compare:
        # Compare multiple encoders
        if args.compare == 'all':
            encoders = ['mera', 'mera_uprt', 'gru', 'transformer', 'mlp']
        elif args.compare == 'baselines':
            encoders = ['gru', 'transformer', 'mlp']
        elif args.compare == 'mera_variants':
            encoders = ['mera', 'mera_uprt']

        print(f"Comparing encoders: {encoders}")
        results = run_encoder_comparison(
            encoders, args.seeds, args.epochs, args.robots, args.parallel, args.sparse_rewards
        )

    elif args.scaling:
        # Scaling study
        robot_counts = [4, 8, 16, 32]
        print(f"Scaling study with {args.encoder}: {robot_counts} robots")
        results = run_scaling_study(
            args.encoder, robot_counts, args.seeds, args.epochs, args.sparse_rewards
        )

    else:
        # Single encoder experiment
        print(f"Running {args.encoder} with {args.seeds} seeds")
        exp_results = []
        for seed in range(args.seeds):
            try:
                result = run_single_experiment(
                    args.encoder, seed, args.epochs, args.robots, "config.yaml", args.sparse_rewards
                )
                exp_results.append(result)
                print(f"Seed {seed}: reward={result.final_reward:.2f}")
            except Exception as e:
                print(f"Seed {seed} FAILED: {e}")

        if exp_results:
            results[args.encoder] = aggregate_results(exp_results)

    # Print results table
    if results:
        print_results_table(results)

        # Save to file
        output_data = {k: asdict(v) for k, v in results.items()}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
