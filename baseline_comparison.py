"""
Baseline Comparison Experiment
Compare QREA against traditional warehouse robot control methods
"""

import torch
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.run_experiment import QREASimulation
from utils.metrics import PerformanceMetrics
from utils.logger import create_logger
from utils.visualization import MetricsVisualizer, save_figure


class BaselineController:
    """Traditional baseline controllers for comparison"""
    
    class GreedyNearest:
        """Greedy nearest-package controller"""
        
        def __init__(self, n_robots: int):
            self.n_robots = n_robots
        
        def select_action(self, state: Dict) -> np.ndarray:
            """Select nearest unassigned package"""
            actions = []
            for robot_id in range(self.n_robots):
                robot_pos = state['robots'][robot_id][:2]
                
                # Find nearest package
                min_dist = float('inf')
                best_pkg = None
                
                for pkg in state['packages']:
                    if not pkg['assigned']:
                        dist = np.linalg.norm(robot_pos - pkg['pickup_pos'])
                        if dist < min_dist:
                            min_dist = dist
                            best_pkg = pkg
                
                if best_pkg is not None:
                    # Move toward package
                    direction = best_pkg['pickup_pos'] - robot_pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    actions.append(direction)
                else:
                    actions.append(np.zeros(2))
            
            return np.array(actions)
    
    class PriorityBased:
        """Priority-based controller (handle high-priority first)"""
        
        def __init__(self, n_robots: int):
            self.n_robots = n_robots
        
        def select_action(self, state: Dict) -> np.ndarray:
            """Select highest priority package"""
            actions = []
            for robot_id in range(self.n_robots):
                robot_pos = state['robots'][robot_id][:2]
                
                # Find highest priority package
                best_priority = float('inf')
                best_pkg = None
                
                for pkg in state['packages']:
                    if not pkg['assigned']:
                        if pkg['priority'] < best_priority:
                            best_priority = pkg['priority']
                            best_pkg = pkg
                        elif pkg['priority'] == best_priority:
                            # Tie-break by distance
                            if best_pkg is None:
                                best_pkg = pkg
                            else:
                                dist = np.linalg.norm(robot_pos - pkg['pickup_pos'])
                                best_dist = np.linalg.norm(robot_pos - best_pkg['pickup_pos'])
                                if dist < best_dist:
                                    best_pkg = pkg
                
                if best_pkg is not None:
                    direction = best_pkg['pickup_pos'] - robot_pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    actions.append(direction)
                else:
                    actions.append(np.zeros(2))
            
            return np.array(actions)
    
    class AuctionBased:
        """Market-based auction controller"""
        
        def __init__(self, n_robots: int):
            self.n_robots = n_robots
            self.assignments = {}
        
        def select_action(self, state: Dict) -> np.ndarray:
            """Auction-based task allocation"""
            # Simple auction: assign packages to minimize total cost
            robots = state['robots']
            packages = [p for p in state['packages'] if not p['assigned']]
            
            if not packages:
                return np.zeros((self.n_robots, 2))
            
            # Compute cost matrix
            cost_matrix = np.zeros((self.n_robots, len(packages)))
            for i, robot in enumerate(robots):
                for j, pkg in enumerate(packages):
                    dist = np.linalg.norm(robot[:2] - pkg['pickup_pos'])
                    urgency = pkg['priority']  # Higher priority = lower cost
                    cost_matrix[i, j] = dist * urgency
            
            # Greedy assignment
            assignments = {}
            used_packages = set()
            
            for _ in range(min(self.n_robots, len(packages))):
                # Find minimum cost assignment
                min_cost = float('inf')
                best_robot = None
                best_pkg_idx = None
                
                for i in range(self.n_robots):
                    if i in assignments:
                        continue
                    for j in range(len(packages)):
                        if j in used_packages:
                            continue
                        if cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_robot = i
                            best_pkg_idx = j
                
                if best_robot is not None:
                    assignments[best_robot] = packages[best_pkg_idx]
                    used_packages.add(best_pkg_idx)
            
            # Generate actions
            actions = []
            for i in range(self.n_robots):
                if i in assignments:
                    pkg = assignments[i]
                    direction = pkg['pickup_pos'] - robots[i][:2]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    actions.append(direction)
                else:
                    actions.append(np.zeros(2))
            
            return np.array(actions)


def run_baseline_experiment(config_path: str, baseline_type: str,
                           n_episodes: int = 100) -> PerformanceMetrics:
    """Run baseline experiment"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create baseline controller
    n_robots = config['environment']['n_robots']
    
    if baseline_type == 'greedy':
        controller = BaselineController.GreedyNearest(n_robots)
    elif baseline_type == 'priority':
        controller = BaselineController.PriorityBased(n_robots)
    elif baseline_type == 'auction':
        controller = BaselineController.AuctionBased(n_robots)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    print(f"Running {baseline_type} baseline for {n_episodes} episodes...")
    
    # TODO: Implement baseline evaluation
    # This would require creating a simple evaluation loop
    
    metrics = PerformanceMetrics()
    
    # Placeholder results
    # In real implementation, run episodes and collect metrics
    
    return metrics


def run_comparison_suite(config_path: str, output_dir: str = "results/baseline_comparison"):
    """Run complete baseline comparison suite"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(
        "baseline_comparison",
        str(output_dir / "logs"),
        use_tensorboard=True,
        use_wandb=False
    )
    
    logger.info("Starting baseline comparison experiment")
    
    # Run QREA
    logger.info("Running QREA...")
    qrea_sim = QREASimulation(config_path)
    qrea_metrics = qrea_sim.run()
    
    # Run baselines
    baseline_types = ['greedy', 'priority', 'auction']
    baseline_results = {}
    
    for baseline in baseline_types:
        logger.info(f"Running {baseline} baseline...")
        metrics = run_baseline_experiment(config_path, baseline, n_episodes=100)
        baseline_results[baseline] = metrics.to_dict()
    
    # Comparison
    qrea_results = qrea_metrics.to_dict() if hasattr(qrea_metrics, 'to_dict') else qrea_metrics
    
    logger.info("Baseline Comparison Results:")
    logger.info(f"QREA: {qrea_results}")
    for name, results in baseline_results.items():
        logger.info(f"{name}: {results}")
    
    # Visualize comparison
    viz = MetricsVisualizer()
    
    # Create comparison plot for each baseline
    for baseline_name, baseline_metrics in baseline_results.items():
        fig = viz.plot_performance_comparison(
            baseline_metrics,
            qrea_results,
            title=f"QREA vs {baseline_name.capitalize()}"
        )
        save_figure(fig, output_dir / f"comparison_{baseline_name}.png")
    
    # Save results
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'qrea': qrea_results,
            'baselines': baseline_results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.finish()
    
    return qrea_results, baseline_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Comparison Experiment")
    parser.add_argument("--config", type=str, default="../config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results/baseline_comparison",
                       help="Output directory")
    
    args = parser.parse_args()
    
    run_comparison_suite(args.config, args.output)
