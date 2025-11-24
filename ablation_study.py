"""
Ablation Study Experiment
Test the contribution of each QREA component to overall performance
"""

import torch
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from copy import deepcopy

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.run_experiment import QREASimulation
from utils.logger import create_logger
from utils.visualization import save_figure


class AblationConfig:
    """Configuration for ablation studies"""
    
    @staticmethod
    def no_intrinsic_motivation(config: Dict) -> Dict:
        """Disable intrinsic motivation"""
        config = deepcopy(config)
        config['learning']['intrinsic_rewards'] = {
            'novelty_weight': 0.0,
            'competence_weight': 0.0,
            'empowerment_weight': 0.0
        }
        return config
    
    @staticmethod
    def no_uprt(config: Dict) -> Dict:
        """Disable UPRT field dynamics"""
        config = deepcopy(config)
        config['uprt']['enabled'] = False
        return config
    
    @staticmethod
    def no_communication(config: Dict) -> Dict:
        """Disable emergent communication"""
        config = deepcopy(config)
        config['communication']['enabled'] = False
        return config
    
    @staticmethod
    def no_evolution(config: Dict) -> Dict:
        """Disable evolutionary component"""
        config = deepcopy(config)
        config['evolution']['enabled'] = False
        config['evolution']['population_size'] = 1
        return config
    
    @staticmethod
    def no_world_model(config: Dict) -> Dict:
        """Disable world model (use reactive policy)"""
        config = deepcopy(config)
        config['agent']['use_world_model'] = False
        return config
    
    @staticmethod
    def no_hgt(config: Dict) -> Dict:
        """Disable horizontal gene transfer"""
        config = deepcopy(config)
        config['evolution']['hgt_enabled'] = False
        return config
    
    @staticmethod
    def only_novelty(config: Dict) -> Dict:
        """Use only novelty intrinsic reward"""
        config = deepcopy(config)
        config['learning']['intrinsic_rewards'] = {
            'novelty_weight': 1.0,
            'competence_weight': 0.0,
            'empowerment_weight': 0.0
        }
        return config
    
    @staticmethod
    def only_competence(config: Dict) -> Dict:
        """Use only competence intrinsic reward"""
        config = deepcopy(config)
        config['learning']['intrinsic_rewards'] = {
            'novelty_weight': 0.0,
            'competence_weight': 1.0,
            'empowerment_weight': 0.0
        }
        return config
    
    @staticmethod
    def only_empowerment(config: Dict) -> Dict:
        """Use only empowerment intrinsic reward"""
        config = deepcopy(config)
        config['learning']['intrinsic_rewards'] = {
            'novelty_weight': 0.0,
            'competence_weight': 0.0,
            'empowerment_weight': 1.0
        }
        return config


def run_ablation_study(base_config_path: str, ablation_name: str,
                      n_trials: int = 3) -> Dict:
    """Run single ablation experiment"""
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Apply ablation
    ablation_func = getattr(AblationConfig, ablation_name)
    ablated_config = ablation_func(base_config)
    
    print(f"Running ablation: {ablation_name} ({n_trials} trials)...")
    
    results = []
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        
        # Set different seed for each trial
        ablated_config['simulation']['seed'] = base_config['simulation']['seed'] + trial
        
        # Save temporary config
        temp_config_path = f"/tmp/ablation_{ablation_name}_trial{trial}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(ablated_config, f)
        
        # Run simulation
        sim = QREASimulation(temp_config_path)
        metrics = sim.run()
        
        if hasattr(metrics, 'to_dict'):
            results.append(metrics.to_dict())
        else:
            results.append(metrics)
    
    # Aggregate results
    aggregated = {}
    if results:
        keys = results[0].keys()
        for key in keys:
            values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    return aggregated


def run_ablation_suite(config_path: str, output_dir: str = "results/ablation_study",
                      n_trials: int = 3):
    """Run complete ablation study suite"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(
        "ablation_study",
        str(output_dir / "logs"),
        use_tensorboard=True,
        use_wandb=False
    )
    
    logger.info("Starting ablation study")
    logger.info(f"Number of trials per ablation: {n_trials}")
    
    # Define ablations to test
    ablations = [
        'no_intrinsic_motivation',
        'no_uprt',
        'no_communication',
        'no_evolution',
        'no_world_model',
        'no_hgt',
        'only_novelty',
        'only_competence',
        'only_empowerment'
    ]
    
    # Run full QREA as baseline
    logger.info("Running full QREA (baseline)...")
    full_results = []
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    for trial in range(n_trials):
        print(f"  Full QREA trial {trial + 1}/{n_trials}")
        base_config['simulation']['seed'] = base_config['simulation']['seed'] + trial
        
        temp_config = f"/tmp/full_qrea_trial{trial}.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(base_config, f)
        
        sim = QREASimulation(temp_config)
        metrics = sim.run()
        
        if hasattr(metrics, 'to_dict'):
            full_results.append(metrics.to_dict())
        else:
            full_results.append(metrics)
    
    # Aggregate full QREA results
    full_aggregated = {}
    if full_results:
        keys = full_results[0].keys()
        for key in keys:
            values = [r[key] for r in full_results if isinstance(r.get(key), (int, float))]
            if values:
                full_aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    
    # Run ablations
    ablation_results = {}
    for ablation_name in ablations:
        logger.info(f"Running ablation: {ablation_name}")
        results = run_ablation_study(config_path, ablation_name, n_trials)
        ablation_results[ablation_name] = results
    
    # Analyze results
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*80)
    
    logger.info("\nFull QREA Performance:")
    for metric, stats in full_aggregated.items():
        logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    logger.info("\nAblation Performance (relative to full QREA):")
    
    # Calculate relative performance
    comparison = {}
    for ablation_name, results in ablation_results.items():
        logger.info(f"\n{ablation_name}:")
        comparison[ablation_name] = {}
        
        for metric in full_aggregated.keys():
            if metric in results:
                full_mean = full_aggregated[metric]['mean']
                ablated_mean = results[metric]['mean']
                
                if full_mean != 0:
                    relative_change = ((ablated_mean - full_mean) / full_mean) * 100
                    comparison[ablation_name][metric] = relative_change
                    
                    logger.info(f"  {metric}: {ablated_mean:.4f} ({relative_change:+.2f}%)")
                else:
                    logger.info(f"  {metric}: {ablated_mean:.4f}")
    
    # Visualize results
    logger.info("\nGenerating visualizations...")
    
    # Create heatmap of relative performance
    metrics_list = list(full_aggregated.keys())[:10]  # Top 10 metrics
    ablations_list = list(ablation_results.keys())
    
    heatmap_data = np.zeros((len(ablations_list), len(metrics_list)))
    for i, ablation in enumerate(ablations_list):
        for j, metric in enumerate(metrics_list):
            if metric in comparison[ablation]:
                heatmap_data[i, j] = comparison[ablation][metric]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    ax.set_xticks(np.arange(len(metrics_list)))
    ax.set_yticks(np.arange(len(ablations_list)))
    ax.set_xticklabels(metrics_list, rotation=45, ha='right')
    ax.set_yticklabels(ablations_list)
    
    # Add text annotations
    for i in range(len(ablations_list)):
        for j in range(len(metrics_list)):
            text = ax.text(j, i, f"{heatmap_data[i, j]:.1f}%",
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Ablation Study: Relative Performance Change (%)", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Performance Change (%)')
    plt.tight_layout()
    
    save_figure(fig, output_dir / "ablation_heatmap.png")
    
    # Save detailed results
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'full_qrea': full_aggregated,
            'ablations': ablation_results,
            'relative_performance': comparison
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.finish()
    
    return full_aggregated, ablation_results, comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ablation Study Experiment")
    parser.add_argument("--config", type=str, default="../config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results/ablation_study",
                       help="Output directory")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials per ablation")
    
    args = parser.parse_args()
    
    run_ablation_suite(args.config, args.output, args.trials)
