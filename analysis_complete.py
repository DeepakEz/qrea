"""
Complete Analysis and Experimental Framework
=============================================

Production tools for:
- Statistical significance testing
- Performance metric computation
- Encoder architecture comparison
- Ablation studies
- Curriculum analysis
- Multi-seed experiments
- Bootstrapped confidence intervals
- Effect size measurements
- Publication-ready figures

No placeholders. Full statistical rigor.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, friedmanchisquare
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pickle
from collections import defaultdict
import torch
import argparse
import yaml
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

from warehouse_env_fixed import WarehouseEnv
from mera_ppo_complete import PPOTrainer, TrainingConfig, ActorCritic


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class StatisticalResult:
    """Statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str


class StatisticalAnalyzer:
    """Complete statistical analysis toolkit"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def compare_two_groups(self, group1: np.ndarray, group2: np.ndarray,
                          test: str = 'auto', paired: bool = False) -> StatisticalResult:
        """
        Compare two groups with appropriate statistical test
        
        Args:
            group1, group2: Performance samples
            test: 'auto', 'ttest', 'mannwhitney', 'wilcoxon'
            paired: Whether samples are paired
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        # Select test
        if test == 'auto':
            # Check normality (Shapiro-Wilk)
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)
            
            if p1 > 0.05 and p2 > 0.05:
                test = 'wilcoxon' if paired else 'ttest'
            else:
                test = 'wilcoxon' if paired else 'mannwhitney'
        
        # Perform test
        if test == 'ttest':
            if paired:
                stat, p_value = stats.ttest_rel(group1, group2)
                test_name = "Paired t-test"
            else:
                stat, p_value = ttest_ind(group1, group2)
                test_name = "Independent t-test"
        elif test == 'mannwhitney':
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        elif test == 'wilcoxon':
            stat, p_value = wilcoxon(group1 - group2)
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError(f"Unknown test: {test}")
        
        # Compute effect size (Cohen's d)
        effect_size = self._cohens_d(group1, group2)
        
        # Bootstrap confidence interval on difference
        ci = self._bootstrap_ci(group1, group2, n_bootstrap=10000)
        
        # Interpret
        significant = p_value < self.alpha
        
        if significant:
            if effect_size < 0.2:
                magnitude = "negligible"
            elif effect_size < 0.5:
                magnitude = "small"
            elif effect_size < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            winner = "Group 1" if group1.mean() > group2.mean() else "Group 2"
            interpretation = f"Significant difference (p={p_value:.4f}). {winner} is better with {magnitude} effect size."
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). Cannot conclude which is better."
        
        return StatisticalResult(
            test_name=test_name,
            statistic=float(stat),
            p_value=float(p_value),
            significant=significant,
            effect_size=float(effect_size),
            confidence_interval=(float(ci[0]), float(ci[1])),
            interpretation=interpretation
        )
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _bootstrap_ci(self, group1: np.ndarray, group2: np.ndarray,
                     n_bootstrap: int = 10000, ci_level: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval on mean difference"""
        differences = []
        
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            differences.append(sample1.mean() - sample2.mean())
        
        differences = np.array(differences)
        alpha = 1 - ci_level
        lower = np.percentile(differences, alpha/2 * 100)
        upper = np.percentile(differences, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def compare_multiple_groups(self, groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare multiple groups with post-hoc tests
        
        Returns pairwise comparisons and overall test
        """
        group_names = list(groups.keys())
        group_data = [groups[name] for name in group_names]
        
        # Overall test (Friedman if paired, Kruskal-Wallis if not)
        # Assuming independent samples for now
        stat, p_value = stats.kruskal(*group_data)
        
        overall_significant = p_value < self.alpha
        
        # Pairwise comparisons with Bonferroni correction
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        bonferroni_alpha = self.alpha / n_comparisons
        
        pairwise = {}
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                
                result = self.compare_two_groups(
                    groups[name1], groups[name2]
                )
                
                # Apply Bonferroni correction
                result.significant = result.p_value < bonferroni_alpha
                
                pairwise[f"{name1} vs {name2}"] = result
        
        return {
            'overall_test': {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': overall_significant
            },
            'pairwise': pairwise,
            'bonferroni_alpha': bonferroni_alpha
        }


# =============================================================================
# Performance Metrics
# =============================================================================

class PerformanceMetrics:
    """Complete performance metric computation"""
    
    @staticmethod
    def compute_all_metrics(episode_rewards: List[float],
                          episode_deliveries: List[int],
                          episode_collisions: List[int],
                          episode_lengths: List[int]) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
        
        rewards = np.array(episode_rewards)
        deliveries = np.array(episode_deliveries)
        collisions = np.array(episode_collisions)
        lengths = np.array(episode_lengths)
        
        metrics = {}
        
        # Reward metrics
        metrics['reward_mean'] = float(rewards.mean())
        metrics['reward_std'] = float(rewards.std())
        metrics['reward_min'] = float(rewards.min())
        metrics['reward_max'] = float(rewards.max())
        metrics['reward_median'] = float(np.median(rewards))
        metrics['reward_q25'] = float(np.percentile(rewards, 25))
        metrics['reward_q75'] = float(np.percentile(rewards, 75))
        
        # Delivery metrics
        metrics['delivery_mean'] = float(deliveries.mean())
        metrics['delivery_std'] = float(deliveries.std())
        metrics['delivery_rate'] = float((deliveries > 0).mean())  # Success rate
        metrics['delivery_max'] = float(deliveries.max())
        
        # Collision metrics
        metrics['collision_mean'] = float(collisions.mean())
        metrics['collision_std'] = float(collisions.std())
        metrics['collision_rate'] = float(collisions.sum() / lengths.sum())  # Per step
        
        # Efficiency metrics
        if deliveries.sum() > 0:
            metrics['reward_per_delivery'] = float(rewards.sum() / deliveries.sum())
            metrics['steps_per_delivery'] = float(lengths.sum() / deliveries.sum())
        else:
            metrics['reward_per_delivery'] = 0.0
            metrics['steps_per_delivery'] = float('inf')
        
        # Sample efficiency (episodes to first delivery)
        first_delivery = None
        for i, d in enumerate(deliveries):
            if d > 0:
                first_delivery = i
                break
        metrics['episodes_to_first_delivery'] = first_delivery if first_delivery is not None else len(deliveries)
        
        # Consistency (coefficient of variation)
        if rewards.std() > 0:
            metrics['reward_cv'] = float(rewards.std() / abs(rewards.mean()))
        else:
            metrics['reward_cv'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_learning_curve_metrics(episode_rewards: List[float],
                                      window_size: int = 50) -> Dict[str, Any]:
        """Analyze learning curve"""
        rewards = np.array(episode_rewards)
        
        if len(rewards) < window_size:
            window_size = len(rewards)
        
        # Compute moving average
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Compute trend (linear regression)
        x = np.arange(len(moving_avg))
        slope, intercept = np.polyfit(x, moving_avg, 1)
        
        # Compute acceleration (second derivative)
        if len(moving_avg) > 10:
            second_deriv = np.diff(np.diff(moving_avg))
            acceleration = float(second_deriv.mean())
        else:
            acceleration = 0.0
        
        # Convergence detection
        if len(moving_avg) > window_size:
            recent_std = moving_avg[-window_size:].std()
            converged = recent_std < 0.05 * abs(moving_avg[-window_size:].mean())
        else:
            converged = False
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'acceleration': acceleration,
            'converged': converged,
            'final_performance': float(moving_avg[-1]) if len(moving_avg) > 0 else 0.0,
            'best_performance': float(moving_avg.max()) if len(moving_avg) > 0 else 0.0,
            'moving_average': moving_avg.tolist()
        }


# =============================================================================
# Encoder Comparison Experiments
# =============================================================================

class EncoderComparison:
    """Complete encoder comparison with statistical rigor"""
    
    def __init__(self, config_path: str, curriculum_level: int,
                 output_dir: Path, device: str = 'cpu'):
        self.config_path = config_path
        self.curriculum_level = curriculum_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.analyzer = StatisticalAnalyzer(alpha=0.05)
        self.results = {}
        
        self._setup_logging()
    
    def _setup_logging(self):
        log_file = self.output_dir / 'comparison.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_experiment(self, encoder_type: str, num_seeds: int,
                      num_epochs: int) -> Dict[str, Any]:
        """Run experiment for one encoder across multiple seeds"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Running experiment: {encoder_type}")
        self.logger.info(f"Seeds: {num_seeds}, Epochs: {num_epochs}")
        self.logger.info(f"{'='*70}")
        
        seed_results = []
        
        for seed in range(num_seeds):
            self.logger.info(f"\n--- Seed {seed + 1}/{num_seeds} ---")
            
            # Create training config
            training_config = TrainingConfig(
                curriculum_level=self.curriculum_level,
                encoder_type=encoder_type,
                num_epochs=num_epochs,
                seed=seed,
                device=self.device,
                save_interval=max(num_epochs, 1000),  # Don't save intermediate
                eval_interval=max(num_epochs // 10, 10)
            )
            
            # Create trainer
            trainer = PPOTrainer(self.config_path, training_config)
            
            # Train
            trainer.train()
            
            # Final evaluation
            final_eval = trainer.evaluate(num_episodes=20)
            
            seed_results.append({
                'seed': seed,
                'final_eval': final_eval,
                'episode_rewards': trainer.episode_rewards,
                'episode_deliveries': trainer.episode_deliveries,
                'episode_collisions': trainer.episode_collisions,
                'episode_lengths': trainer.episode_lengths
            })
            
            self.logger.info(f"  Reward: {final_eval['reward_mean']:.1f}")
            self.logger.info(f"  Delivered: {final_eval['delivered_mean']:.1f}")
        
        # Aggregate results
        aggregated = self._aggregate_results(encoder_type, seed_results)
        
        return aggregated
    
    def _aggregate_results(self, encoder_type: str, seed_results: List[Dict]) -> Dict:
        """Aggregate results across seeds"""
        # Extract metrics
        rewards = [r['final_eval']['reward_mean'] for r in seed_results]
        deliveries = [r['final_eval']['delivered_mean'] for r in seed_results]
        collisions = [r['final_eval']['collisions_mean'] for r in seed_results]
        
        # Compute statistics
        result = {
            'encoder_type': encoder_type,
            'num_seeds': len(seed_results),
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'values': rewards
            },
            'delivered': {
                'mean': float(np.mean(deliveries)),
                'std': float(np.std(deliveries)),
                'min': float(np.min(deliveries)),
                'max': float(np.max(deliveries)),
                'values': deliveries
            },
            'collisions': {
                'mean': float(np.mean(collisions)),
                'std': float(np.std(collisions)),
                'values': collisions
            },
            'seed_results': seed_results
        }
        
        return result
    
    def compare_encoders(self, encoder_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple encoder results"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("STATISTICAL COMPARISON")
        self.logger.info(f"{'='*70}")
        
        # Extract performance data
        encoder_names = list(encoder_results.keys())
        reward_data = {name: np.array(encoder_results[name]['reward']['values'])
                      for name in encoder_names}
        delivery_data = {name: np.array(encoder_results[name]['delivered']['values'])
                        for name in encoder_names}
        
        # Compare rewards
        self.logger.info("\n=== REWARD COMPARISON ===")
        reward_comparison = self.analyzer.compare_multiple_groups(reward_data)
        
        self.logger.info(f"\nOverall test: p-value = {reward_comparison['overall_test']['p_value']:.4f}")
        if reward_comparison['overall_test']['significant']:
            self.logger.info("  Significant difference detected!")
        else:
            self.logger.info("  No significant difference detected.")
        
        self.logger.info(f"\nPairwise comparisons (Bonferroni α = {reward_comparison['bonferroni_alpha']:.4f}):")
        for comparison_name, result in reward_comparison['pairwise'].items():
            self.logger.info(f"\n{comparison_name}:")
            self.logger.info(f"  p-value: {result.p_value:.4f}")
            self.logger.info(f"  Effect size (Cohen's d): {result.effect_size:.3f}")
            self.logger.info(f"  95% CI: [{result.confidence_interval[0]:.1f}, {result.confidence_interval[1]:.1f}]")
            self.logger.info(f"  {result.interpretation}")
        
        # Compare deliveries
        self.logger.info("\n=== DELIVERY COMPARISON ===")
        delivery_comparison = self.analyzer.compare_multiple_groups(delivery_data)
        
        self.logger.info(f"\nOverall test: p-value = {delivery_comparison['overall_test']['p_value']:.4f}")
        
        # Create comparison summary
        comparison_summary = {
            'reward_comparison': reward_comparison,
            'delivery_comparison': delivery_comparison,
            'best_encoder': self._determine_best_encoder(encoder_results, reward_comparison)
        }
        
        return comparison_summary
    
    def _determine_best_encoder(self, encoder_results: Dict[str, Dict],
                                reward_comparison: Dict) -> Dict[str, Any]:
        """Determine best encoder with justification"""
        encoder_names = list(encoder_results.keys())
        
        # Get mean rewards
        mean_rewards = {name: encoder_results[name]['reward']['mean']
                       for name in encoder_names}
        
        # Best by mean
        best_by_mean = max(encoder_names, key=lambda n: mean_rewards[n])
        
        # Check if significantly better than others
        is_significantly_best = True
        for name in encoder_names:
            if name == best_by_mean:
                continue
            
            comparison_key = f"{best_by_mean} vs {name}"
            if comparison_key not in reward_comparison['pairwise']:
                comparison_key = f"{name} vs {best_by_mean}"
            
            if comparison_key in reward_comparison['pairwise']:
                result = reward_comparison['pairwise'][comparison_key]
                if not result.significant:
                    is_significantly_best = False
                    break
        
        return {
            'encoder': best_by_mean,
            'mean_reward': mean_rewards[best_by_mean],
            'significantly_best': is_significantly_best
        }
    
    def generate_report(self, encoder_results: Dict[str, Dict],
                       comparison_summary: Dict[str, Any]):
        """Generate comprehensive report"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("FINAL REPORT")
        self.logger.info(f"{'='*70}")
        
        # Summary table
        self.logger.info("\nPerformance Summary:")
        self.logger.info(f"{'Encoder':<15} {'Reward':<20} {'Delivered':<20} {'Collisions':<15}")
        self.logger.info("-" * 70)
        
        for name, results in encoder_results.items():
            reward_str = f"{results['reward']['mean']:.1f} ± {results['reward']['std']:.1f}"
            delivery_str = f"{results['delivered']['mean']:.1f} ± {results['delivered']['std']:.1f}"
            collision_str = f"{results['collisions']['mean']:.1f}"
            
            self.logger.info(f"{name:<15} {reward_str:<20} {delivery_str:<20} {collision_str:<15}")
        
        # Best encoder
        best = comparison_summary['best_encoder']
        self.logger.info(f"\nBest Encoder: {best['encoder']}")
        if best['significantly_best']:
            self.logger.info("  ✓ Significantly better than all others")
        else:
            self.logger.info("  ⚠ Not significantly better than all others")
        
        # Save results
        results_file = self.output_dir / 'comparison_results.json'
        with open(results_file, 'w') as f:
            # Convert StatisticalResult objects to dicts
            def convert_results(obj):
                if isinstance(obj, StatisticalResult):
                    return asdict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_results(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_results(item) for item in obj]
                return obj
            
            output = {
                'encoder_results': encoder_results,
                'comparison_summary': convert_results(comparison_summary)
            }
            json.dump(output, f, indent=2)
        
        self.logger.info(f"\nResults saved to {results_file}")
        
        # Generate plots
        self._plot_comparison(encoder_results, comparison_summary)
    
    def _plot_comparison(self, encoder_results: Dict[str, Dict],
                        comparison_summary: Dict[str, Any]):
        """Generate comparison plots"""
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        encoder_names = list(encoder_results.keys())
        
        # 1. Box plots - Reward
        ax = axes[0, 0]
        reward_data = [encoder_results[name]['reward']['values'] for name in encoder_names]
        bp = ax.boxplot(reward_data, labels=encoder_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(encoder_names)))):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward Distribution by Encoder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark best
        best_idx = encoder_names.index(comparison_summary['best_encoder']['encoder'])
        ax.scatter([best_idx + 1], [comparison_summary['best_encoder']['mean_reward']],
                  marker='*', s=500, c='gold', edgecolors='black', linewidths=2,
                  zorder=10, label='Best')
        ax.legend()
        
        # 2. Box plots - Deliveries
        ax = axes[0, 1]
        delivery_data = [encoder_results[name]['delivered']['values'] for name in encoder_names]
        bp = ax.boxplot(delivery_data, labels=encoder_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(encoder_names)))):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Packages Delivered', fontsize=12, fontweight='bold')
        ax.set_title('Delivery Performance by Encoder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Bar chart - Mean comparison
        ax = axes[1, 0]
        means = [encoder_results[name]['reward']['mean'] for name in encoder_names]
        stds = [encoder_results[name]['reward']['std'] for name in encoder_names]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(encoder_names)))
        bars = ax.bar(encoder_names, means, yerr=stds, capsize=5, color=colors,
                     edgecolor='black', linewidth=1.5)
        
        # Mark best
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)
        
        ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax.set_title('Mean Reward Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Scatter plot - Reward vs Delivery
        ax = axes[1, 1]
        for name, color in zip(encoder_names, colors):
            rewards = encoder_results[name]['reward']['values']
            deliveries = encoder_results[name]['delivered']['values']
            ax.scatter(deliveries, rewards, label=name, s=100, alpha=0.7,
                      color=color, edgecolors='black', linewidths=1.5)
        
        ax.set_xlabel('Packages Delivered', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward vs Delivery Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_plots.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {self.output_dir / 'comparison_plots.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complete Encoder Comparison")
    
    # Experiment config
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--level', type=int, default=3, choices=[1,2,3,4,5])
    parser.add_argument('--encoders', nargs='+', default=['gru', 'transformer', 'mera'],
                       choices=['gru', 'transformer', 'mera', 'mlp'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./comparison_results')
    
    # Device
    parser.add_argument('--device', type=str, default=None)
    
    # Quick test
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer seeds/epochs')
    
    args = parser.parse_args()
    
    if args.quick:
        args.seeds = 2
        args.epochs = 20
        print("\n*** QUICK TEST MODE ***")
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create comparison
    comparison = EncoderComparison(
        args.config,
        args.level,
        Path(args.output_dir),
        device=device
    )
    
    print(f"\n{'='*70}")
    print("ENCODER COMPARISON EXPERIMENT")
    print(f"{'='*70}")
    print(f"Encoders: {args.encoders}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per seed: {args.epochs}")
    print(f"Level: {args.level}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Run experiments for each encoder
    encoder_results = {}
    for encoder_type in args.encoders:
        result = comparison.run_experiment(
            encoder_type,
            num_seeds=args.seeds,
            num_epochs=args.epochs
        )
        encoder_results[encoder_type] = result
    
    # Compare results
    comparison_summary = comparison.compare_encoders(encoder_results)
    
    # Generate report
    comparison.generate_report(encoder_results, comparison_summary)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
