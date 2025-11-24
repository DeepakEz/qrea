"""
Visualization Tools for QREA Warehouse System
Includes plotting for metrics, fields, agent behaviors, and evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class WarehouseVisualizer:
    """Visualize warehouse environment and robot behaviors"""
    
    def __init__(self, warehouse_size: Tuple[float, float]):
        self.warehouse_size = warehouse_size
        
    def plot_warehouse_state(self, robots: List[Dict], packages: List[Dict],
                            obstacles: List[Dict], title: str = "Warehouse State"):
        """Plot current warehouse state"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw warehouse boundary
        rect = patches.Rectangle((0, 0), self.warehouse_size[0], self.warehouse_size[1],
                                linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(rect)
        
        # Draw obstacles
        for obs in obstacles:
            obs_rect = patches.Rectangle(
                (obs['x'], obs['y']), obs['width'], obs['height'],
                linewidth=1, edgecolor='gray', facecolor='gray', alpha=0.7
            )
            ax.add_patch(obs_rect)
        
        # Draw packages
        for pkg in packages:
            color = {1: 'red', 2: 'orange', 3: 'yellow'}.get(pkg['priority'], 'blue')
            pkg_circle = patches.Circle(
                (pkg['pickup_x'], pkg['pickup_y']), 0.3,
                facecolor=color, edgecolor='black', alpha=0.6
            )
            ax.add_patch(pkg_circle)
            
            # Draw destination
            dest_circle = patches.Circle(
                (pkg['dropoff_x'], pkg['dropoff_y']), 0.3,
                facecolor='none', edgecolor=color, linestyle='--', linewidth=2
            )
            ax.add_patch(dest_circle)
        
        # Draw robots
        for i, robot in enumerate(robots):
            robot_circle = patches.Circle(
                (robot['x'], robot['y']), 0.5,
                facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.8
            )
            ax.add_patch(robot_circle)
            
            # Draw heading direction
            dx = 0.7 * np.cos(robot['theta'])
            dy = 0.7 * np.sin(robot['theta'])
            ax.arrow(robot['x'], robot['y'], dx, dy,
                    head_width=0.3, head_length=0.2, fc='darkblue', ec='darkblue')
            
            # Label robot
            ax.text(robot['x'], robot['y'] - 0.8, f"R{i}",
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-1, self.warehouse_size[0] + 1)
        ax.set_ylim(-1, self.warehouse_size[1] + 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        
        return fig


class UPRTFieldVisualizer:
    """Visualize UPRT field dynamics"""
    
    def __init__(self, field_shape: Tuple[int, int]):
        self.field_shape = field_shape
    
    def plot_field(self, field_values: np.ndarray, title: str = "UPRT Field"):
        """Plot 2D field heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(field_values.T, origin='lower', cmap='viridis',
                      aspect='auto', interpolation='bilinear')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Intensity')
        
        return fig
    
    def plot_multiple_fields(self, fields_dict: Dict[str, np.ndarray],
                           suptitle: str = "UPRT Fields"):
        """Plot multiple fields in a grid"""
        n_fields = len(fields_dict)
        n_cols = min(3, n_fields)
        n_rows = (n_fields + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for idx, (name, field) in enumerate(fields_dict.items()):
            ax = axes[idx]
            im = ax.imshow(field.T, origin='lower', cmap='viridis',
                          aspect='auto', interpolation='bilinear')
            ax.set_title(name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        # Hide empty subplots
        for idx in range(n_fields, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_field_evolution(self, field_sequence: List[np.ndarray],
                           timestamps: List[float], title: str = "Field Evolution"):
        """Plot field evolution over time"""
        n_snapshots = min(6, len(field_sequence))
        indices = np.linspace(0, len(field_sequence)-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, field_idx in enumerate(indices):
            ax = axes[idx]
            im = ax.imshow(field_sequence[field_idx].T, origin='lower',
                          cmap='viridis', aspect='auto', interpolation='bilinear')
            ax.set_title(f't = {timestamps[field_idx]:.2f}s')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


class MetricsVisualizer:
    """Visualize training and performance metrics"""
    
    @staticmethod
    def plot_learning_curves(metrics_dict: Dict[str, List[float]],
                            title: str = "Learning Curves"):
        """Plot learning curves"""
        n_metrics = len(metrics_dict)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for idx, (name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            steps = np.arange(len(values))
            ax.plot(steps, values, linewidth=2)
            ax.set_title(name)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add smoothed trend
            if len(values) > 10:
                window = min(50, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], smoothed, 'r--', linewidth=2, alpha=0.7,
                       label='Trend')
                ax.legend()
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_performance_comparison(baseline: Dict[str, float],
                                   qrea: Dict[str, float],
                                   title: str = "Performance Comparison"):
        """Compare QREA vs baseline performance"""
        metrics = list(baseline.keys())
        baseline_values = [baseline[k] for k in metrics]
        qrea_values = [qrea[k] for k in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, qrea_values, width, label='QREA', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_intrinsic_rewards(novelty: List[float], competence: List[float],
                              empowerment: List[float],
                              title: str = "Intrinsic Motivation Components"):
        """Plot intrinsic reward components"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        steps = np.arange(len(novelty))
        
        # Novelty
        axes[0].plot(steps, novelty, label='Novelty', color='blue', linewidth=2)
        axes[0].fill_between(steps, novelty, alpha=0.3)
        axes[0].set_ylabel('Novelty')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Competence
        axes[1].plot(steps, competence, label='Competence', color='green', linewidth=2)
        axes[1].fill_between(steps, competence, alpha=0.3)
        axes[1].set_ylabel('Competence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Empowerment
        axes[2].plot(steps, empowerment, label='Empowerment', color='red', linewidth=2)
        axes[2].fill_between(steps, empowerment, alpha=0.3)
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Empowerment')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


class EvolutionVisualizer:
    """Visualize evolutionary progress"""
    
    @staticmethod
    def plot_fitness_evolution(fitness_history: List[Dict[str, float]],
                              title: str = "Fitness Evolution"):
        """Plot fitness evolution over generations"""
        generations = np.arange(len(fitness_history))
        means = [f['mean'] for f in fitness_history]
        maxs = [f['max'] for f in fitness_history]
        mins = [f['min'] for f in fitness_history]
        stds = [f['std'] for f in fitness_history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean with std bands
        ax.plot(generations, means, 'b-', linewidth=2, label='Mean')
        ax.fill_between(generations,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.3, label='±1 std')
        
        # Plot max and min
        ax.plot(generations, maxs, 'g--', linewidth=2, label='Max')
        ax.plot(generations, mins, 'r--', linewidth=2, label='Min')
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Fitness', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_diversity_evolution(diversity_history: List[float],
                                title: str = "Population Diversity"):
        """Plot population diversity over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        generations = np.arange(len(diversity_history))
        ax.plot(generations, diversity_history, 'purple', linewidth=2)
        ax.fill_between(generations, diversity_history, alpha=0.3)
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Diversity', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_hgt_impact(hgt_events: List[Dict],
                       title: str = "Horizontal Gene Transfer Impact"):
        """Plot impact of HGT events"""
        if not hgt_events:
            return None
        
        generations = [e['generation'] for e in hgt_events]
        improvements = [e['improvement'] for e in hgt_events]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(generations, improvements, s=100, alpha=0.6, c='orange')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Fitness Improvement', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


class CommunicationVisualizer:
    """Visualize emergent communication"""
    
    @staticmethod
    def plot_message_frequency(message_counts: Dict[str, int],
                              title: str = "Message Type Frequency"):
        """Plot frequency of different message types"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        messages = list(message_counts.keys())
        counts = list(message_counts.values())
        
        ax.bar(messages, counts, alpha=0.7)
        ax.set_xlabel('Message Type', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_communication_network(adjacency_matrix: np.ndarray,
                                  title: str = "Robot Communication Network"):
        """Plot communication network between robots"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(adjacency_matrix, cmap='YlOrRd', aspect='auto')
        
        n_robots = adjacency_matrix.shape[0]
        ax.set_xticks(np.arange(n_robots))
        ax.set_yticks(np.arange(n_robots))
        ax.set_xticklabels([f'R{i}' for i in range(n_robots)])
        ax.set_yticklabels([f'R{i}' for i in range(n_robots)])
        
        # Add text annotations
        for i in range(n_robots):
            for j in range(n_robots):
                text = ax.text(j, i, f'{adjacency_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Communication Strength')
        
        plt.tight_layout()
        
        return fig


def save_figure(fig, filepath: str, dpi: int = 300):
    """Save figure to file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_dashboard(metrics: Dict, save_path: Optional[str] = None):
    """Create comprehensive metrics dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # TODO: Add dashboard panels based on metrics
    # This is a placeholder for a comprehensive dashboard
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
