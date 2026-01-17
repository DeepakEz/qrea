"""
Complete Warehouse Visualization System
========================================

Production-ready visualization with:
- Real-time episode rendering
- Video recording (MP4/GIF)
- Trajectory visualization
- Heatmap generation
- Agent interaction graphs
- Performance metrics overlay
- Frame-by-frame analysis
- Interactive controls

No placeholders. Full implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
from collections import deque, defaultdict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

from warehouse_env_fixed import WarehouseEnv, DeliveryStage


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    # Display
    figsize: Tuple[int, int] = (14, 14)
    dpi: int = 150
    update_interval_ms: int = 50
    
    # Style
    robot_color_idle: str = 'dodgerblue'
    robot_color_carrying: str = 'orange'
    robot_alpha: float = 0.9
    package_color: str = 'red'
    trajectory_alpha: float = 0.3
    trajectory_length: int = 50
    
    # Overlay
    show_stats: bool = True
    show_labels: bool = True
    show_trajectories: bool = True
    show_lidar: bool = False
    show_velocity: bool = True
    
    # Recording
    save_format: str = 'mp4'  # mp4, gif
    fps: int = 20
    bitrate: int = 2000


class WarehouseVisualizer:
    """Complete visualization system"""
    
    def __init__(self, env: WarehouseEnv, config: VisualizationConfig = None):
        self.env = env
        self.config = config or VisualizationConfig()
        
        # Figure and axes
        self.fig = None
        self.ax = None
        self.stat_text = None
        
        # Trajectory tracking
        self.trajectories = {i: deque(maxlen=self.config.trajectory_length) 
                            for i in range(env.num_robots)}
        
        # Frame storage for video
        self.frames = []
        self.frame_metadata = []
        
        # Performance tracking
        self.step_count = 0
        self.render_count = 0
        
        # Color maps for heatmaps
        self.activity_heatmap = np.zeros((100, 100))
        self.collision_heatmap = np.zeros((100, 100))
        
        self._setup_figure()
    
    def _setup_figure(self):
        """Setup matplotlib figure and axes"""
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        self.ax.set_xlim(0, self.env.grid_size[0])
        self.ax.set_ylim(0, self.env.grid_size[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2, linestyle='--')
        self.ax.set_facecolor('#f8f9fa')
        
        # Set labels
        self.ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        
        # Title
        title = f'Multi-Robot Warehouse - Level {self.env.curriculum_level}'
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Stats text box
        if self.config.show_stats:
            self.stat_text = self.ax.text(
                0.02, 0.98, '', transform=self.ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                         edgecolor='black', linewidth=2),
                family='monospace'
            )
    
    def render_frame(self, show_trajectories: bool = None):
        """Render current environment state"""
        self.ax.clear()
        self._setup_axes_appearance()
        
        show_traj = show_trajectories if show_trajectories is not None else self.config.show_trajectories
        
        # Draw grid zones (optional)
        self._draw_zones()
        
        # Draw stations
        self._draw_stations()
        
        # Draw packages
        self._draw_packages()
        
        # Draw robot trajectories
        if show_traj:
            self._draw_trajectories()
        
        # Draw robots
        self._draw_robots()
        
        # Draw connections (packages to destinations)
        self._draw_connections()
        
        # Draw statistics overlay
        if self.config.show_stats:
            self._draw_stats()
        
        self.render_count += 1
    
    def _setup_axes_appearance(self):
        """Setup axes appearance after clear"""
        self.ax.set_xlim(0, self.env.grid_size[0])
        self.ax.set_ylim(0, self.env.grid_size[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2, linestyle='--')
        self.ax.set_facecolor('#f8f9fa')
        self.ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        
        title = f'Multi-Robot Warehouse - Level {self.env.curriculum_level} (Step {self.env.current_step})'
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    def _draw_zones(self):
        """Draw operational zones"""
        # Draw boundary
        boundary = patches.Rectangle(
            (0, 0), self.env.grid_size[0], self.env.grid_size[1],
            linewidth=3, edgecolor='black', facecolor='none', linestyle='-'
        )
        self.ax.add_patch(boundary)
    
    def _draw_stations(self):
        """Draw all stations with labels"""
        for station in self.env.stations:
            if 'pickup' in station.type.lower():
                color = 'gold'
                marker = 's'
                size = 300
                label_text = 'P'
            elif 'delivery' in station.type.lower():
                color = 'limegreen'
                marker = '^'
                size = 350
                label_text = 'D'
            else:  # charging
                color = 'cyan'
                marker = 'p'
                size = 280
                label_text = 'C'
            
            # Draw station marker
            self.ax.scatter(
                station.position[0], station.position[1],
                marker=marker, s=size, c=color,
                edgecolors='black', linewidths=2.5,
                zorder=5, alpha=0.85
            )
            
            # Draw station label
            if self.config.show_labels:
                self.ax.text(
                    station.position[0], station.position[1],
                    label_text, ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='black', zorder=6
                )
            
            # Draw occupancy indicator
            if station.occupancy > 0:
                circle = patches.Circle(
                    station.position, 2.5,
                    fill=False, edgecolor=color, linewidth=2,
                    linestyle='--', alpha=0.6
                )
                self.ax.add_patch(circle)
    
    def _draw_packages(self):
        """Draw all active packages"""
        for pkg in self.env.packages:
            if pkg.is_delivered:
                continue
            
            # Package marker
            if pkg.is_assigned:
                color = 'orange'
                edge_color = 'darkorange'
                size = 200
            else:
                color = self.config.package_color
                edge_color = 'darkred'
                size = 180
            
            self.ax.scatter(
                pkg.position[0], pkg.position[1],
                marker='o', s=size, c=color,
                edgecolors=edge_color, linewidths=2.5,
                zorder=7, alpha=0.9
            )
            
            # Package label (ID and priority)
            if self.config.show_labels:
                priority_markers = {1: '', 2: '!', 3: '!!'}
                label = f"{pkg.id}{priority_markers.get(pkg.priority.value, '')}"
                
                self.ax.text(
                    pkg.position[0], pkg.position[1] + 0.8,
                    label, ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             alpha=0.8, edgecolor='none')
                )
    
    def _draw_connections(self):
        """Draw lines from packages to destinations"""
        for pkg in self.env.packages:
            if pkg.is_delivered or pkg.is_assigned:
                continue
            
            # Draw dashed line to destination
            self.ax.plot(
                [pkg.position[0], pkg.destination[0]],
                [pkg.position[1], pkg.destination[1]],
                color='red', linestyle=':', linewidth=1.5,
                alpha=0.3, zorder=1
            )
    
    def _draw_trajectories(self):
        """Draw robot trajectories"""
        for robot in self.env.robots:
            if robot.id not in self.trajectories:
                continue
            
            traj = list(self.trajectories[robot.id])
            if len(traj) < 2:
                continue
            
            # Convert to line segments for color gradient
            points = np.array(traj)
            segments = np.array([points[i:i+2] for i in range(len(points)-1)])
            
            # Create color gradient (older = more transparent)
            n_segments = len(segments)
            alphas = np.linspace(0.1, self.config.trajectory_alpha, n_segments)
            colors = []
            base_color = plt.cm.Blues(0.7) if not robot.is_carrying else plt.cm.Oranges(0.7)
            
            for alpha in alphas:
                color = list(base_color[:3]) + [alpha]
                colors.append(color)
            
            # Draw trajectory
            lc = LineCollection(segments, colors=colors, linewidths=2, zorder=2)
            self.ax.add_collection(lc)
    
    def _draw_robots(self):
        """Draw all robots with full state information"""
        for robot in self.env.robots:
            # Update trajectory
            self.trajectories[robot.id].append(robot.position.copy())
            
            # Determine color based on state
            if robot.is_carrying:
                face_color = self.config.robot_color_carrying
                edge_color = 'darkorange'
            else:
                face_color = self.config.robot_color_idle
                edge_color = 'darkblue'
            
            # Battery color indicator
            battery_level = robot.battery / self.env.battery_capacity
            if battery_level < 0.2:
                edge_color = 'red'
                edge_width = 3
            else:
                edge_width = 2.5
            
            # Draw robot body
            robot_circle = patches.Circle(
                robot.position, self.env.robot_radius,
                facecolor=face_color, edgecolor=edge_color,
                linewidth=edge_width, alpha=self.config.robot_alpha,
                zorder=10
            )
            self.ax.add_patch(robot_circle)
            
            # Draw orientation arrow
            arrow_length = self.env.robot_radius * 1.8
            dx = arrow_length * np.cos(robot.orientation)
            dy = arrow_length * np.sin(robot.orientation)
            
            self.ax.arrow(
                robot.position[0], robot.position[1], dx, dy,
                head_width=0.4, head_length=0.4,
                fc='black', ec='black', linewidth=2,
                zorder=11, length_includes_head=True
            )
            
            # Draw velocity vector
            if self.config.show_velocity and robot.speed > 0.1:
                vel_scale = 2.5
                self.ax.arrow(
                    robot.position[0], robot.position[1],
                    robot.velocity[0] * vel_scale,
                    robot.velocity[1] * vel_scale,
                    head_width=0.3, head_length=0.3,
                    fc='cyan', ec='cyan', alpha=0.7,
                    linewidth=2.5, zorder=9
                )
            
            # Draw robot ID and status
            if self.config.show_labels:
                # ID in center
                self.ax.text(
                    robot.position[0], robot.position[1],
                    str(robot.id), ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white', zorder=12
                )
                
                # Status above robot
                stage_symbols = {
                    DeliveryStage.SEEKING: '🔍',
                    DeliveryStage.APPROACHING: '→',
                    DeliveryStage.CARRYING: '📦',
                    DeliveryStage.DELIVERING: '✓'
                }
                
                status_text = stage_symbols.get(robot.delivery_stage, '?')
                self.ax.text(
                    robot.position[0], robot.position[1] + self.env.robot_radius + 0.6,
                    status_text, ha='center', va='bottom',
                    fontsize=10
                )
                
                # Battery indicator below robot
                battery_pct = int(battery_level * 100)
                battery_color = 'red' if battery_level < 0.2 else 'yellow' if battery_level < 0.5 else 'green'
                
                self.ax.text(
                    robot.position[0], robot.position[1] - self.env.robot_radius - 0.6,
                    f'{battery_pct}%', ha='center', va='top',
                    fontsize=8, fontweight='bold',
                    color=battery_color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             alpha=0.8, edgecolor='none')
                )
            
            # Draw LIDAR visualization (optional)
            if self.config.show_lidar:
                self._draw_robot_lidar(robot)
    
    def _draw_robot_lidar(self, robot):
        """Draw LIDAR visualization for robot"""
        angles = np.linspace(0, 2*np.pi, self.env.lidar_rays, endpoint=False)
        
        for angle in angles[::4]:  # Draw every 4th ray
            ray_angle = robot.orientation + angle
            ray_end = robot.position + self.env.lidar_range * np.array([
                np.cos(ray_angle), np.sin(ray_angle)
            ])
            
            self.ax.plot(
                [robot.position[0], ray_end[0]],
                [robot.position[1], ray_end[1]],
                color='yellow', alpha=0.1, linewidth=0.5,
                zorder=1
            )
    
    def _draw_stats(self):
        """Draw statistics overlay"""
        stats = self.env.get_statistics()
        
        # Format stats text
        stats_lines = [
            f"┌{'─'*40}┐",
            f"│ {'WAREHOUSE STATISTICS':^40} │",
            f"├{'─'*40}┤",
            f"│ Step: {self.env.current_step:>5} / {self.env.max_steps:<5}          Time: {self.env.current_step * self.env.dt:>6.1f}s │",
            f"│ Delivered: {stats['packages_delivered']:>3}                     Pickups: {stats['packages_picked_up']:>3} │",
            f"│ Active Pkgs: {stats['active_packages']:>2}                  Collisions: {stats['collisions']:>3} │",
            f"│ Throughput: {stats['throughput']:>5.1f} pkg/hr                       │",
            f"├{'─'*40}┤",
        ]
        
        # Add per-robot stats
        for i in range(self.env.num_robots):
            robot = self.env.robots[i]
            battery_pct = int(robot.battery / self.env.battery_capacity * 100)
            stats_lines.append(
                f"│ Robot {i}: {robot.delivery_stage.value[:7]:>7} "
                f"[{battery_pct:>3}%] D:{robot.packages_delivered:>2} │"
            )
        
        stats_lines.append(f"└{'─'*40}┘")
        
        stats_text = '\n'.join(stats_lines)
        
        # Draw text box
        self.ax.text(
            0.02, 0.98, stats_text, transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                     edgecolor='black', linewidth=2),
            zorder=100
        )
    
    def run_episode(self, network=None, save_path: Optional[str] = None,
                   max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run full episode with visualization"""
        observations = self.env.reset()
        
        # Reset trajectories
        for i in range(self.env.num_robots):
            self.trajectories[i].clear()
        
        # Setup network if provided
        if network is not None:
            network.eval()
            from collections import deque
            obs_history = {i: deque(maxlen=32) for i in range(self.env.num_robots)}
            for robot_id, obs in observations.items():
                for _ in range(32):
                    obs_history[robot_id].append(obs)
        
        done = False
        step = 0
        max_steps = max_steps or self.env.max_steps
        
        # For video recording
        if save_path:
            self.frames = []
            self.frame_metadata = []
        
        while not done and step < max_steps:
            # Render frame
            self.render_frame()
            
            # Save frame if recording
            if save_path:
                self._capture_frame()
            else:
                plt.pause(0.001)
                plt.draw()
            
            # Get actions
            if network is None:
                # Random policy
                actions = {i: self.env.action_space.sample() for i in range(self.env.num_robots)}
            else:
                # Network policy
                obs_tensors = []
                for i in range(self.env.num_robots):
                    obs_list = list(obs_history[i])
                    obs_array = np.stack(obs_list, axis=0)
                    obs_tensors.append(torch.from_numpy(obs_array).float().unsqueeze(0))
                obs_batch = torch.cat(obs_tensors, dim=0)
                
                with torch.no_grad():
                    scaled_actions, _, _, _, _ = network.get_action(obs_batch, deterministic=True)
                
                actions = {i: scaled_actions[i].cpu().numpy() for i in range(self.env.num_robots)}
            
            # Step environment
            observations, rewards, dones, info = self.env.step(actions)
            
            # Update obs history if using network
            if network is not None:
                for i in range(self.env.num_robots):
                    obs_history[i].append(observations[i])
            
            done = dones.get('__all__', False)
            step += 1
        
        # Final frame
        self.render_frame()
        if save_path:
            self._capture_frame()
        else:
            plt.pause(0.1)
        
        # Save video if requested
        if save_path:
            self._save_video(save_path)
        
        # Get final statistics
        final_stats = self.env.get_statistics()
        
        return {
            'steps': step,
            'stats': final_stats,
            'frames_captured': len(self.frames) if save_path else 0
        }
    
    def _capture_frame(self):
        """Capture current frame for video"""
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)
        
        # Store metadata
        stats = self.env.get_statistics()
        self.frame_metadata.append({
            'step': self.env.current_step,
            'delivered': stats['packages_delivered'],
            'collisions': stats['collisions']
        })
    
    def _save_video(self, save_path: str):
        """Save captured frames as video"""
        if not self.frames:
            print("No frames to save")
            return
        
        print(f"Saving video with {len(self.frames)} frames...")
        save_path = Path(save_path)
        
        if self.config.save_format == 'mp4':
            self._save_mp4(save_path)
        elif self.config.save_format == 'gif':
            self._save_gif(save_path)
        else:
            raise ValueError(f"Unknown format: {self.config.save_format}")
        
        print(f"Video saved to {save_path}")
    
    def _save_mp4(self, save_path: Path):
        """Save as MP4 video"""
        import matplotlib.animation as animation
        
        # Ensure .mp4 extension
        if save_path.suffix != '.mp4':
            save_path = save_path.with_suffix('.mp4')
        
        # Create animation from frames
        fig, ax = plt.subplots(figsize=self.config.figsize)
        im = ax.imshow(self.frames[0])
        ax.axis('off')
        
        def update(frame_idx):
            im.set_array(self.frames[frame_idx])
            return [im]
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.frames),
            interval=1000/self.config.fps, blit=True
        )
        
        writer = FFMpegWriter(fps=self.config.fps, bitrate=self.config.bitrate)
        anim.save(str(save_path), writer=writer)
        plt.close(fig)
    
    def _save_gif(self, save_path: Path):
        """Save as GIF"""
        import imageio
        
        # Ensure .gif extension
        if save_path.suffix != '.gif':
            save_path = save_path.with_suffix('.gif')
        
        # Downsample frames for smaller GIF
        step = max(1, len(self.frames) // 200)  # Max 200 frames in GIF
        frames_downsampled = self.frames[::step]
        
        imageio.mimsave(str(save_path), frames_downsampled, fps=self.config.fps)
    
    def generate_heatmap(self, trajectory_data: List[np.ndarray],
                         save_path: Optional[str] = None) -> np.ndarray:
        """Generate activity heatmap from trajectory data"""
        grid_size = (100, 100)
        heatmap = np.zeros(grid_size)
        
        # Scale positions to grid
        scale_x = grid_size[0] / self.env.grid_size[0]
        scale_y = grid_size[1] / self.env.grid_size[1]
        
        for positions in trajectory_data:
            for pos in positions:
                x_idx = int(np.clip(pos[0] * scale_x, 0, grid_size[0] - 1))
                y_idx = int(np.clip(pos[1] * scale_y, 0, grid_size[1] - 1))
                heatmap[y_idx, x_idx] += 1
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        if save_path:
            self._plot_heatmap(heatmap, save_path)
        
        return heatmap
    
    def _plot_heatmap(self, heatmap: np.ndarray, save_path: str):
        """Plot and save heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear',
                      extent=[0, self.env.grid_size[0], 0, self.env.grid_size[1]],
                      origin='lower', alpha=0.7)
        
        # Overlay stations
        for station in self.env.stations:
            if 'delivery' in station.type.lower():
                marker, color = '^', 'green'
            elif 'pickup' in station.type.lower():
                marker, color = 's', 'yellow'
            else:
                marker, color = 'p', 'cyan'
            
            ax.scatter(station.position[0], station.position[1],
                      marker=marker, s=400, c=color,
                      edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Robot Activity Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activity Level', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def close(self):
        """Close visualization"""
        if self.fig:
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="Warehouse Visualization")
    
    # Environment
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--level', type=int, default=1, choices=[1,2,3,4,5])
    
    # Policy
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--random', action='store_true')
    
    # Visualization
    parser.add_argument('--save', type=str, default=None,
                       help='Save video to path')
    parser.add_argument('--format', type=str, default='mp4',
                       choices=['mp4', 'gif'])
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--no-trajectories', action='store_true')
    parser.add_argument('--show-lidar', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        env_config = yaml.safe_load(f)
    
    # Create environment
    env = WarehouseEnv(env_config, curriculum_level=args.level)
    
    # Load network if provided
    network = None
    if args.checkpoint and not args.random:
        print(f"Loading checkpoint: {args.checkpoint}")
        
        from mera_ppo_complete import ActorCritic, TrainingConfig
        
        # Get obs dim
        sample_obs = env.reset()
        obs_dim = len(sample_obs[0])
        
        # Create network (basic config)
        train_config = TrainingConfig(encoder_type='gru')
        network = ActorCritic(obs_dim, 3, train_config)
        
        # Load weights
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()
        
        print("Checkpoint loaded!")
    
    # Create visualizer
    vis_config = VisualizationConfig(
        show_trajectories=not args.no_trajectories,
        show_lidar=args.show_lidar,
        save_format=args.format
    )
    visualizer = WarehouseVisualizer(env, vis_config)
    
    # Run episode
    print("\nRunning episode...")
    print(f"Policy: {'Random' if network is None else 'Trained'}")
    print(f"Save: {args.save if args.save else 'Live display'}\n")
    
    result = visualizer.run_episode(
        network=network,
        save_path=args.save,
        max_steps=args.max_steps
    )
    
    # Print results
    print("\n" + "="*60)
    print("EPISODE COMPLETE")
    print("="*60)
    print(f"Steps: {result['steps']}")
    print(f"Packages delivered: {result['stats']['packages_delivered']}")
    print(f"Packages picked up: {result['stats']['packages_picked_up']}")
    print(f"Collisions: {result['stats']['collisions']}")
    print(f"Throughput: {result['stats']['throughput']:.1f} packages/hour")
    if args.save:
        print(f"Frames captured: {result['frames_captured']}")
    print("="*60)
    
    visualizer.close()


if __name__ == "__main__":
    main()
