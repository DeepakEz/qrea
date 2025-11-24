"""
Main QREA Warehouse Simulation
Integrates all components: agents, UPRT, evolution, communication, safety
"""

import torch
import numpy as np
import yaml
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.warehouse_env import WarehouseEnv
from agents.robot_agent import QREARobotAgent
from fields.warehouse_uprt import WarehouseUPRT
from learning.trainer import QREATrainer
from evolution.evolutionary import EvolutionaryPopulation, HybridLearning
from communication.language import EmergentLanguage
from safety.safety_monitor import SafetyMonitor


class QREASimulation:
    """Main QREA simulation orchestrator"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(
            self.config['simulation']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Set seed
        seed = self.config['simulation']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize components
        self._initialize_environment()
        self._initialize_agents()
        self._initialize_uprt()
        self._initialize_communication()
        self._initialize_safety()
        self._initialize_logging()
        
        print("QREA Simulation initialized successfully!")
    
    def _initialize_environment(self):
        """Initialize warehouse environment"""
        print("Initializing environment...")
        self.env = WarehouseEnv(self.config)
    
    def _initialize_agents(self):
        """Initialize agents and learning systems"""
        print("Initializing agents...")
        
        # Create base agent
        self.base_agent = QREARobotAgent(self.config, self.device)
        
        # Initialize trainer
        self.trainer = QREATrainer(self.base_agent, self.config, self.device)
        
        # Initialize evolutionary population
        self.population = EvolutionaryPopulation(
            self.base_agent, self.config, self.device
        )
        
        # Initialize hybrid learning
        self.hybrid_learning = HybridLearning(self.population, self.trainer)
    
    def _initialize_uprt(self):
        """Initialize UPRT field system"""
        print("Initializing UPRT fields...")
        self.uprt = WarehouseUPRT(self.config, self.device)
    
    def _initialize_communication(self):
        """Initialize communication system"""
        print("Initializing communication...")
        
        # Calculate obs_dim dynamically based on num_robots
        num_robots = self.config['environment']['num_robots']
        lidar_rays = self.config['environment']['sensors']['lidar']['num_rays']
        # Observation: robot_state(10) + lidar(rays) + packages(400) + stations(12) + other_robots((num_robots-1)*8)
        obs_dim = 10 + lidar_rays + 400 + 12 + (num_robots - 1) * 8
        action_dim = 3
        
        self.language = EmergentLanguage(
            self.config, obs_dim, action_dim, self.device
        )
        
        # Communication optimizer
        comm_params = list(self.language.encoder.parameters()) + \
                     list(self.language.decoder.parameters())
        self.comm_optimizer = torch.optim.Adam(
            comm_params, lr=self.config['learning']['learning_rate']
        )
    
    def _initialize_safety(self):
        """Initialize safety monitoring"""
        print("Initializing safety systems...")
        self.safety_monitor = SafetyMonitor(self.config)
    
    def _initialize_logging(self):
        """Initialize logging and checkpointing"""
        print("Initializing logging...")
        
        # Create directories
        self.log_dir = Path(self.config['simulation'].get('log_dir', 'logs'))
        self.checkpoint_dir = Path(self.config['simulation'].get('checkpoint_dir', 'checkpoints'))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics = {
            'epoch': [],
            'mean_reward': [],
            'throughput': [],
            'collision_rate': [],
            'field_coherence': [],
            'language_diversity': [],
            'safety_score': [],
            'best_fitness': [],
            'population_diversity': []
        }
    
    def run_epoch(self, epoch: int) -> Dict:
        """Run one training epoch"""
        
        # Get best agent from population
        agent = self.population.get_best_agent()
        
        # Collect episodes
        episodes = []
        epoch_metrics = {
            'total_reward': 0.0,
            'total_packages': 0,
            'total_collisions': 0,
            'total_steps': 0
        }
        
        steps_per_epoch = self.config['simulation']['steps_per_epoch']
        current_steps = 0
        
        while current_steps < steps_per_epoch:
            episode, episode_metrics = self.run_episode(agent, epoch)
            episodes.append(episode)
            
            # Accumulate metrics
            epoch_metrics['total_reward'] += episode_metrics['total_reward']
            epoch_metrics['total_packages'] += episode_metrics['packages_delivered']
            epoch_metrics['total_collisions'] += episode_metrics['collisions']
            epoch_metrics['total_steps'] += episode_metrics['steps']
            
            current_steps += episode_metrics['steps']
        
        # Train on collected episodes
        train_metrics = self.trainer.train_step(episodes)
        
        # Train communication
        if self.config['communication']['language']['enabled']:
            comm_metrics = self.language.train_language(episodes, self.comm_optimizer)
        else:
            comm_metrics = {}
        
        # Combine metrics
        metrics = {
            'epoch': epoch,
            'mean_reward': epoch_metrics['total_reward'] / len(episodes),
            'mean_packages': epoch_metrics['total_packages'] / len(episodes),
            'mean_collisions': epoch_metrics['total_collisions'] / len(episodes),
            **train_metrics,
            **comm_metrics
        }
        
        return metrics
    
    def run_episode(self, agent, epoch: int) -> tuple:
        """Run one episode"""
        
        obs_dict = self.env.reset()
        agent.reset()
        self.uprt.reset()
        self.language.channel.reset()
        
        episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        episode_metrics = {
            'total_reward': 0.0,
            'packages_delivered': 0,
            'collisions': 0,
            'steps': 0
        }
        
        done = False
        step = 0
        
        while not done and step < self.config['environment']['max_episode_steps']:
            # Get robot states
            robot_states = {}
            for robot_id, info in enumerate(self.env.robots):
                robot_states[robot_id] = {
                    'position': info.position.tolist(),
                    'velocity': info.speed,
                    'battery': info.battery,
                    'carrying': info.is_carrying,
                    'packages_delivered': info.packages_delivered
                }
            
            # Update UPRT fields
            observations_list = [obs_dict[rid] for rid in sorted(obs_dict.keys())]
            self.uprt.update(list(robot_states.values()), observations_list, self.env.dt)
            
            # Communication
            if self.config['communication']['language']['enabled']:
                comm_influences = self.language.communicate(robot_states, obs_dict)
            else:
                comm_influences = {rid: torch.zeros(3, device=self.device) 
                                  for rid in obs_dict.keys()}
            
            # Get actions from agent
            actions = {}
            for robot_id in obs_dict.keys():
                action = agent.act(obs_dict[robot_id])
                
                # Add communication influence
                action = action + comm_influences[robot_id].detach().cpu().numpy() * 0.1
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                actions[robot_id] = action
            
            # Safety verification
            actions, all_safe = self.safety_monitor.verify_step(actions, robot_states)
            
            # Environment step
            next_obs_dict, rewards, dones, infos = self.env.step(actions)
            
            # Store experience (for first robot)
            if 0 in obs_dict:
                episode['observations'].append(obs_dict[0])
                episode['actions'].append(actions[0])
                episode['rewards'].append(rewards.get(0, 0.0))
                episode['dones'].append(dones.get(0, False))
            
            # Update metrics
            episode_metrics['total_reward'] += sum(rewards.values())
            episode_metrics['steps'] += 1
            
            # Count collisions
            for info in infos.values():
                if isinstance(info, dict) and 'collision' in info:
                    episode_metrics['collisions'] += 1
            
            obs_dict = next_obs_dict
            done = dones.get('__all__', False)
            step += 1
        
        # Get final statistics
        stats = self.env.get_statistics()
        episode_metrics['packages_delivered'] = stats['packages_delivered']
        
        return episode, episode_metrics
    
    def evaluate(self, agent, num_episodes: int = 10) -> Dict:
        """Evaluate agent performance"""
        
        total_reward = 0.0
        total_packages = 0
        total_collisions = 0
        total_energy = 0.0
        
        for _ in range(num_episodes):
            obs_dict = self.env.reset()
            agent.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                actions = {}
                for robot_id in obs_dict.keys():
                    actions[robot_id] = agent.act(obs_dict[robot_id], deterministic=True)
                
                obs_dict, rewards, dones, _ = self.env.step(actions)
                
                episode_reward += sum(rewards.values())
                done = dones.get('__all__', False)
            
            stats = self.env.get_statistics()
            total_reward += episode_reward
            total_packages += stats['packages_delivered']
            total_collisions += stats['collisions']
            total_energy += stats['total_energy']
        
        return {
            'mean_reward': total_reward / num_episodes,
            'throughput': total_packages / num_episodes,
            'collision_rate': total_collisions / num_episodes,
            'efficiency': total_packages / (total_energy + 1e-6)
        }
    
    def train(self):
        """Main training loop"""
        
        num_epochs = self.config['simulation']['num_epochs']
        eval_interval = self.config['simulation']['eval_interval']
        checkpoint_interval = self.config['simulation']['checkpoint_interval']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            # Train for one epoch
            epoch_metrics = self.run_epoch(epoch)
            
            # Evaluate
            if epoch % eval_interval == 0:
                best_agent = self.population.get_best_agent()
                eval_metrics = self.evaluate(best_agent)
                
                # Get UPRT metrics
                uprt_metrics = self.uprt.get_metrics()
                
                # Get safety metrics
                safety_metrics = self.safety_monitor.get_safety_metrics()
                
                # Get evolution metrics
                evo_metrics = self.population.get_statistics()
                
                # Get language metrics
                if self.config['communication']['language']['enabled']:
                    lang_metrics = self.language.get_language_statistics()
                else:
                    lang_metrics = {}
                
                # Combine all metrics
                all_metrics = {
                    **epoch_metrics,
                    **eval_metrics,
                    **uprt_metrics,
                    **safety_metrics,
                    **evo_metrics,
                    **lang_metrics
                }
                
                # Log metrics
                self._log_metrics(epoch, all_metrics)
                
                # Print progress
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Reward: {eval_metrics['mean_reward']:.2f}")
                print(f"  Throughput: {eval_metrics['throughput']:.2f} packages/episode")
                print(f"  Collisions: {eval_metrics['collision_rate']:.2f}")
                print(f"  Field Coherence: {uprt_metrics['field_coherence']:.4f}")
                print(f"  Safety Score: {safety_metrics['safety_score']:.4f}")
                print(f"  Best Fitness: {evo_metrics['best_fitness']:.2f}")
                if lang_metrics:
                    print(f"  Language Diversity: {lang_metrics['message_diversity']:.4f}")
            
            # Checkpoint
            if epoch % checkpoint_interval == 0 and epoch > 0:
                self.save_checkpoint(epoch)
            
            # Evolutionary step (every 10 epochs)
            if epoch % 10 == 0 and epoch > 0:
                print(f"\nRunning evolutionary step...")
                
                # Evaluate population
                eval_metrics = self.population.evaluate_population(
                    self.env,
                    num_episodes=self.config['evolution']['evaluation_episodes']
                )
                
                print(f"  Population fitness: {eval_metrics['best_fitness']:.2f} " +
                      f"(mean: {eval_metrics['mean_fitness']:.2f})")
                
                # Evolve
                self.population.evolve_generation()
        
        print("\n" + "=" * 80)
        print("Training complete!")
        
        # Final evaluation
        print("\nFinal evaluation...")
        best_agent = self.population.get_best_agent()
        final_metrics = self.evaluate(best_agent, num_episodes=50)
        
        print(f"\nFinal Performance:")
        print(f"  Mean Reward: {final_metrics['mean_reward']:.2f}")
        print(f"  Throughput: {final_metrics['throughput']:.2f} packages/episode")
        print(f"  Collision Rate: {final_metrics['collision_rate']:.2f}")
        print(f"  Efficiency: {final_metrics['efficiency']:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint('final')
        
        return final_metrics
    
    def _log_metrics(self, epoch: int, metrics: Dict):
        """Log metrics to file"""
        
        # Append to metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            if isinstance(value, (int, float)):
                self.metrics[key].append(value)
        
        # Save metrics to JSON
        metrics_file = self.run_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        # Save trainer
        self.trainer.save_checkpoint(str(checkpoint_path))
        
        # Save population
        population_path = self.checkpoint_dir / f'population_epoch_{epoch}.pt'
        self.population.save_population(str(population_path))
        
        # Save UPRT
        uprt_path = self.checkpoint_dir / f'uprt_epoch_{epoch}.pt'
        torch.save(self.uprt.field.state_dict(), uprt_path)
        
        # Save communication
        if self.config['communication']['language']['enabled']:
            comm_path = self.checkpoint_dir / f'communication_epoch_{epoch}.pt'
            torch.save({
                'encoder': self.language.encoder.state_dict(),
                'decoder': self.language.decoder.state_dict()
            }, comm_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, epoch):
        """Load training checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        self.trainer.load_checkpoint(str(checkpoint_path))
        
        population_path = self.checkpoint_dir / f'population_epoch_{epoch}.pt'
        if population_path.exists():
            self.population.load_population(str(population_path))
        
        print(f"Loaded checkpoint from {checkpoint_path}")


def main():
    """Main entry point"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='QREA Warehouse Simulation')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint epoch to resume from')
    
    args = parser.parse_args()
    
    # Create simulation
    sim = QREASimulation(args.config)
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        sim.load_checkpoint(args.checkpoint)
    
    # Run training
    sim.train()


if __name__ == '__main__':
    main()
