"""
QREA Evolution System
Population-based learning with genetic operations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from copy import deepcopy
import random


class EvolutionaryPopulation:
    """
    Manage population of agents with evolutionary operations
    """
    
    def __init__(self, base_agent, config: dict, device: torch.device):
        self.config = config
        self.device = device
        evo_cfg = config['evolution']
        
        self.population_size = evo_cfg['population_size']
        self.elite_size = evo_cfg['elite_size']
        self.crossover_rate = evo_cfg['crossover']['rate']
        self.mutation_rate = evo_cfg['mutation']['rate']
        self.mutation_std = evo_cfg['mutation']['std']
        self.hgt_enabled = evo_cfg['hgt']['enabled']
        self.hgt_rate = evo_cfg['hgt']['rate']
        
        # Inheritance weights
        self.darwinian_weight = evo_cfg['inheritance']['darwinian_weight']
        self.baldwinian_weight = evo_cfg['inheritance']['baldwinian_weight']
        self.lamarckian_weight = evo_cfg['inheritance']['lamarckian_weight']
        
        # Population
        self.population = []
        self.fitnesses = []
        
        # Initialize population
        for i in range(self.population_size):
            agent = deepcopy(base_agent)
            # Add random variation
            self._randomize_agent(agent, scale=0.1)
            self.population.append(agent)
            self.fitnesses.append(0.0)
        
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        
        # SAES: Success tracking for adaptive mutation
        self.success_rate = 0.0
        self.success_history = []
        self.adaptive_mutation_std = self.mutation_std
    
    def _randomize_agent(self, agent, scale: float = 1.0):
        """Add random noise to agent parameters"""
        networks = agent.get_networks()
        
        with torch.no_grad():  # Prevent autograd baggage
            for network in networks.values():
                for param in network.parameters():
                    if param.requires_grad:
                        param.data.add_(torch.randn_like(param.data) * scale * 0.01)
    
    def evaluate_population(self, env, num_episodes: int = 5) -> Dict:
        """Evaluate all agents in population"""
        self.fitnesses = []
        
        for i, agent in enumerate(self.population):
            total_reward = 0.0
            total_packages = 0
            total_energy = 0.0
            
            for _ in range(num_episodes):
                obs_dict = env.reset()
                agent.reset()
                done = False
                episode_reward = 0.0
                
                while not done:
                    actions = {}
                    for robot_id in obs_dict.keys():
                        actions[robot_id] = agent.act(obs_dict[robot_id])
                    
                    obs_dict, rewards, dones, infos = env.step(actions)
                    
                    episode_reward += sum(rewards.values())
                    done = dones.get('__all__', False)
                
                # Get statistics
                stats = env.get_statistics()
                total_reward += episode_reward
                total_packages += stats['packages_delivered']
                total_energy += stats['total_energy']
            
            # Fitness = weighted combination with better shaping
            avg_reward = total_reward / num_episodes
            avg_packages = total_packages / num_episodes
            avg_energy = total_energy / num_episodes
            
            # Normalize and shape fitness for better selection pressure
            # Even with 0 deliveries, reward shaping provides signal
            fitness = avg_reward  # Primary: use the shaped reward directly
            
            # Bonuses for actual progress
            if avg_packages > 0:
                fitness += 100.0 * avg_packages  # Big bonus for deliveries
            
            # Small energy penalty (don't dominate early when rewards are small)
            fitness -= 0.001 * avg_energy
            
            self.fitnesses.append(fitness)
        
        # Track metrics
        self.best_fitness_history.append(max(self.fitnesses))
        self.diversity_history.append(self._compute_diversity())
        
        return {
            'best_fitness': max(self.fitnesses),
            'mean_fitness': np.mean(self.fitnesses),
            'worst_fitness': min(self.fitnesses),
            'diversity': self.diversity_history[-1]
        }
    
    def evolve_generation(self) -> Dict:
        """Evolve to next generation with SAES-style adaptation"""
        # Selection
        elite_indices, selected_indices = self._selection()
        
        # Track best fitness before evolution
        best_fitness_before = max(self.fitnesses) if self.fitnesses else 0.0
        
        # Create next generation
        next_population = []
        improvements = 0
        offspring_count = 0
        
        # Elitism: keep best agents
        for idx in elite_indices:
            next_population.append(deepcopy(self.population[idx]))
        
        # Generate offspring
        while len(next_population) < self.population_size:
            # Select parents
            parent1_idx = random.choice(selected_indices)
            parent2_idx = random.choice(selected_indices)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            parent_fitness = max(self.fitnesses[parent1_idx], self.fitnesses[parent2_idx])
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = deepcopy(parent1)
            
            # Mutation with adaptive std
            if random.random() < self.mutation_rate:
                self._mutate_adaptive(offspring)
            
            # Horizontal Gene Transfer (share successful strategies)
            if self.hgt_enabled and random.random() < self.hgt_rate:
                best_idx = elite_indices[0]
                self._horizontal_gene_transfer(offspring, self.population[best_idx])
            
            next_population.append(offspring)
            offspring_count += 1
        
        self.population = next_population
        
        # SAES: Update success rate based on improvements
        # (Will be accurate after next evaluation)
        best_fitness_after = max(self.fitnesses) if self.fitnesses else 0.0
        improved = best_fitness_after > best_fitness_before
        self._update_success(improved)
        
        self.generation += 1
        
        return {
            'generation': self.generation,
            'elite_size': len(elite_indices),
            'selected_size': len(selected_indices),
            'success_rate': self.success_rate,
            'mutation_std': self.adaptive_mutation_std
        }
    
    def _selection(self) -> Tuple[List[int], List[int]]:
        """
        Tournament selection
        Returns: elite_indices, selected_indices
        """
        # Elite: top performers
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elite_indices = sorted_indices[:self.elite_size].tolist()
        
        # Tournament selection for breeding
        selection_cfg = self.config['evolution']['selection']
        tournament_size = selection_cfg['tournament_size']
        
        selected_indices = []
        for _ in range(self.population_size):
            # Random tournament
            tournament = random.sample(range(self.population_size), tournament_size)
            winner = max(tournament, key=lambda i: self.fitnesses[i])
            selected_indices.append(winner)
        
        return elite_indices, selected_indices
    
    def _crossover(self, parent1, parent2):
        """
        Uniform crossover: mix parameters from both parents
        Now with torch.no_grad() and device safety
        """
        offspring = deepcopy(parent1)
        
        networks1 = parent1.get_networks()
        networks2 = parent2.get_networks()
        offspring_networks = offspring.get_networks()
        
        crossover_cfg = self.config['evolution']['crossover']
        alpha = crossover_cfg['alpha']
        
        with torch.no_grad():  # Prevent autograd baggage
            # Mix each network
            for net_name in networks1.keys():
                net1 = networks1[net_name]
                net2 = networks2[net_name]
                offspring_net = offspring_networks[net_name]
                
                # Match parameters by name for safety
                params1 = dict(net1.named_parameters())
                params2 = dict(net2.named_parameters())
                params_off = dict(offspring_net.named_parameters())
                
                for name in params1.keys():
                    if name in params2 and name in params_off:
                        param1 = params1[name]
                        param2 = params2[name]
                        param_off = params_off[name]
                        
                        if param1.requires_grad:
                            # Ensure same device
                            p2_data = param2.data.to(param1.device)
                            # Weighted average
                            param_off.data.copy_(alpha * param1.data + (1 - alpha) * p2_data)
        
        return offspring
    
    def _mutate(self, agent):
        """
        Gaussian mutation with torch.no_grad() for safety
        """
        mutation_cfg = self.config['evolution']['mutation']
        std = mutation_cfg['std']
        adaptive = mutation_cfg.get('adaptive', False)
        
        # Adaptive mutation: smaller mutations later
        if adaptive:
            std = std * np.exp(-self.generation / 100.0)
        
        networks = agent.get_networks()
        
        with torch.no_grad():  # Prevent autograd baggage
            for network in networks.values():
                for param in network.parameters():
                    if param.requires_grad:
                        mutation = torch.randn_like(param.data) * std
                        param.data.add_(mutation)
    
    def _mutate_adaptive(self, agent):
        """Mutation using adaptive SAES-style sigma"""
        networks = agent.get_networks()
        
        with torch.no_grad():
            for network in networks.values():
                for param in network.parameters():
                    if param.requires_grad:
                        mutation = torch.randn_like(param.data) * self.adaptive_mutation_std
                        param.data.add_(mutation)
    
    def _update_success(self, improved: bool):
        """Update success rate and adapt mutation strength (SAES-style)"""
        # Track success with exponential moving average
        alpha = 0.2  # EMA weight
        success_signal = 1.0 if improved else 0.0
        self.success_rate = alpha * success_signal + (1 - alpha) * self.success_rate
        self.success_history.append(success_signal)
        
        # Adapt mutation std based on success rate
        # Target success rate ~0.2 (1/5 rule)
        target_success = 0.2
        if self.success_rate > target_success:
            # Too many successes → increase mutation (explore more)
            self.adaptive_mutation_std *= 1.05
        else:
            # Too few successes → decrease mutation (exploit more)
            self.adaptive_mutation_std *= 0.95
        
        # Clamp to reasonable range
        self.adaptive_mutation_std = np.clip(
            self.adaptive_mutation_std,
            self.mutation_std * 0.1,  # Min: 10% of base
            self.mutation_std * 5.0    # Max: 5x base
        )
    
    def _horizontal_gene_transfer(self, recipient, donor):
        """
        Transfer successful strategies (specific parameters) from donor to recipient
        """
        # Transfer only policy network (behavioral strategies)
        donor_policy = donor.get_networks()['policy']
        recipient_policy = recipient.get_networks()['policy']
        
        # Transfer with some probability per parameter
        for (name_d, param_d), (name_r, param_r) in zip(
            donor_policy.named_parameters(),
            recipient_policy.named_parameters()
        ):
            if param_d.requires_grad and random.random() < 0.3:  # 30% transfer rate
                param_r.data = param_d.data.clone()
    
    def _compute_diversity(self) -> float:
        """
        Compute population diversity
        Measures how different agents are from each other
        """
        if len(self.population) < 2:
            return 0.0
        
        # Sample policy parameters from all agents
        param_vectors = []
        
        for agent in self.population:
            policy = agent.get_networks()['policy']
            params = torch.cat([p.flatten() for p in policy.parameters() if p.requires_grad])
            param_vectors.append(params)
        
        param_vectors = torch.stack(param_vectors)
        
        # Compute pairwise distances
        distances = []
        for i in range(len(param_vectors)):
            for j in range(i + 1, len(param_vectors)):
                dist = torch.norm(param_vectors[i] - param_vectors[j]).item()
                distances.append(dist)
        
        diversity = np.mean(distances) if distances else 0.0
        
        return diversity
    
    def get_best_agent(self):
        """Get best performing agent"""
        best_idx = np.argmax(self.fitnesses)
        return self.population[best_idx]
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics"""
        return {
            'generation': self.generation,
            'best_fitness': max(self.fitnesses) if self.fitnesses else 0.0,
            'mean_fitness': np.mean(self.fitnesses) if self.fitnesses else 0.0,
            'fitness_std': np.std(self.fitnesses) if self.fitnesses else 0.0,
            'diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'best_fitness_history': self.best_fitness_history
        }
    
    def save_population(self, path: str):
        """Save entire population"""
        population_data = []
        
        for i, agent in enumerate(self.population):
            agent_data = {
                'networks': {
                    name: net.state_dict()
                    for name, net in agent.get_networks().items()
                },
                'fitness': self.fitnesses[i] if i < len(self.fitnesses) else 0.0
            }
            population_data.append(agent_data)
        
        save_dict = {
            'population': population_data,
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }
        
        torch.save(save_dict, path)
    
    def load_population(self, path: str):
        """Load population from checkpoint"""
        save_dict = torch.load(path, map_location=self.device)
        
        self.generation = save_dict['generation']
        self.best_fitness_history = save_dict['best_fitness_history']
        self.diversity_history = save_dict['diversity_history']
        
        # Load agents
        for i, agent_data in enumerate(save_dict['population']):
            if i < len(self.population):
                networks = self.population[i].get_networks()
                for name, state_dict in agent_data['networks'].items():
                    networks[name].load_state_dict(state_dict)
                
                if i < len(self.fitnesses):
                    self.fitnesses[i] = agent_data['fitness']


class HybridLearning:
    """
    Combine evolutionary learning with gradient-based learning
    Darwinian + Baldwinian + Lamarckian inheritance
    """
    
    def __init__(self, population: EvolutionaryPopulation, trainer):
        self.population = population
        self.trainer = trainer
        self.config = population.config
        
        # Inheritance weights
        evo_cfg = self.config['evolution']['inheritance']
        self.darwinian = evo_cfg['darwinian_weight']
        self.baldwinian = evo_cfg['baldwinian_weight']
        self.lamarckian = evo_cfg['lamarckian_weight']
    
    def train_generation(self, env, num_episodes_per_agent: int = 5) -> Dict:
        """
        Train one generation with hybrid learning
        """
        metrics = {
            'darwinian_improvement': 0.0,
            'baldwinian_improvement': 0.0,
            'lamarckian_improvement': 0.0
        }
        
        # 1. Darwinian: Pure evolution (no learning)
        if self.darwinian > 0:
            initial_fitness = np.mean(self.population.fitnesses)
            eval_metrics = self.population.evaluate_population(env, num_episodes_per_agent)
            self.population.evolve_generation()
            final_fitness = np.mean(self.population.fitnesses)
            
            metrics['darwinian_improvement'] = final_fitness - initial_fitness
        
        # 2. Baldwinian: Learn during lifetime but don't inherit
        if self.baldwinian > 0:
            # Train agents but don't modify their genomes
            for agent in self.population.population:
                # Collect experience
                episodes = self._collect_episodes(env, agent, num_episodes_per_agent)
                
                # Train (but save original parameters)
                original_params = {}
                networks = agent.get_networks()
                for name, net in networks.items():
                    original_params[name] = {k: v.clone() for k, v in net.state_dict().items()}
                
                # Learn
                self.trainer.agent = agent
                self.trainer.train_step(episodes)
                
                # Restore original parameters (Baldwinian)
                for name, net in networks.items():
                    net.load_state_dict(original_params[name])
        
        # 3. Lamarckian: Learn and inherit acquired traits
        if self.lamarckian > 0:
            for agent in self.population.population:
                # Collect experience
                episodes = self._collect_episodes(env, agent, num_episodes_per_agent)
                
                # Train (parameters ARE modified)
                self.trainer.agent = agent
                train_metrics = self.trainer.train_step(episodes)
                
                if train_metrics:
                    metrics['lamarckian_improvement'] += train_metrics.get('mean_return', 0.0)
            
            metrics['lamarckian_improvement'] /= len(self.population.population)
        
        return metrics
    
    def _collect_episodes(self, env, agent, num_episodes: int) -> List[Dict]:
        """Collect episodes from agent interaction - ALL robots"""
        episodes = []
        
        for _ in range(num_episodes):
            episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            obs_dict = env.reset()
            agent.reset()
            done = False
            
            while not done:
                # Get actions for all robots
                actions = {}
                for robot_id in obs_dict.keys():
                    action = agent.act(obs_dict[robot_id])
                    actions[robot_id] = action
                    
                    # Store for ALL robots (critical fix!)
                    episode['observations'].append(obs_dict[robot_id])
                    episode['actions'].append(action)
                
                obs_dict, rewards, dones, _ = env.step(actions)
                
                # Sum rewards across ALL robots for multi-agent credit
                total_reward = sum(rewards.values())
                episode['rewards'].append(total_reward)
                
                # Done when all robots done or episode terminates
                done = dones.get('__all__', False)
                episode['dones'].append(done)
            
            episodes.append(episode)
        
        return episodes
