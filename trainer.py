"""
QREA Training Module
World model learning and actor-critic training
Feature 1: Mixed-precision (AMP) + Ensemble Disagreement
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

# Try new AMP API first (PyTorch 2.0+)
try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_DEVICE = 'cuda'  # New API requires device parameter
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AUTOCAST_DEVICE = None  # Old API doesn't use device parameter


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int, prioritized: bool = True):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, experience: Dict, priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        if self.prioritized:
            self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample batch from buffer"""
        if self.prioritized and len(self.priorities) > 0:
            priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities ** 0.6
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(len(self.buffer), batch_size)
            samples = [self.buffer[idx] for idx in indices]
            weights = np.ones(batch_size, dtype=np.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for prioritized replay"""
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class QREATrainer:
    """Train QREA agent"""
    
    def __init__(self, agent, config: dict, device: torch.device):
        self.agent = agent
        self.config = config
        self.device = device
        
        learning_cfg = config['learning']
        
        # Feature 1: AMP support
        self.use_amp = learning_cfg.get('amp', False)
        # Use new API: torch.amp.GradScaler('cuda', ...) instead of torch.cuda.amp.GradScaler(...)
        if self.use_amp:
            try:
                # Try new API first (PyTorch 2.0+)
                from torch.amp import GradScaler as NewGradScaler
                self.scaler_world = NewGradScaler('cuda', enabled=self.use_amp)
                self.scaler_ac = NewGradScaler('cuda', enabled=self.use_amp)  # Single scaler for both actor+critic
            except (ImportError, TypeError):
                # Fall back to old API
                self.scaler_world = GradScaler(enabled=self.use_amp)
                self.scaler_ac = GradScaler(enabled=self.use_amp)  # Single scaler for both actor+critic
        else:
            self.scaler_world = GradScaler(enabled=False)
            self.scaler_ac = GradScaler(enabled=False)  # Single scaler for both actor+critic
        
        # Feature 1: Ensemble world models for disagreement
        self.num_ensemble = learning_cfg.get('ensemble_world_models', 1)
        
        # Get networks
        networks = agent.get_networks()
        
        # Create ensemble if needed
        if self.num_ensemble > 1:
            self.world_models = [networks['world_model']]
            # Clone world model for ensemble
            for _ in range(self.num_ensemble - 1):
                import copy
                ensemble_model = copy.deepcopy(networks['world_model'])
                self.world_models.append(ensemble_model)
            self.world_model = self.world_models[0]  # Primary model
        else:
            self.world_model = networks['world_model']
            self.world_models = [self.world_model]
        
        self.policy = networks['policy']
        self.value = networks['value']
        self.intrinsic = networks['intrinsic']
        
        # Optimizers
        lr = learning_cfg['learning_rate']
        weight_decay = learning_cfg['weight_decay']
        
        # World model optimizer(s) - one per ensemble member
        if self.num_ensemble > 1:
            self.world_model_optimizer = optim.Adam(
                [p for wm in self.world_models for p in wm.parameters()],
                lr=lr, weight_decay=weight_decay
            )
        else:
            self.world_model_optimizer = optim.Adam(
                self.world_model.parameters(), lr=lr, weight_decay=weight_decay
            )
        
        self.actor_optimizer = optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.value.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        if config['agent']['intrinsic']['novelty']['enabled']:
            intrinsic_lr = config['agent']['intrinsic']['novelty']['predictor_lr']
            self.intrinsic_optimizer = optim.Adam(
                self.intrinsic.predictor_net.parameters(), lr=intrinsic_lr
            )
        
        # Replay buffer
        replay_cfg = learning_cfg['replay']
        self.replay_buffer = ReplayBuffer(
            capacity=replay_cfg['capacity'],
            prioritized=replay_cfg['prioritized']
        )
        
        # Training parameters
        self.batch_size = learning_cfg['world_model']['batch_size']
        self.sequence_length = learning_cfg['world_model']['sequence_length']
        self.free_nats = learning_cfg['world_model']['free_nats']
        self.kl_weight = learning_cfg['world_model']['kl_weight']
        self.discount = learning_cfg['actor_critic']['discount']
        self.lambda_gae = learning_cfg['actor_critic']['lambda_gae']
        self.entropy_weight = learning_cfg['actor_critic']['entropy_weight']
        self.grad_clip = learning_cfg['grad_clip']
        
        # Imagination
        self.imagination_horizon = learning_cfg['imagination']['horizon']
        self.num_imagination_trajectories = learning_cfg['imagination']['num_trajectories']
        
        # Actor-critic loss coefficients for combined optimization
        self.actor_coef = 1.0
        self.critic_coef = 0.5
        
        # Metrics
        self.metrics = {
            'world_model_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'intrinsic_loss': []
        }
    
    def train_step(self, episodes: List[Dict]) -> Dict[str, float]:
        """
        Train on batch of episodes
        
        Args:
            episodes: List of episode dictionaries with obs, actions, rewards, etc.
        
        Returns:
            metrics: Training metrics
        """
        # Add to replay buffer
        for episode in episodes:
            self.replay_buffer.push(episode)
        
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Train world model
        wm_metrics = self._train_world_model()
        
        # Train actor-critic with imagination
        ac_metrics = self._train_actor_critic()
        
        # Train intrinsic motivation
        if self.config['agent']['intrinsic']['novelty']['enabled']:
            intrinsic_metrics = self._train_intrinsic()
        else:
            intrinsic_metrics = {}
        
        return {**wm_metrics, **ac_metrics, **intrinsic_metrics}
    
    def _train_world_model(self) -> Dict[str, float]:
        """Train world model (RSSM) with AMP and ensemble disagreement"""
        # Sample batch
        episodes, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Prepare sequences
        obs_seqs = []
        action_seqs = []
        reward_seqs = []
        
        for episode in episodes:
            # Sample subsequence
            start_idx = random.randint(0, max(0, len(episode['observations']) - self.sequence_length))
            end_idx = start_idx + self.sequence_length
            
            obs_seqs.append(episode['observations'][start_idx:end_idx])
            action_seqs.append(episode['actions'][start_idx:end_idx])
            reward_seqs.append(episode['rewards'][start_idx:end_idx])
        
        # Pad sequences
        max_len = max(len(seq) for seq in obs_seqs)
        
        obs_tensor = torch.zeros(self.batch_size, max_len, obs_seqs[0][0].shape[0], device=self.device)
        action_tensor = torch.zeros(self.batch_size, max_len, action_seqs[0][0].shape[0], device=self.device)
        reward_tensor = torch.zeros(self.batch_size, max_len, 1, device=self.device)
        
        for i in range(self.batch_size):
            seq_len = len(obs_seqs[i])
            obs_tensor[i, :seq_len] = torch.FloatTensor(np.array(obs_seqs[i]))
            action_tensor[i, :seq_len] = torch.FloatTensor(np.array(action_seqs[i]))
            reward_tensor[i, :seq_len] = torch.FloatTensor(np.array(reward_seqs[i])).unsqueeze(-1)
        
        # Feature 1: Train with AMP
        self.world_model_optimizer.zero_grad(set_to_none=True)
        
        # Use device parameter for new API, none for old API
        autocast_kwargs = {'enabled': self.use_amp}
        if AUTOCAST_DEVICE is not None:
            autocast_kwargs['device_type'] = AUTOCAST_DEVICE
        
        with autocast(**autocast_kwargs):
            # Forward pass through ensemble
            ensemble_losses = []
            ensemble_states = []
            
            for wm in self.world_models:
                states = []
                state = wm.initial_state(self.batch_size, self.device)
                
                for t in range(max_len - 1):
                    state = wm.observe(obs_tensor[:, t], action_tensor[:, t], state)
                    states.append(state)
                
                ensemble_states.append(states)
                
                # Compute losses for this ensemble member
                # 1. Observation reconstruction
                recon_loss = 0
                for t, state in enumerate(states):
                    pred_obs = wm.decode_obs(state)
                    recon_loss += F.mse_loss(pred_obs, obs_tensor[:, t + 1])
                recon_loss /= len(states)
                
                # 2. Reward prediction
                reward_loss = 0
                for t, state in enumerate(states):
                    pred_reward = wm.predict_reward(state)
                    reward_loss += F.mse_loss(pred_reward, reward_tensor[:, t + 1])
                reward_loss /= len(states)
                
                # 3. KL divergence (posterior vs prior)
                kl_loss = 0
                for state in states:
                    # Posterior distribution
                    post_mean = state['mean']
                    post_std = state['std']
                    
                    # Prior: N(0, 1)
                    prior_mean = torch.zeros_like(post_mean)
                    prior_std = torch.ones_like(post_std)
                    
                    # KL divergence
                    kl = torch.distributions.kl_divergence(
                        torch.distributions.Normal(post_mean, post_std),
                        torch.distributions.Normal(prior_mean, prior_std)
                    )
                    kl = kl.sum(dim=-1).mean()
                    
                    # Free nats
                    kl = torch.maximum(kl, torch.tensor(self.free_nats, device=self.device))
                    kl_loss += kl
                kl_loss /= len(states)
                
                # Total loss for this ensemble member
                total_loss = recon_loss + reward_loss + self.kl_weight * kl_loss
                ensemble_losses.append({
                    'total': total_loss,
                    'recon': recon_loss,
                    'reward': reward_loss,
                    'kl': kl_loss
                })
            
            # Average losses across ensemble
            avg_total_loss = sum(l['total'] for l in ensemble_losses) / len(ensemble_losses)
            avg_recon = sum(l['recon'] for l in ensemble_losses) / len(ensemble_losses)
            avg_reward = sum(l['reward'] for l in ensemble_losses) / len(ensemble_losses)
            avg_kl = sum(l['kl'] for l in ensemble_losses) / len(ensemble_losses)
        
        # Optimize with AMP
        self.scaler_world.scale(avg_total_loss).backward()
        self.scaler_world.unscale_(self.world_model_optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for wm in self.world_models for p in wm.parameters()], 
            self.grad_clip
        )
        self.scaler_world.step(self.world_model_optimizer)
        self.scaler_world.update()
        
        # Feature 1: Compute disagreement if ensemble
        disagreement = 0.0
        if self.num_ensemble > 1:
            disagreement = self._compute_disagreement(ensemble_states)
        
        return {
            'world_model_loss': avg_total_loss.item(),
            'recon_loss': avg_recon.item(),
            'reward_loss': avg_reward.item(),
            'kl_loss': avg_kl.item(),
            'disagreement': disagreement
        }
    
    def _compute_disagreement(self, ensemble_states: List[List[Dict]]) -> float:
        """
        Compute disagreement bonus from ensemble predictions.
        
        Args:
            ensemble_states: [num_models][time_steps] list of states
            
        Returns:
            avg_disagreement: Average variance across models as intrinsic reward
        """
        if len(ensemble_states) < 2:
            return 0.0
        
        # Get stochastic latent predictions from each model
        with torch.no_grad():
            all_stoch = []
            for model_states in ensemble_states:
                # Extract stochastic component from states
                stoch_seq = torch.stack([s['stoch'] for s in model_states], dim=0)  # [T, B, D]
                all_stoch.append(stoch_seq)
            
            # Stack: [N_models, T, B, D]
            stoch_stack = torch.stack(all_stoch, dim=0)
            
            # Compute variance across models
            variance = stoch_stack.var(dim=0)  # [T, B, D]
            
            # Average variance as disagreement measure
            disagreement = variance.mean().item()
        
        return disagreement
    
    def _train_actor_critic(self) -> Dict[str, float]:
        """Train actor-critic with imagined rollouts using AMP"""
        # Sample starting states from replay buffer
        episodes, _, _ = self.replay_buffer.sample(self.num_imagination_trajectories)
        
        start_obs = []
        for episode in episodes:
            idx = random.randint(0, len(episode['observations']) - 1)
            start_obs.append(episode['observations'][idx])
        
        start_obs = torch.FloatTensor(np.array(start_obs)).to(self.device)
        
        # Get starting states
        start_actions = torch.zeros(self.num_imagination_trajectories, 3, device=self.device)
        prev_state = self.world_model.initial_state(self.num_imagination_trajectories, self.device)
        start_state = self.world_model.observe(start_obs, start_actions, prev_state)
        
        # Imagine trajectories with AMP
        autocast_kwargs = {'enabled': self.use_amp}
        if AUTOCAST_DEVICE is not None:
            autocast_kwargs['device_type'] = AUTOCAST_DEVICE
        
        with autocast(**autocast_kwargs):
            states = [start_state]
            actions = []
            rewards = []
            log_probs = []
            
            state = start_state
            for t in range(self.imagination_horizon):
                # Sample action from policy
                action, log_prob = self.policy(state, deterministic=False)
                
                # Imagine next state
                next_state = self.world_model.imagine(action, state)
                
                # Predict reward
                reward = self.world_model.predict_reward(state)
                
                # Add intrinsic reward
                intrinsic_reward = self.intrinsic.compute_novelty(state)
                reward = reward + intrinsic_reward
                
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                state = next_state
            
            # Compute returns and advantages
            values = []
            for state in states[:-1]:
                value = self.value(state)
                values.append(value)
            
            # Bootstrap value
            final_value = self.value(states[-1])
            values.append(final_value)
        
        # GAE - compute under no_grad to avoid shared autograd history
        with torch.no_grad():
            advantages = []
            returns = []
            gae = 0.0
            
            for t in reversed(range(len(rewards))):
                v_t = values[t].detach()
                v_tp1 = values[t + 1].detach()
                delta = rewards[t] + self.discount * v_tp1 - v_t
                gae = delta + self.discount * self.lambda_gae * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + v_t)
            
            advantages = torch.stack(advantages)
            returns = torch.stack(returns)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Recompute losses outside autocast for combined backward
        with autocast(**autocast_kwargs):
            # Actor loss
            log_probs = torch.stack(log_probs)
            actor_loss = -(log_probs * advantages).mean()
            
            # Entropy bonus
            entropy = -(log_probs * log_probs.exp()).mean()
            actor_loss -= self.entropy_weight * entropy
            
            # Critic loss
            values = torch.stack(values[:-1])
            critic_loss = F.mse_loss(values, returns)
            
            # Combined loss - single backward for both optimizers
            total_loss = actor_loss * self.actor_coef + critic_loss * self.critic_coef
        
        # Zero gradients
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        
        # Single backward pass with AMP
        self.scaler_ac.scale(total_loss).backward()
        
        # Unscale and clip actor gradients
        self.scaler_ac.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        
        # Unscale and clip critic gradients
        self.scaler_ac.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)
        
        # Step both optimizers
        self.scaler_ac.step(self.actor_optimizer)
        self.scaler_ac.step(self.critic_optimizer)
        self.scaler_ac.update()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_return': returns.mean().item()
        }
    
    def _train_intrinsic(self) -> Dict[str, float]:
        """Train intrinsic motivation (RND)"""
        # Sample batch
        episodes, _, _ = self.replay_buffer.sample(self.batch_size)
        
        # Get random states
        states_list = []
        for episode in episodes:
            idx = random.randint(0, len(episode['observations']) - 1)
            obs = torch.FloatTensor(episode['observations'][idx]).to(self.device)
            action = torch.zeros(3, device=self.device)
            prev_state = self.world_model.initial_state(1, self.device)
            state = self.world_model.observe(obs.unsqueeze(0), action.unsqueeze(0), prev_state)
            states_list.append(state)
        
        # Stack states
        batch_state = {
            'stoch': torch.cat([s['stoch'] for s in states_list], dim=0),
            'deter': torch.cat([s['deter'] for s in states_list], dim=0)
        }
        
        # Train RND predictor
        latent = torch.cat([batch_state['stoch'], batch_state['deter']], dim=-1)
        
        with torch.no_grad():
            target_features = self.intrinsic.target_net(latent)
        
        predicted_features = self.intrinsic.predictor_net(latent)
        
        intrinsic_loss = F.mse_loss(predicted_features, target_features)
        
        self.intrinsic_optimizer.zero_grad()
        intrinsic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.intrinsic.predictor_net.parameters(), self.grad_clip)
        self.intrinsic_optimizer.step()
        
        return {
            'intrinsic_loss': intrinsic_loss.item()
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'world_model': self.world_model.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        
        if hasattr(self, 'intrinsic_optimizer'):
            checkpoint['intrinsic'] = self.intrinsic.state_dict()
            checkpoint['intrinsic_optimizer'] = self.intrinsic_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if 'intrinsic' in checkpoint and hasattr(self, 'intrinsic_optimizer'):
            self.intrinsic.load_state_dict(checkpoint['intrinsic'])
            self.intrinsic_optimizer.load_state_dict(checkpoint['intrinsic_optimizer'])
    
    # ========================================================================
    # Feature 6: Federated MARL
    # ========================================================================
    
    def get_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get trainable parameters for federated averaging.
        Feature 6: Enables decentralized multi-robot learning.
        
        Returns:
            weights: Dictionary of network parameters (on CPU)
        """
        weights = {
            'policy': {k: v.cpu().clone() for k, v in self.policy.state_dict().items()},
            'value': {k: v.cpu().clone() for k, v in self.value.state_dict().items()},
        }
        
        # Optionally include world model (can be large)
        if self.config['learning'].get('federate_world_model', False):
            weights['world_model'] = {
                k: v.cpu().clone() for k, v in self.world_model.state_dict().items()
            }
        
        return weights
    
    def set_weights(self, weights: Dict[str, Dict[str, torch.Tensor]]):
        """
        Set trainable parameters from federated averaging.
        Feature 6: Updates local agent with averaged parameters.
        
        Args:
            weights: Dictionary of network parameters
        """
        if 'policy' in weights:
            self.policy.load_state_dict({
                k: v.to(self.device) for k, v in weights['policy'].items()
            })
        
        if 'value' in weights:
            self.value.load_state_dict({
                k: v.to(self.device) for k, v in weights['value'].items()
            })
        
        if 'world_model' in weights:
            self.world_model.load_state_dict({
                k: v.to(self.device) for k, v in weights['world_model'].items()
            })
    
    def federated_round(self, peer_trainers: List['QREATrainer'], 
                       aggregation: str = 'average') -> Dict[str, float]:
        """
        Perform one round of federated learning (FedAvg).
        Feature 6: Averages parameters across multiple robot agents.
        
        Args:
            peer_trainers: List of other QREATrainer instances
            aggregation: Method ('average', 'weighted_average')
            
        Returns:
            metrics: Aggregation statistics
        """
        # Collect weights from all peers
        all_weights = [self.get_weights()]
        for peer in peer_trainers:
            all_weights.append(peer.get_weights())
        
        # Aggregate
        if aggregation == 'average':
            averaged_weights = self._average_weights(all_weights)
        elif aggregation == 'weighted_average':
            # Could weight by performance, data size, etc.
            # For now, just use uniform averaging
            averaged_weights = self._average_weights(all_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Update local weights
        self.set_weights(averaged_weights)
        
        # Compute divergence metrics
        divergence = self._compute_parameter_divergence(all_weights)
        
        return {
            'num_peers': len(peer_trainers),
            'param_divergence': divergence,
        }
    
    def _average_weights(self, weights_list: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Average parameters across multiple agents (FedAvg).
        
        Args:
            weights_list: List of weight dictionaries
            
        Returns:
            averaged: Averaged parameters
        """
        n = len(weights_list)
        averaged = {}
        
        # Get network names from first agent
        for net_name in weights_list[0].keys():
            averaged[net_name] = {}
            
            # Average each parameter
            for param_name in weights_list[0][net_name].keys():
                # Stack parameters from all agents
                param_stack = torch.stack([
                    w[net_name][param_name] for w in weights_list
                ], dim=0)
                
                # Compute mean
                averaged[net_name][param_name] = param_stack.mean(dim=0)
        
        return averaged
    
    def _compute_parameter_divergence(self, weights_list: List[Dict]) -> float:
        """
        Compute average pairwise parameter divergence.
        Useful for monitoring federated training convergence.
        
        Args:
            weights_list: List of weight dictionaries
            
        Returns:
            divergence: Average L2 distance between parameters
        """
        if len(weights_list) < 2:
            return 0.0
        
        total_divergence = 0.0
        num_pairs = 0
        
        # Pairwise divergence
        for i in range(len(weights_list)):
            for j in range(i + 1, len(weights_list)):
                # Compute L2 distance for policy parameters
                if 'policy' in weights_list[i] and 'policy' in weights_list[j]:
                    for param_name in weights_list[i]['policy'].keys():
                        p1 = weights_list[i]['policy'][param_name]
                        p2 = weights_list[j]['policy'][param_name]
                        
                        diff = (p1 - p2).pow(2).sum().sqrt()
                        total_divergence += diff.item()
                
                num_pairs += 1
        
        return total_divergence / max(num_pairs, 1)
    
    def gossip_update(self, peer_trainer: 'QREATrainer', mix_ratio: float = 0.5):
        """
        Gossip-based parameter mixing with one neighbor.
        Feature 6: Lightweight alternative to full FedAvg.
        
        Args:
            peer_trainer: Single peer to exchange with
            mix_ratio: How much to mix (0.5 = equal mixing)
        """
        my_weights = self.get_weights()
        peer_weights = peer_trainer.get_weights()
        
        mixed_weights = {}
        
        for net_name in my_weights.keys():
            mixed_weights[net_name] = {}
            
            for param_name in my_weights[net_name].keys():
                # Linear interpolation
                mixed = (
                    mix_ratio * my_weights[net_name][param_name] +
                    (1 - mix_ratio) * peer_weights[net_name][param_name]
                )
                mixed_weights[net_name][param_name] = mixed
        
        # Update both agents
        self.set_weights(mixed_weights)
        peer_trainer.set_weights(mixed_weights)
