"""
QREA Robot Agent
World model, policy, value, and intrinsic motivation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Dict, Tuple, Optional, List


class RSSMWorldModel(nn.Module):
    """
    Recurrent State Space Model (Dreamer-style)
    Models: p(s_t+1 | s_t, a_t), p(o_t | s_t), p(r_t | s_t)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        wm_cfg = config['agent']['world_model']
        self.latent_dim = wm_cfg['latent_dim']
        self.hidden_dim = wm_cfg['hidden_dim']
        self.stochastic_dim = wm_cfg['stochastic_dim']
        self.deterministic_dim = wm_cfg['deterministic_dim']
        
        # Observation encoder: obs -> latent
        self.encoder = ObservationEncoder(
            obs_dim=config['observation_dim'],
            hidden_dims=wm_cfg['encoder']['hidden_dims'],
            latent_dim=self.latent_dim
        )
        
        # Recurrent model: (s_t, a_t) -> h_t+1
        self.rnn = nn.GRUCell(
            self.stochastic_dim + config['action_dim'],
            self.deterministic_dim
        )
        
        # Transition model: h_t+1 -> p(s_t+1)
        self.transition = nn.Sequential(
            nn.Linear(self.deterministic_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim)  # mean, std
        )
        
        # Representation model: (h_t, o_t) -> p(s_t | h_t, o_t)
        self.representation = nn.Sequential(
            nn.Linear(self.deterministic_dim + self.latent_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim)
        )
        
        # Observation decoder: s_t -> o_t
        self.decoder = ObservationDecoder(
            latent_dim=self.stochastic_dim + self.deterministic_dim,
            hidden_dims=wm_cfg['decoder']['hidden_dims'],
            obs_dim=config['observation_dim']
        )
        
        # Reward predictor: s_t -> r_t
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.stochastic_dim + self.deterministic_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Discount predictor: s_t -> gamma_t
        self.discount_predictor = nn.Sequential(
            nn.Linear(self.stochastic_dim + self.deterministic_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get initial RNN state"""
        return {
            'deter': torch.zeros(batch_size, self.deterministic_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stochastic_dim, device=device),
            'logit': torch.zeros(batch_size, 2 * self.stochastic_dim, device=device)
        }
    
    def observe(self, obs: torch.Tensor, action: torch.Tensor, 
                prev_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update state given observation
        Returns posterior state: p(s_t | o_t, s_t-1, a_t-1)
        """
        # Encode observation
        embed = self.encoder(obs)
        
        # Deterministic path
        prev_stoch = prev_state['stoch']
        prev_deter = prev_state['deter']
        x = torch.cat([prev_stoch, action], dim=-1)
        deter = self.rnn(x, prev_deter)
        
        # Stochastic path (posterior)
        x = torch.cat([deter, embed], dim=-1)
        logit = self.representation(x)
        mean, std = torch.chunk(logit, 2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch_dist = dist.Normal(mean, std)
        stoch = stoch_dist.rsample()
        
        return {
            'deter': deter,
            'stoch': stoch,
            'logit': logit,
            'mean': mean,
            'std': std
        }
    
    def imagine(self, action: torch.Tensor, 
                prev_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict next state without observation (prior)
        Returns: p(s_t+1 | s_t, a_t)
        """
        prev_stoch = prev_state['stoch']
        prev_deter = prev_state['deter']
        
        # Deterministic path
        x = torch.cat([prev_stoch, action], dim=-1)
        deter = self.rnn(x, prev_deter)
        
        # Stochastic path (prior)
        logit = self.transition(deter)
        mean, std = torch.chunk(logit, 2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch_dist = dist.Normal(mean, std)
        stoch = stoch_dist.rsample()
        
        return {
            'deter': deter,
            'stoch': stoch,
            'logit': logit,
            'mean': mean,
            'std': std
        }
    
    def decode_obs(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode observation from state"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        return self.decoder(latent)
    
    def predict_reward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict reward from state"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        return self.reward_predictor(latent)
    
    def predict_discount(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict discount factor from state"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        return self.discount_predictor(latent)


class ObservationEncoder(nn.Module):
    """Encode observations to latent space"""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ObservationDecoder(nn.Module):
    """Decode latent states to observations"""
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], obs_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, obs_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class Policy(nn.Module):
    """Stochastic policy network"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        policy_cfg = config['agent']['policy']
        latent_dim = (config['agent']['world_model']['stochastic_dim'] + 
                     config['agent']['world_model']['deterministic_dim'])
        
        self.action_dim = policy_cfg['action_dim']
        self.log_std_min = policy_cfg['log_std_min']
        self.log_std_max = policy_cfg['log_std_max']
        
        # Policy network
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in policy_cfg['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        # Output heads
        self.mean = nn.Linear(prev_dim, self.action_dim)
        self.log_std = nn.Linear(prev_dim, self.action_dim)
    
    def forward(self, state: Dict[str, torch.Tensor], 
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        Returns: action, log_prob
        """
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        features = self.net(latent)
        
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action)
        else:
            action_dist = dist.Normal(mean, std)
            action_sample = action_dist.rsample()
            action = torch.tanh(action_sample)
            
            # Compute log prob with tanh correction
            log_prob = action_dist.log_prob(action_sample)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob


class ValueFunction(nn.Module):
    """State value function"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        value_cfg = config['agent']['value']
        latent_dim = (config['agent']['world_model']['stochastic_dim'] + 
                     config['agent']['world_model']['deterministic_dim'])
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in value_cfg['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict state value"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        return self.net(latent)


class IntrinsicMotivation(nn.Module):
    """
    Intrinsic motivation system
    - Novelty (RND)
    - Competence (learning progress)
    - Empowerment (mutual information)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        intrinsic_cfg = config['agent']['intrinsic']
        latent_dim = (config['agent']['world_model']['stochastic_dim'] + 
                     config['agent']['world_model']['deterministic_dim'])
        
        self.novelty_weight = intrinsic_cfg['novelty']['weight']
        self.competence_weight = intrinsic_cfg['competence']['weight']
        self.empowerment_weight = intrinsic_cfg['empowerment']['weight']
        
        # Random Network Distillation for novelty
        if intrinsic_cfg['novelty']['enabled']:
            network_dim = intrinsic_cfg['novelty']['network_dim']
            
            # Target network (fixed)
            self.target_net = nn.Sequential(
                nn.Linear(latent_dim, network_dim),
                nn.ReLU(),
                nn.Linear(network_dim, network_dim),
                nn.ReLU(),
                nn.Linear(network_dim, network_dim)
            )
            
            # Predictor network (learned)
            self.predictor_net = nn.Sequential(
                nn.Linear(latent_dim, network_dim),
                nn.ReLU(),
                nn.Linear(network_dim, network_dim),
                nn.ReLU(),
                nn.Linear(network_dim, network_dim)
            )
            
            # Freeze target
            for param in self.target_net.parameters():
                param.requires_grad = False
        
        # Competence: track prediction errors
        self.competence_enabled = intrinsic_cfg['competence']['enabled']
        self.error_history = []
        self.history_length = intrinsic_cfg['competence']['history_length']
        
        # Empowerment: estimate MI(S_{t+h}; A_t | S_t)
        self.empowerment_enabled = intrinsic_cfg['empowerment']['enabled']
        self.planning_horizon = intrinsic_cfg['empowerment']['planning_horizon']
    
    def compute_novelty(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute novelty bonus using RND"""
        latent = torch.cat([state['stoch'], state['deter']], dim=-1)
        
        with torch.no_grad():
            target_features = self.target_net(latent)
        
        predicted_features = self.predictor_net(latent)
        
        # Novelty = prediction error
        novelty = F.mse_loss(predicted_features, target_features, reduction='none')
        novelty = novelty.sum(dim=-1, keepdim=True)
        
        return novelty * self.novelty_weight
    
    def compute_competence(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """Compute competence bonus (learning progress)"""
        if not self.competence_enabled:
            return torch.zeros_like(prediction_error)
        
        # Store error
        self.error_history.append(prediction_error.detach().mean().item())
        if len(self.error_history) > self.history_length:
            self.error_history.pop(0)
        
        if len(self.error_history) < 2:
            return torch.zeros_like(prediction_error)
        
        # Learning progress = reduction in error
        recent_error = np.mean(self.error_history[-10:])
        old_error = np.mean(self.error_history[:10])
        progress = max(0, old_error - recent_error)
        
        competence = torch.full_like(prediction_error, progress * self.competence_weight)
        
        return competence
    
    def compute_empowerment(self, states: List[Dict[str, torch.Tensor]], 
                          actions: torch.Tensor) -> torch.Tensor:
        """
        Compute empowerment (simplified)
        Measures how much control agent has over future
        """
        if not self.empowerment_enabled:
            return torch.zeros(actions.shape[0], 1, device=actions.device)
        
        # Simplified: variance in imagined trajectories
        latents = []
        for state in states:
            latent = torch.cat([state['stoch'], state['deter']], dim=-1)
            latents.append(latent)
        
        latents = torch.stack(latents, dim=1)  # [batch, horizon, latent_dim]
        
        # Empowerment â‰ˆ state variance induced by actions
        state_variance = latents.var(dim=1).sum(dim=-1, keepdim=True)
        
        empowerment = state_variance * self.empowerment_weight
        
        return empowerment
    
    def compute_total_intrinsic_reward(self, state: Dict[str, torch.Tensor],
                                      prediction_error: torch.Tensor,
                                      imagined_states: Optional[List] = None,
                                      actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute total intrinsic reward"""
        reward = torch.zeros(state['stoch'].shape[0], 1, device=state['stoch'].device)
        
        # Novelty
        if self.novelty_weight > 0:
            reward += self.compute_novelty(state)
        
        # Competence
        if self.competence_weight > 0:
            reward += self.compute_competence(prediction_error)
        
        # Empowerment
        if self.empowerment_weight > 0 and imagined_states is not None:
            reward += self.compute_empowerment(imagined_states, actions)
        
        return reward


class QREARobotAgent:
    """Complete QREA agent for warehouse robot"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Add derived dimensions - calculate dynamically based on num_robots
        num_robots = config['environment']['num_robots']
        lidar_rays = config['environment']['sensors']['lidar']['num_rays']
        # Observation: robot_state(10) + lidar(rays) + packages(400) + stations(12) + other_robots((num_robots-1)*8)
        obs_space = 10 + lidar_rays + 400 + 12 + (num_robots - 1) * 8
        config['observation_dim'] = obs_space
        config['action_dim'] = 3
        
        # Components
        self.world_model = RSSMWorldModel(config).to(device)
        self.policy = Policy(config).to(device)
        self.value = ValueFunction(config).to(device)
        self.intrinsic = IntrinsicMotivation(config).to(device)
        
        # State
        self.state = None
        self.reset()
    
    def reset(self):
        """Reset agent state"""
        self.state = self.world_model.initial_state(1, self.device)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Update state with observation (no action for first step)
            if self.state is None or not hasattr(self, 'last_action'):
                action_tensor = torch.zeros(1, 3, device=self.device)
            else:
                action_tensor = self.last_action
            
            self.state = self.world_model.observe(obs_tensor, action_tensor, self.state)
            
            # Sample action from policy (outputs in [-1, 1] via tanh)
            action, _ = self.policy(self.state, deterministic)
            
            # Map actions to proper ranges
            # Policy outputs are in [-1, 1] from tanh activation
            mapped_action = action.clone()
            
            # Linear velocity: map [-1, 1] -> [-max_speed, max_speed]
            max_speed = self.config['environment']['robot']['max_speed']
            mapped_action[0, 0] = action[0, 0] * max_speed
            
            # Angular velocity: map [-1, 1] -> [-max_angular_velocity, max_angular_velocity]
            max_angular_vel = self.config['environment']['robot']['max_angular_velocity']
            mapped_action[0, 1] = action[0, 1] * max_angular_vel
            
            # Gripper: map [-1, 1] -> [0, 1] for proper pickup triggering
            # Using sigmoid-like mapping: (tanh_output + 1) / 2
            # This ensures gripper > 0.5 threshold can be reliably learned
            mapped_action[0, 2] = (action[0, 2] + 1.0) / 2.0
            
        self.last_action = mapped_action
        
        return mapped_action.cpu().numpy()[0]
    
    def get_networks(self) -> Dict[str, nn.Module]:
        """Get all networks for training"""
        return {
            'world_model': self.world_model,
            'policy': self.policy,
            'value': self.value,
            'intrinsic': self.intrinsic
        }
