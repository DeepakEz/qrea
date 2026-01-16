"""
Spatial Activity Fields for Warehouse Environment
=================================================

Implements spatial grid-based activity tracking for multi-robot coordination.

NOTE: The original names ("consciousness field", "resonance field", "genetic field")
were misleading pseudoscience terminology. These are simply:
- activity_field: Accumulates robot activity at spatial locations
- interaction_field: Tracks pairwise robot proximity/similarity
- memory_field: Long-term activity accumulation (slow decay)

These fields provide spatial context to the RL agent but have no connection
to actual consciousness, quantum resonance, or genetics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Symbol:
    """Pattern symbol in UPRT"""
    id: int
    embedding: torch.Tensor
    activation: float
    confidence: float
    emergence_time: int
    usage_count: int


class UPRTField(nn.Module):
    """
    Spatial Activity Field for multi-robot warehouse.

    Maintains grid-based fields that track robot activity over space and time.
    Fields are updated via diffusion and decay dynamics.

    NOTE: Despite the class name, this is NOT based on any real physics theory.
    The fields are simply spatial accumulators used as auxiliary observations.
    """
    
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        uprt_cfg = config['uprt']

        # Field parameters
        self.grid_resolution = uprt_cfg['field']['grid_resolution']
        self.diffusion_coeff = uprt_cfg['field']['diffusion_coeff']
        self.decay_rate = uprt_cfg['field']['decay_rate']
        self.update_rate = uprt_cfg['field']['update_rate']

        # World grid size from config - registered as buffer for proper device handling
        self.register_buffer('world_size', torch.tensor(
            config['environment']['grid_size'], dtype=torch.float32
        ))

        # Field grids [H, W, channels] - registered as buffers for proper device handling
        self.register_buffer('activity_field', torch.zeros(
            *self.grid_resolution, 16, requires_grad=False
        ))
        self.register_buffer('interaction_field', torch.zeros(
            *self.grid_resolution, 16, requires_grad=False
        ))
        self.register_buffer('memory_field', torch.zeros(
            *self.grid_resolution, 32, requires_grad=False
        ))
        
        # Symbol system
        self.symbol_dim = uprt_cfg['symbols']['embedding_dim']
        self.num_prototypes = uprt_cfg['symbols']['num_prototypes']
        self.detection_threshold = uprt_cfg['symbols']['detection_threshold']
        self.emergence_threshold = uprt_cfg['symbols']['emergence_threshold']
        
        # Symbol prototypes (learnable)
        self.symbol_prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.symbol_dim) * 0.1
        )
        
        # Pattern encoder: observation -> pattern embedding
        # Calculate obs_dim dynamically based on num_robots
        num_robots = config['environment']['num_robots']
        lidar_rays = config['environment']['sensors']['lidar']['num_rays']
        # Observation: robot_state(10) + carrying_dest(4) + lidar(rays) + packages(400) + stations(12) + other_robots((num_robots-1)*8)
        # NOTE: +4 for carrying destination added in warehouse_env.py observation fix
        obs_dim = 10 + 4 + lidar_rays + 400 + 12 + (num_robots - 1) * 8
        self.pattern_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.symbol_dim)
        )
        
        # Similarity computation network for interaction field
        self.similarity_net = nn.Sequential(
            nn.Linear(self.symbol_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Active symbols
        self.symbols: List[Symbol] = []
        self.next_symbol_id = 0
        self.timestep = 0
    
    def update_fields(self, robot_positions: torch.Tensor, 
                     robot_activities: torch.Tensor,
                     dt: float = 0.1):
        """
        Update field dynamics
        
        Args:
            robot_positions: [N, 2] positions in world coords
            robot_activities: [N, D] activity vectors
        """
        # Convert world coords to grid coords using config world_size
        world_size = self.world_size.to(robot_positions.device)
        grid_x = (robot_positions[:, 0] / world_size[0] * self.grid_resolution[0]).long()
        grid_y = (robot_positions[:, 1] / world_size[1] * self.grid_resolution[1]).long()
        
        grid_x = torch.clamp(grid_x, 0, self.grid_resolution[0] - 1)
        grid_y = torch.clamp(grid_y, 0, self.grid_resolution[1] - 1)
        
        # Update activity field (spatial activity accumulator) - vectorized
        flat_idx = grid_x * self.grid_resolution[1] + grid_y  # (N,)
        field_flat = self.activity_field.view(-1, self.activity_field.shape[-1])  # (H*W, 16)
        activity_contrib = robot_activities[:, :16] * dt  # (N, 16)
        field_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, 16), activity_contrib)
        self.activity_field = field_flat.view(*self.activity_field.shape)
        
        # Diffusion
        self.activity_field = self._apply_diffusion(
            self.activity_field, self.diffusion_coeff, dt
        )
        
        # Decay
        self.activity_field *= (1 - self.decay_rate * dt)
        
        # Update interaction field (pairwise proximity/similarity)
        self._update_interaction_field(robot_positions, robot_activities, dt)

        # Update memory field (long-term activity accumulation)
        self._update_memory_field(robot_positions, robot_activities, dt)
        
        self.timestep += 1
    
    def _apply_diffusion(self, field: torch.Tensor, coeff: float, dt: float) -> torch.Tensor:
        """Apply diffusion to field using finite differences"""
        # Simple 4-neighbor diffusion
        dx = 1.0
        dy = 1.0
        
        # Pad field for boundary conditions
        padded = F.pad(field.permute(2, 0, 1).unsqueeze(0), 
                      (1, 1, 1, 1), mode='replicate')
        
        # Compute Laplacian
        laplacian = (
            padded[:, :, 2:, 1:-1] + padded[:, :, :-2, 1:-1] +  # x neighbors
            padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, :-2] -  # y neighbors
            4 * padded[:, :, 1:-1, 1:-1]
        ) / (dx * dy)
        
        # Apply diffusion
        diffused = field.permute(2, 0, 1) + coeff * dt * laplacian.squeeze(0)
        
        return diffused.permute(1, 2, 0)
    
    def _update_interaction_field(self, positions: torch.Tensor,
                               activities: torch.Tensor, dt: float):
        """Update interaction field based on pairwise robot proximity/similarity (vectorized)"""
        N = positions.shape[0]
        if N < 2:
            # No pairs to process
            self.interaction_field = self._apply_diffusion(
                self.interaction_field, self.diffusion_coeff * 2, dt
            )
            self.interaction_field *= (1 - self.decay_rate * 0.5 * dt)
            return

        # Vectorized pairwise distance computation - O(N²) but in parallel
        dists = torch.cdist(positions, positions)  # (N, N)

        # Get upper triangle indices (pairs i < j)
        i_idx, j_idx = torch.triu_indices(N, N, offset=1)

        # Filter pairs within resonance range
        pair_dists = dists[i_idx, j_idx]
        mask = pair_dists < 10.0

        if mask.sum() > 0:
            # Get valid pair indices
            valid_i = i_idx[mask]
            valid_j = j_idx[mask]

            # Vectorized resonance computation
            act_i = activities[valid_i]  # (num_pairs, D)
            act_j = activities[valid_j]  # (num_pairs, D)

            # Normalize activities
            act_i_norm = act_i / (torch.norm(act_i, dim=1, keepdim=True) + 1e-8)
            act_j_norm = act_j / (torch.norm(act_j, dim=1, keepdim=True) + 1e-8)

            # Cosine similarity on first 16 dims
            similarity = (act_i_norm[:, :16] * act_j_norm[:, :16]).sum(dim=1)
            resonances = torch.sigmoid(similarity * 5.0)  # (num_pairs,)

            # Compute midpoints and grid coords
            mid_pos = (positions[valid_i] + positions[valid_j]) / 2  # (num_pairs, 2)
            world_size = self.world_size.to(positions.device)
            gx = (mid_pos[:, 0] / world_size[0] * self.grid_resolution[0]).long()
            gy = (mid_pos[:, 1] / world_size[1] * self.grid_resolution[1]).long()
            gx = torch.clamp(gx, 0, self.grid_resolution[0] - 1)
            gy = torch.clamp(gy, 0, self.grid_resolution[1] - 1)

            # Scatter add resonances to field (truly vectorized)
            # Convert 2D grid indices to 1D for scatter_add_
            flat_idx = gx * self.grid_resolution[1] + gy  # (num_pairs,)
            # Reshape field for scatter, then reshape back
            field_flat = self.interaction_field.view(-1, self.interaction_field.shape[-1])  # (H*W, C)
            # Expand resonances to match field channels
            resonance_contrib = (resonances * dt).unsqueeze(-1).expand(-1, field_flat.shape[-1])  # (num_pairs, C)
            # Scatter add
            field_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, field_flat.shape[-1]), resonance_contrib)
            self.interaction_field = field_flat.view(*self.interaction_field.shape)

        # Diffuse and decay
        self.interaction_field = self._apply_diffusion(
            self.interaction_field, self.diffusion_coeff * 2, dt
        )
        self.interaction_field *= (1 - self.decay_rate * 0.5 * dt)
    
    def _update_memory_field(self, positions: torch.Tensor,
                             activities: torch.Tensor, dt: float):
        """Update memory field (long-term activity accumulation with slow decay)"""
        # Memory field accumulates activity patterns over time (vectorized)
        world_size = self.world_size.to(positions.device)
        N = positions.shape[0]

        # Compute grid coordinates (vectorized)
        gx = (positions[:, 0] / world_size[0] * self.grid_resolution[0]).long()
        gy = (positions[:, 1] / world_size[1] * self.grid_resolution[1]).long()
        gx = torch.clamp(gx, 0, self.grid_resolution[0] - 1)
        gy = torch.clamp(gy, 0, self.grid_resolution[1] - 1)

        # Scatter add activities to field
        flat_idx = gx * self.grid_resolution[1] + gy  # (N,)
        field_flat = self.memory_field.view(-1, self.memory_field.shape[-1])  # (H*W, 32)
        activity_contrib = activities[:, :32] * dt * 0.1  # (N, 32)
        field_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, 32), activity_contrib)
        self.memory_field = field_flat.view(*self.memory_field.shape)

        # Slower decay for long-term memory
        self.memory_field *= (1 - self.decay_rate * 0.1 * dt)
    
    def _compute_similarity(self, act1: torch.Tensor, act2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two activity patterns"""
        # Normalize
        act1_norm = act1 / (torch.norm(act1) + 1e-8)
        act2_norm = act2 / (torch.norm(act2) + 1e-8)

        # Cosine similarity
        similarity = torch.dot(act1_norm[:16], act2_norm[:16])

        # Transform to [0,1] range with sigmoid
        sim_score = torch.sigmoid(similarity * 5.0)  # Sharpen

        return sim_score.unsqueeze(-1).expand(16)
    
    def detect_patterns(self, observations: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Detect patterns in observations and match to symbols
        
        Args:
            observations: [batch, obs_dim]
        
        Returns:
            embeddings: [batch, symbol_dim]
            matched_symbols: List of symbol IDs
        """
        # Encode observations to pattern space
        embeddings = self.pattern_encoder(observations)
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarity to prototypes
        similarities = torch.matmul(embeddings, self.symbol_prototypes.t())
        
        # Detect matches
        max_sims, best_protos = similarities.max(dim=-1)
        
        matched_symbols = []
        for i in range(embeddings.shape[0]):
            if max_sims[i] > self.detection_threshold:
                matched_symbols.append(best_protos[i].item())
            else:
                matched_symbols.append(-1)  # No match
        
        return embeddings, matched_symbols
    
    def emerge_new_symbol(self, embedding: torch.Tensor, 
                         confidence: float) -> Optional[Symbol]:
        """Create new symbol if pattern is novel and consistent"""
        if confidence < self.emergence_threshold:
            return None
        
        # Check if truly novel
        with torch.no_grad():
            similarities = torch.matmul(
                F.normalize(embedding, dim=-1),
                F.normalize(self.symbol_prototypes, dim=-1).t()
            )
            
            if similarities.max() > 0.9:  # Too similar to existing
                return None
        
        # Create new symbol
        symbol = Symbol(
            id=self.next_symbol_id,
            embedding=embedding.detach().clone(),
            activation=1.0,
            confidence=confidence,
            emergence_time=self.timestep,
            usage_count=1
        )
        
        self.symbols.append(symbol)
        self.next_symbol_id += 1
        
        # Update prototypes (add new or replace least used)
        if len(self.symbols) <= self.num_prototypes:
            idx = len(self.symbols) - 1
        else:
            # Replace least used symbol
            idx = min(range(len(self.symbols)), 
                     key=lambda i: self.symbols[i].usage_count)
        
        with torch.no_grad():
            self.symbol_prototypes[idx] = embedding
        
        return symbol
    
    def compute_field_coherence(self) -> float:
        """Compute global field structure (variance-based measure)"""
        # Measure how organized/structured the fields are

        # Activity field variance
        a_field_flat = self.activity_field.flatten()
        a_variance = a_field_flat.var().item()

        # Interaction field variance
        i_field_flat = self.interaction_field.flatten()
        i_variance = i_field_flat.var().item()

        # High variance = more structure (localized activity rather than uniform)
        structure = (a_variance + i_variance) / 2

        return structure
    
    def get_field_at_position(self, position: np.ndarray) -> Dict[str, torch.Tensor]:
        """Get field values at world position"""
        world_size = self.world_size.cpu().numpy()
        gx = int(position[0] / world_size[0] * self.grid_resolution[0])
        gy = int(position[1] / world_size[1] * self.grid_resolution[1])

        gx = np.clip(gx, 0, self.grid_resolution[0] - 1)
        gy = np.clip(gy, 0, self.grid_resolution[1] - 1)
        
        return {
            'activity': self.activity_field[gx, gy].clone(),
            'interaction': self.interaction_field[gx, gy].clone(),
            'memory': self.memory_field[gx, gy].clone()
        }
    
    def visualize_fields(self) -> Dict[str, np.ndarray]:
        """Get field visualizations"""
        return {
            'activity': self.activity_field.norm(dim=-1).cpu().numpy(),
            'interaction': self.interaction_field.norm(dim=-1).cpu().numpy(),
            'memory': self.memory_field.norm(dim=-1).cpu().numpy()
        }
    
    def get_symbol_statistics(self) -> Dict:
        """Get statistics about emerged symbols"""
        if not self.symbols:
            return {
                'num_symbols': 0,
                'avg_usage': 0.0,
                'diversity': 0.0,
                'active_symbols': 0
            }

        # With only 1 symbol, diversity is undefined (no pairs to compare)
        if len(self.symbols) == 1:
            return {
                'num_symbols': 1,
                'avg_usage': float(self.symbols[0].usage_count),
                'diversity': 0.0,  # No diversity with single symbol
                'active_symbols': 1 if self.symbols[0].activation > 0.1 else 0
            }

        usages = [s.usage_count for s in self.symbols]

        # Diversity: how different are the symbols
        embeddings = torch.stack([s.embedding for s in self.symbols])
        similarities = torch.matmul(
            F.normalize(embeddings, dim=-1),
            F.normalize(embeddings, dim=-1).t()
        )
        # Exclude diagonal - ensure mask is on same device
        mask = torch.eye(len(self.symbols), dtype=torch.bool, device=embeddings.device)
        avg_similarity = similarities[~mask].mean().item()
        diversity = 1 - avg_similarity
        
        return {
            'num_symbols': len(self.symbols),
            'avg_usage': np.mean(usages),
            'diversity': diversity,
            'active_symbols': sum(1 for s in self.symbols if s.activation > 0.1)
        }
    
    def reset(self):
        """Reset fields"""
        self.activity_field.zero_()
        self.interaction_field.zero_()
        self.memory_field.zero_()
        self.symbols = []
        self.timestep = 0


class WarehouseUPRT:
    """UPRT field manager for warehouse environment"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        self.field = UPRTField(config).to(device)
        
    def update(self, robot_states: List[Dict], observations: List[np.ndarray], dt: float = 0.1):
        """Update fields based on robot states"""
        if not robot_states:
            return
        
        # Extract positions and activities
        positions = []
        activities = []
        
        for robot in robot_states:
            positions.append([robot['position'][0], robot['position'][1]])
            
            # Activity = combination of velocity, battery, task status
            activity = np.zeros(64)
            activity[0] = robot.get('velocity', 0.0)
            activity[1] = robot.get('battery', 0.0) / 1000.0
            activity[2] = 1.0 if robot.get('carrying', False) else 0.0
            activity[3] = robot.get('packages_delivered', 0) / 10.0
            
            # Add noise for diversity
            activity += np.random.randn(64) * 0.01
            
            activities.append(activity)
        
        positions = torch.FloatTensor(np.array(positions)).to(self.device)
        activities = torch.FloatTensor(np.array(activities)).to(self.device)
        
        # Update field dynamics
        self.field.update_fields(positions, activities, dt)
        
        # Detect patterns
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        embeddings, matched_symbols = self.field.detect_patterns(obs_tensor)
        
        # Try to emerge new symbols
        for i, (emb, match) in enumerate(zip(embeddings, matched_symbols)):
            if match == -1:  # No match found
                # Check if should emerge
                confidence = torch.sigmoid(emb.norm() - 0.5).item()
                self.field.emerge_new_symbol(emb, confidence)
    
    def get_field_observation(self, position: np.ndarray) -> np.ndarray:
        """Get field values at position for agent observation"""
        fields = self.field.get_field_at_position(position)
        
        # Concatenate field values
        obs = torch.cat([
            fields['activity'],
            fields['interaction'][:8],  # Subsample
            fields['memory'][:16]       # Subsample
        ])
        
        return obs.cpu().numpy()
    
    def get_metrics(self) -> Dict:
        """Get UPRT metrics"""
        return {
            'field_coherence': self.field.compute_field_coherence(),
            **self.field.get_symbol_statistics()
        }
    
    def reset(self):
        """Reset fields"""
        self.field.reset()
