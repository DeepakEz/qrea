"""
Neural Doctrine Engine
======================

Neural network-based doctrine formation that replaces statistical heuristics
with learned pattern recognition. Uses sequence models to detect coordination
patterns and generate doctrine recommendations.

This is Phase 4 upgrade - use after validating simpler statistical doctrines.

Usage:
    from holographic.neural_doctrine_engine import NeuralDoctrineEngine

    engine = NeuralDoctrineEngine(obs_dim=64)
    engine.add_observation(context, outcome)
    doctrine = engine.attempt_doctrine_formation()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import random


class DoctrineType(Enum):
    """Types of learned doctrines"""
    COORDINATION = "coordination"  # Multi-agent coordination rules
    EFFICIENCY = "efficiency"      # Resource optimization
    SAFETY = "safety"              # Collision avoidance
    ALLOCATION = "allocation"      # Task allocation strategies


@dataclass
class Doctrine:
    """
    A learned coordination doctrine/strategy.

    Doctrines are empirically learned rules that emerge from experience.
    Example: "When queue > 5 packages, prioritize nearest tasks first"
    """
    id: str
    type: DoctrineType
    trigger: Dict[str, Any]  # Conditions that activate this doctrine
    action: Dict[str, Any]   # What to do when triggered
    confidence: float = 0.5  # How confident we are in this doctrine
    success_count: int = 0
    failure_count: int = 0
    created_time: float = field(default_factory=time.time)
    neural_repr: Optional[np.ndarray] = None  # Neural representation

    @property
    def effectiveness(self) -> float:
        """Compute doctrine effectiveness"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if doctrine trigger matches context"""
        for key, condition in self.trigger.items():
            if key not in context:
                return False

            value = context[key]

            # Handle different condition types
            if isinstance(condition, dict):
                if 'gt' in condition and value <= condition['gt']:
                    return False
                if 'lt' in condition and value >= condition['lt']:
                    return False
                if 'eq' in condition and value != condition['eq']:
                    return False
            else:
                if value != condition:
                    return False

        return True

    def record_outcome(self, success: bool):
        """Record the outcome of applying this doctrine"""
        if success:
            self.success_count += 1
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.failure_count += 1
            self.confidence = max(0.0, self.confidence - 0.1)


class NeuralDoctrineEngine:
    """
    Neural network-based doctrine formation.

    Replaces statistical heuristics with learned pattern recognition.
    Uses sequence model to detect coordination patterns from observation history.

    Architecture:
    1. Pattern Recognizer: Encodes observation sequences to pattern embeddings
    2. Doctrine Classifier: Classifies patterns into doctrine types
    3. Doctrine Generator: VAE-style generation of doctrine representations
    """

    def __init__(self, obs_dim: int = 64, min_confidence: float = 0.3,
                 max_doctrines: int = 50, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.min_confidence = min_confidence
        self.max_doctrines = max_doctrines
        self.device = device

        self.doctrines: Dict[str, Doctrine] = {}
        self.pattern_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 200

        # Pattern recognition network
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

        # Doctrine type classifier: is this a learnable doctrine?
        self.doctrine_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [coordination, efficiency, safety]
        ).to(device)

        # Doctrine encoder (from sequence to latent)
        self.window_size = 10
        self.doctrine_encoder = nn.Sequential(
            nn.Linear(obs_dim * self.window_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64)
        ).to(device)

        # Doctrine decoder (latent to representation)
        self.doctrine_decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Doctrine representation
        ).to(device)

        # Optimizer for all networks
        self.optimizer = torch.optim.Adam(
            list(self.pattern_recognizer.parameters()) +
            list(self.doctrine_classifier.parameters()) +
            list(self.doctrine_encoder.parameters()) +
            list(self.doctrine_decoder.parameters()),
            lr=1e-4
        )

        # Training data buffer
        self.training_episodes: List[Tuple[np.ndarray, int]] = []
        self.training_capacity = 5000

    def add_observation(self, context: Dict[str, Any], outcome: Dict[str, Any]):
        """Add an observation for pattern detection"""
        self.pattern_buffer.append({
            'context': context,
            'outcome': outcome,
            'timestamp': time.time()
        })

        if len(self.pattern_buffer) > self.buffer_size:
            self.pattern_buffer.pop(0)

    def _context_to_array(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context dict to fixed-size array"""
        features = [
            float(context.get('robot_id', 0)) / 8.0,  # Normalize by num_robots
            float(context.get('reward', 0)) / 100.0,  # Normalize reward
            1.0 if context.get('carrying', False) else 0.0,
            float(context.get('battery', 100)) / 100.0,
            float(context.get('num_packages', 0)) / 20.0,
            float(context.get('distance_to_target', 0)) / 70.0,
            float(context.get('nearby_robots', 0)) / 8.0,
            float(context.get('collision_risk', 0)),
        ]

        # Pad to fixed size
        while len(features) < self.obs_dim:
            features.append(0.0)

        return np.array(features[:self.obs_dim], dtype=np.float32)

    def _extract_sequences(self) -> List[np.ndarray]:
        """Extract observation sequences from buffer"""
        if len(self.pattern_buffer) < self.window_size:
            return []

        sequences = []
        for i in range(len(self.pattern_buffer) - self.window_size):
            seq = []
            for j in range(self.window_size):
                obs_array = self._context_to_array(self.pattern_buffer[i + j]['context'])
                seq.append(obs_array)
            sequences.append(np.array(seq))

        return sequences

    def attempt_doctrine_formation(self) -> Optional[Doctrine]:
        """
        Neural doctrine formation instead of statistical heuristics.

        Process:
        1. Encode recent pattern buffer with neural network
        2. Classify if it's a learnable doctrine
        3. Generate doctrine representation
        4. Decode to action rules
        """
        if len(self.pattern_buffer) < 20:
            return None

        # Extract observation sequences
        sequences = self._extract_sequences()
        if not sequences:
            return None

        # Sample a batch for classification
        sample_size = min(10, len(sequences))
        sample_indices = np.random.choice(len(sequences), sample_size, replace=False)
        sample_sequences = [sequences[i] for i in sample_indices]

        # Encode patterns
        pattern_embeddings = []
        for seq in sample_sequences:
            seq_tensor = torch.from_numpy(seq).float().to(self.device)
            # Mean pool over sequence
            embedding = self.pattern_recognizer(seq_tensor.mean(dim=0))
            pattern_embeddings.append(embedding)

        pattern_emb = torch.stack(pattern_embeddings).mean(dim=0)

        # Classify doctrine type
        doctrine_logits = self.doctrine_classifier(pattern_emb)
        doctrine_probs = F.softmax(doctrine_logits, dim=-1)

        max_prob = doctrine_probs.max().item()
        doctrine_type_idx = doctrine_probs.argmax().item()

        if max_prob < 0.6:  # Confidence threshold
            return None

        # Generate doctrine representation
        # Flatten sequence for encoder
        flat_seq = sample_sequences[0].flatten()
        if len(flat_seq) < self.obs_dim * self.window_size:
            flat_seq = np.pad(flat_seq, (0, self.obs_dim * self.window_size - len(flat_seq)))
        else:
            flat_seq = flat_seq[:self.obs_dim * self.window_size]

        seq_tensor = torch.from_numpy(flat_seq).float().to(self.device)
        doctrine_latent = self.doctrine_encoder(seq_tensor)
        doctrine_repr = self.doctrine_decoder(doctrine_latent)

        # Create doctrine object
        doctrine_types = [DoctrineType.COORDINATION, DoctrineType.EFFICIENCY, DoctrineType.SAFETY]

        trigger = self._extract_trigger_from_repr(doctrine_repr)
        action = self._extract_action_from_repr(doctrine_repr)

        doctrine = Doctrine(
            id=f"neural_doctrine_{len(self.doctrines)}_{int(time.time())}",
            type=doctrine_types[doctrine_type_idx],
            trigger=trigger,
            action=action,
            confidence=max_prob,
            neural_repr=doctrine_repr.detach().cpu().numpy()
        )

        self.doctrines[doctrine.id] = doctrine
        self._prune_doctrines()

        return doctrine

    def _extract_trigger_from_repr(self, repr: torch.Tensor) -> Dict[str, Any]:
        """Extract trigger conditions from learned representation"""
        repr_np = repr.detach().cpu().numpy()
        trigger = {}

        # Interpret first few dimensions as trigger conditions
        if repr_np[0] > 0.3:  # Battery importance
            trigger['battery'] = {'gt': 30}

        if repr_np[1] > 0.3:  # Package count matters
            trigger['num_packages'] = {'gt': 3}

        if repr_np[2] > 0.5:  # Congestion detection
            trigger['nearby_robots'] = {'gt': 2}

        if repr_np[3] < -0.3:  # Low carrying = seeking packages
            trigger['carrying'] = False

        return trigger

    def _extract_action_from_repr(self, repr: torch.Tensor) -> Dict[str, Any]:
        """Extract action recommendation from learned representation"""
        repr_np = repr.detach().cpu().numpy()
        action = {'strategy': 'neural_learned'}

        # Decode action type from representation (using different slice)
        action_logits = repr_np[10:13]
        action_idx = np.argmax(action_logits)

        if action_idx == 0:
            action['priority'] = 'coordination'
            action['behavior'] = 'wait_for_others'
        elif action_idx == 1:
            action['priority'] = 'efficiency'
            action['behavior'] = 'nearest_first'
        else:
            action['priority'] = 'safety'
            action['behavior'] = 'avoid_congestion'

        # Speed recommendation
        if repr_np[5] > 0.5:
            action['speed_modifier'] = 'slow'
        elif repr_np[5] < -0.5:
            action['speed_modifier'] = 'fast'
        else:
            action['speed_modifier'] = 'normal'

        return action

    def _prune_doctrines(self):
        """Remove low-confidence or unused doctrines"""
        if len(self.doctrines) <= self.max_doctrines:
            return

        # Sort by confidence * usage
        scored = [
            (d.confidence * (d.success_count + d.failure_count + 1), d_id)
            for d_id, d in self.doctrines.items()
        ]
        scored.sort()

        # Remove lowest scoring
        to_remove = len(self.doctrines) - self.max_doctrines
        for _, d_id in scored[:to_remove]:
            del self.doctrines[d_id]

    def get_applicable_doctrines(self, context: Dict[str, Any]) -> List[Doctrine]:
        """Get all doctrines that apply to current context"""
        applicable = []
        for doctrine in self.doctrines.values():
            if doctrine.confidence >= self.min_confidence and doctrine.matches(context):
                applicable.append(doctrine)

        # Sort by confidence
        applicable.sort(key=lambda d: d.confidence, reverse=True)
        return applicable

    def train_on_successful_episodes(self, episodes: List[Tuple[np.ndarray, str]],
                                     num_steps: int = 50) -> float:
        """
        Train doctrine network on successful coordination episodes.

        Args:
            episodes: List of (context_sequence, outcome) tuples
            num_steps: Training iterations

        Returns:
            Average training loss
        """
        if len(episodes) < 32:
            return 0.0

        total_loss = 0.0

        for step in range(num_steps):
            # Sample batch
            batch = random.sample(episodes, min(32, len(episodes)))

            losses = []
            for context_seq, outcome in batch:
                # Ensure correct shape
                if len(context_seq.shape) == 1:
                    context_seq = context_seq.reshape(-1, self.obs_dim)

                # Forward pass
                seq_tensor = torch.from_numpy(context_seq).float().to(self.device)
                pattern_emb = self.pattern_recognizer(seq_tensor.mean(dim=0))

                # Classification loss (supervised by outcome)
                doctrine_logits = self.doctrine_classifier(pattern_emb)
                target = self._outcome_to_doctrine_type(outcome)

                loss = F.cross_entropy(doctrine_logits.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss)

            if not losses:
                continue

            # Optimize
            batch_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()

        return total_loss / num_steps if num_steps > 0 else 0.0

    def _outcome_to_doctrine_type(self, outcome: str) -> torch.Tensor:
        """Convert outcome string to doctrine type index"""
        if outcome == "success" or "delivered" in outcome.lower():
            return torch.tensor(1, device=self.device)  # Efficiency
        elif "collision" in outcome.lower() or "failure" in outcome.lower():
            return torch.tensor(2, device=self.device)  # Safety
        else:
            return torch.tensor(0, device=self.device)  # Coordination

    def get_stats(self) -> Dict[str, Any]:
        """Get doctrine engine statistics"""
        return {
            'total_doctrines': len(self.doctrines),
            'buffer_size': len(self.pattern_buffer),
            'doctrine_types': {
                dtype.value: sum(1 for d in self.doctrines.values() if d.type == dtype)
                for dtype in DoctrineType
            },
            'avg_confidence': np.mean([d.confidence for d in self.doctrines.values()])
                             if self.doctrines else 0.0,
            'avg_effectiveness': np.mean([d.effectiveness for d in self.doctrines.values()])
                                if self.doctrines else 0.0,
        }


# =============================================================================
# Factory function
# =============================================================================

def create_doctrine_engine(use_neural: bool = False, obs_dim: int = 64,
                          **kwargs) -> 'DoctrineEngine':
    """
    Factory to create either statistical or neural doctrine engine.

    Args:
        use_neural: If True, use NeuralDoctrineEngine
        obs_dim: Observation dimension (required for neural version)
        **kwargs: Additional arguments

    Returns:
        DoctrineEngine instance
    """
    if use_neural:
        return NeuralDoctrineEngine(obs_dim=obs_dim, **kwargs)
    else:
        # Import and return simple statistical version
        from qrea_cognitive import DoctrineEngine
        return DoctrineEngine(**kwargs)
