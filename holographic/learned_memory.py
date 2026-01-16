"""
Learned Episodic Memory Embeddings
==================================

Upgrades EpisodicMemory with learned contrastive embeddings instead of
random projections. Uses InfoNCE loss to learn meaningful representations
where similar outcomes cluster together.

This is Phase 3 upgrade - use after validating holographic hypothesis
with simpler heuristics.

Usage:
    from holographic.learned_memory import LearnedEpisodicMemory

    memory = LearnedEpisodicMemory(obs_dim=514, action_dim=3)
    memory.add(robot_id=0, obs=obs, action=action, reward=reward, outcome="success")
    similar = memory.retrieve_similar(obs, action, k=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import random


@dataclass
class Episode:
    """A remembered episode/experience"""
    timestamp: float
    robot_id: int
    observation: np.ndarray
    action: np.ndarray
    reward: float
    outcome: str  # "success", "failure", "partial"
    context: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    episode_id: int = 0

    def similarity(self, other: 'Episode') -> float:
        """Compute similarity to another episode"""
        if self.embedding is not None and other.embedding is not None:
            return float(np.dot(self.embedding, other.embedding) /
                        (np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding) + 1e-8))
        return 0.0


class LearnedEpisodicMemory:
    """
    Episodic memory with learned embeddings instead of random projections.

    Uses contrastive learning to learn meaningful state embeddings where:
    - Positive pairs: Similar outcomes (both success or both failure)
    - Negative pairs: Different outcomes

    This produces embeddings that cluster by task success, enabling
    better experience retrieval for decision making.
    """

    def __init__(self, obs_dim: int, action_dim: int, embedding_dim: int = 128,
                 capacity: int = 10000, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.device = device

        # Input size for encoder
        self.input_dim = obs_dim + action_dim

        # Learned encoder replaces random projection
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        ).to(device)

        # Contrastive learning temperature
        self.temperature = 0.07

        # Episode storage
        from collections import deque
        self.episodes: deque = deque(maxlen=capacity)
        self.episode_counter = 0

        # Training tracking
        self.train_losses = []
        self.is_trained = False

    def _prepare_input(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Prepare combined input for encoder"""
        combined = np.concatenate([obs.flatten(), action.flatten()])

        # Pad/truncate to match encoder input
        if len(combined) < self.input_dim:
            combined = np.pad(combined, (0, self.input_dim - len(combined)))
        else:
            combined = combined[:self.input_dim]

        return combined.astype(np.float32)

    def _compute_embedding(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute LEARNED embedding instead of random projection"""
        combined = self._prepare_input(obs, action)

        # Forward through learned encoder
        with torch.no_grad():
            x = torch.from_numpy(combined).float().unsqueeze(0).to(self.device)
            embedding = self.encoder(x).squeeze().cpu().numpy()

        # L2 normalize
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def add(self, robot_id: int, obs: np.ndarray, action: np.ndarray,
            reward: float, outcome: str, context: Dict[str, Any] = None):
        """Add a new episode to memory"""
        embedding = self._compute_embedding(obs, action)

        episode = Episode(
            timestamp=time.time(),
            robot_id=robot_id,
            observation=obs.copy(),
            action=action.copy(),
            reward=reward,
            outcome=outcome,
            context=context or {},
            embedding=embedding,
            episode_id=self.episode_counter
        )

        self.episode_counter += 1
        self.episodes.append(episode)

    def retrieve_similar(self, obs: np.ndarray, action: np.ndarray,
                         k: int = 5) -> List[Episode]:
        """Retrieve k most similar episodes"""
        if not self.episodes:
            return []

        query_embedding = self._compute_embedding(obs, action)

        # Compute similarities
        similarities = []
        for ep in self.episodes:
            if ep.embedding is not None:
                sim = float(np.dot(query_embedding, ep.embedding))
                similarities.append((sim, ep))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in similarities[:k]]

    def retrieve_by_outcome(self, outcome: str, k: int = 10) -> List[Episode]:
        """Retrieve episodes with specific outcome"""
        matching = [ep for ep in self.episodes if ep.outcome == outcome]
        return list(matching)[-k:]  # Most recent

    def get_success_rate(self, robot_id: Optional[int] = None,
                         window: int = 100) -> float:
        """Compute recent success rate"""
        recent = list(self.episodes)[-window:]

        if robot_id is not None:
            recent = [ep for ep in recent if ep.robot_id == robot_id]

        if not recent:
            return 0.0

        successes = sum(1 for ep in recent if ep.outcome == "success")
        return successes / len(recent)

    def train_embeddings(self, batch_size: int = 256, num_steps: int = 100) -> float:
        """
        Train embeddings via contrastive learning.

        Positive pairs: Similar outcomes (both success or both failure)
        Negative pairs: Different outcomes

        Returns average loss over training.
        """
        if len(self.episodes) < batch_size:
            return 0.0

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        total_loss = 0.0

        for step in range(num_steps):
            # Sample episodes
            indices = np.random.choice(len(self.episodes), min(batch_size, len(self.episodes)), replace=False)
            episodes = [self.episodes[i] for i in indices]

            # Create positive/negative pairs
            positives, negatives = self._create_contrastive_pairs(episodes)

            if not positives or not negatives:
                continue

            # Compute embeddings
            anchor_inputs = []
            pos_inputs = []
            neg_inputs = []

            for anchor, pos, neg in zip(episodes, positives, negatives):
                anchor_inputs.append(self._prepare_input(anchor.observation, anchor.action))
                pos_inputs.append(self._prepare_input(pos.observation, pos.action))
                neg_inputs.append(self._prepare_input(neg.observation, neg.action))

            anchor_batch = torch.tensor(np.array(anchor_inputs), device=self.device)
            pos_batch = torch.tensor(np.array(pos_inputs), device=self.device)
            neg_batch = torch.tensor(np.array(neg_inputs), device=self.device)

            # Forward pass
            anchor_embs = self.encoder(anchor_batch)
            pos_embs = self.encoder(pos_batch)
            neg_embs = self.encoder(neg_batch)

            # L2 normalize
            anchor_embs = F.normalize(anchor_embs, p=2, dim=-1)
            pos_embs = F.normalize(pos_embs, p=2, dim=-1)
            neg_embs = F.normalize(neg_embs, p=2, dim=-1)

            # InfoNCE loss
            pos_sim = (anchor_embs * pos_embs).sum(dim=-1) / self.temperature
            neg_sim = (anchor_embs * neg_embs).sum(dim=-1) / self.temperature

            # Softmax-style contrastive loss
            logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (batch, 2)
            labels = torch.zeros(len(episodes), dtype=torch.long, device=self.device)  # positive is index 0

            loss = F.cross_entropy(logits, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        self.train_losses.append(avg_loss)
        self.is_trained = True

        # Re-embed all episodes with updated encoder
        self._recompute_all_embeddings()

        return avg_loss

    def _create_contrastive_pairs(self, episodes: List[Episode]) -> Tuple[List, List]:
        """Create positive/negative pairs for contrastive learning"""
        positives = []
        negatives = []

        all_episodes = list(self.episodes)

        for ep in episodes:
            # Positive: same outcome
            pos_candidates = [e for e in all_episodes
                            if e.outcome == ep.outcome and e.episode_id != ep.episode_id]
            if pos_candidates:
                positives.append(random.choice(pos_candidates))
            else:
                positives.append(ep)  # Self if no match

            # Negative: different outcome
            neg_candidates = [e for e in all_episodes if e.outcome != ep.outcome]
            if neg_candidates:
                negatives.append(random.choice(neg_candidates))
            else:
                negatives.append(random.choice(all_episodes))

        return positives, negatives

    def _recompute_all_embeddings(self):
        """Recompute embeddings for all stored episodes after training"""
        for ep in self.episodes:
            ep.embedding = self._compute_embedding(ep.observation, ep.action)

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about learned embeddings"""
        if not self.episodes:
            return {'count': 0}

        embeddings = np.array([ep.embedding for ep in self.episodes if ep.embedding is not None])

        if len(embeddings) == 0:
            return {'count': 0}

        # Compute within-class similarity (same outcome)
        success_embs = np.array([ep.embedding for ep in self.episodes
                                if ep.outcome == "success" and ep.embedding is not None])
        failure_embs = np.array([ep.embedding for ep in self.episodes
                                if ep.outcome == "failure" and ep.embedding is not None])

        stats = {
            'count': len(embeddings),
            'is_trained': self.is_trained,
            'train_loss': self.train_losses[-1] if self.train_losses else None,
        }

        if len(success_embs) > 1:
            # Average pairwise similarity within success class
            sim_matrix = success_embs @ success_embs.T
            np.fill_diagonal(sim_matrix, 0)
            stats['success_coherence'] = float(sim_matrix.sum() / (len(success_embs) * (len(success_embs) - 1)))

        if len(failure_embs) > 1:
            sim_matrix = failure_embs @ failure_embs.T
            np.fill_diagonal(sim_matrix, 0)
            stats['failure_coherence'] = float(sim_matrix.sum() / (len(failure_embs) * (len(failure_embs) - 1)))

        if len(success_embs) > 0 and len(failure_embs) > 0:
            # Between-class similarity (should be low)
            cross_sim = success_embs @ failure_embs.T
            stats['cross_class_sim'] = float(cross_sim.mean())

        return stats


# =============================================================================
# Factory function for easy switching
# =============================================================================

def create_episodic_memory(use_learned: bool = False, obs_dim: int = 514,
                          action_dim: int = 3, **kwargs) -> 'EpisodicMemory':
    """
    Factory to create either simple or learned episodic memory.

    Args:
        use_learned: If True, use LearnedEpisodicMemory with contrastive learning
        obs_dim: Observation dimension (required for learned version)
        action_dim: Action dimension (required for learned version)
        **kwargs: Additional arguments passed to memory constructor

    Returns:
        EpisodicMemory instance (simple or learned)
    """
    if use_learned:
        return LearnedEpisodicMemory(obs_dim=obs_dim, action_dim=action_dim, **kwargs)
    else:
        # Import and return simple version
        from qrea_cognitive import EpisodicMemory
        return EpisodicMemory(**kwargs)
