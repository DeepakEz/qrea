"""
QREA Cognitive Architecture Integration
========================================

Adds cognitive capabilities on top of QREA's RL agents:
- Episodic memory (experience storage and retrieval)
- Semantic memory (knowledge graphs)
- Doctrine formation (learned coordination strategies)
- Social coordination layer (multi-agent communication)

The cognitive layer sits ABOVE the reactive MERA-PPO layer:
- MERA-PPO: Low-level control (motion, gripper, immediate reactions)
- Cognitive: High-level planning (task allocation, coordination, learning)

Usage:
    from qrea_cognitive import CognitiveQREATrainer

    trainer = CognitiveQREATrainer(config_path='config.yaml')
    trainer.train(epochs=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
import json
import time
from enum import Enum

# Import existing QREA components
from mera_ppo_warehouse import MERAWarehousePPO, Transition, CoordinationMetrics
from warehouse_env import WarehouseEnv, Robot, Package


# =============================================================================
# Memory Systems
# =============================================================================

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

    def similarity(self, other: 'Episode') -> float:
        """Compute similarity to another episode"""
        if self.embedding is not None and other.embedding is not None:
            return float(np.dot(self.embedding, other.embedding) /
                        (np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding) + 1e-8))
        return 0.0


class EpisodicMemory:
    """
    Vector-based episodic memory for experience storage and retrieval.

    Stores experiences with embeddings for similarity-based retrieval.
    Supports:
    - Adding new episodes
    - Retrieving similar episodes
    - Forgetting old/irrelevant episodes
    """

    def __init__(self, capacity: int = 10000, embedding_dim: int = 128):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.episodes: List[Episode] = []

        # Simple embedding: random projection (could use learned embeddings)
        self.projection = np.random.randn(512, embedding_dim) / np.sqrt(512)

    def _compute_embedding(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute embedding for an experience"""
        # Flatten and pad/truncate to fixed size
        combined = np.concatenate([obs.flatten(), action.flatten()])
        if len(combined) < 512:
            combined = np.pad(combined, (0, 512 - len(combined)))
        else:
            combined = combined[:512]

        # Project to embedding space
        embedding = combined @ self.projection
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
            embedding=embedding
        )

        self.episodes.append(episode)

        # Forget oldest if over capacity
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

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
        return matching[-k:]  # Most recent

    def get_success_rate(self, robot_id: Optional[int] = None,
                         window: int = 100) -> float:
        """Compute recent success rate"""
        if robot_id is not None:
            recent = [ep for ep in self.episodes[-window:] if ep.robot_id == robot_id]
        else:
            recent = self.episodes[-window:]

        if not recent:
            return 0.0

        successes = sum(1 for ep in recent if ep.outcome == "success")
        return successes / len(recent)


@dataclass
class KnowledgeNode:
    """Node in the semantic knowledge graph"""
    id: str
    type: str  # "concept", "rule", "strategy"
    content: Dict[str, Any]
    connections: List[str] = field(default_factory=list)
    confidence: float = 1.0
    usage_count: int = 0
    last_used: float = 0.0


class SemanticMemory:
    """
    Graph-based semantic memory for learned knowledge.

    Stores:
    - Concepts (e.g., "heavy_package", "congested_area")
    - Rules (e.g., "heavy_package -> need_help")
    - Strategies (e.g., "zone_division", "relay_delivery")
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.type_index: Dict[str, List[str]] = {
            "concept": [],
            "rule": [],
            "strategy": []
        }

    def add_node(self, node_id: str, node_type: str, content: Dict[str, Any],
                 connections: List[str] = None) -> KnowledgeNode:
        """Add a knowledge node"""
        node = KnowledgeNode(
            id=node_id,
            type=node_type,
            content=content,
            connections=connections or [],
            last_used=time.time()
        )
        self.nodes[node_id] = node

        if node_type in self.type_index:
            self.type_index[node_type].append(node_id)

        return node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID"""
        node = self.nodes.get(node_id)
        if node:
            node.usage_count += 1
            node.last_used = time.time()
        return node

    def query_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Get all nodes of a specific type"""
        node_ids = self.type_index.get(node_type, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def find_connected(self, node_id: str, max_depth: int = 2) -> List[KnowledgeNode]:
        """Find all connected nodes up to max_depth"""
        if node_id not in self.nodes:
            return []

        visited = set()
        to_visit = [(node_id, 0)]
        result = []

        while to_visit:
            current_id, depth = to_visit.pop(0)
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            node = self.nodes.get(current_id)
            if node:
                result.append(node)
                for conn_id in node.connections:
                    if conn_id not in visited:
                        to_visit.append((conn_id, depth + 1))

        return result


# =============================================================================
# Doctrine System
# =============================================================================

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


class DoctrineEngine:
    """
    Engine for learning and applying doctrines from experience.

    Learns patterns like:
    - "Heavy packages (>10kg) are delivered 30% faster with 2 robots"
    - "Rush hour (steps 1000-2000) has 50% more collisions"
    - "Zone-based allocation reduces travel distance by 20%"
    """

    def __init__(self, min_confidence: float = 0.3, max_doctrines: int = 50):
        self.doctrines: Dict[str, Doctrine] = {}
        self.min_confidence = min_confidence
        self.max_doctrines = max_doctrines

        # Pattern detection buffers
        self.pattern_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100

    def add_observation(self, context: Dict[str, Any], outcome: Dict[str, Any]):
        """Add an observation for pattern detection"""
        self.pattern_buffer.append({
            'context': context,
            'outcome': outcome,
            'timestamp': time.time()
        })

        if len(self.pattern_buffer) > self.buffer_size:
            self.pattern_buffer.pop(0)

    def attempt_doctrine_formation(self) -> Optional[Doctrine]:
        """
        Attempt to form a new doctrine from observed patterns.

        Uses simple statistical analysis to find correlations.
        """
        if len(self.pattern_buffer) < 20:
            return None

        # Analyze patterns for potential doctrines
        # Example: Look for conditions that correlate with success

        success_contexts = [
            obs['context'] for obs in self.pattern_buffer
            if obs['outcome'].get('success', False)
        ]

        failure_contexts = [
            obs['context'] for obs in self.pattern_buffer
            if not obs['outcome'].get('success', True)
        ]

        if not success_contexts or not failure_contexts:
            return None

        # Find distinguishing features
        potential_triggers = self._find_distinguishing_features(
            success_contexts, failure_contexts
        )

        if not potential_triggers:
            return None

        # Create doctrine
        doctrine_id = f"doctrine_{len(self.doctrines)}_{int(time.time())}"
        doctrine = Doctrine(
            id=doctrine_id,
            type=DoctrineType.EFFICIENCY,
            trigger=potential_triggers,
            action={'strategy': 'apply_learned_pattern'},
            confidence=0.5
        )

        self.doctrines[doctrine_id] = doctrine

        # Prune low-confidence doctrines if over limit
        self._prune_doctrines()

        return doctrine

    def _find_distinguishing_features(self, success: List[Dict],
                                       failure: List[Dict]) -> Dict[str, Any]:
        """Find features that distinguish success from failure"""
        triggers = {}

        # Simple heuristic: find numeric features with different means
        all_keys = set()
        for ctx in success + failure:
            all_keys.update(ctx.keys())

        for key in all_keys:
            success_vals = [ctx.get(key) for ctx in success if key in ctx]
            failure_vals = [ctx.get(key) for ctx in failure if key in ctx]

            # Only analyze numeric values
            success_nums = [v for v in success_vals if isinstance(v, (int, float))]
            failure_nums = [v for v in failure_vals if isinstance(v, (int, float))]

            if len(success_nums) >= 5 and len(failure_nums) >= 5:
                success_mean = np.mean(success_nums)
                failure_mean = np.mean(failure_nums)

                # If means differ significantly, create trigger
                if abs(success_mean - failure_mean) > 0.5 * (abs(success_mean) + 1e-6):
                    if success_mean > failure_mean:
                        triggers[key] = {'gt': (success_mean + failure_mean) / 2}
                    else:
                        triggers[key] = {'lt': (success_mean + failure_mean) / 2}

        return triggers

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


# =============================================================================
# Cognitive Agent
# =============================================================================

class CognitiveAgent:
    """
    Cognitive layer for a single QREA robot.

    Adds to base QREA agent:
    - Episodic memory of past experiences
    - Semantic knowledge about the environment
    - Doctrine-based coordination
    - Communication with other agents
    """

    def __init__(self, agent_id: int, memory_capacity: int = 5000):
        self.agent_id = agent_id

        # Memory systems
        self.episodic_memory = EpisodicMemory(capacity=memory_capacity)
        self.semantic_memory = SemanticMemory()

        # State tracking
        self.current_goal: Optional[str] = None
        self.pending_communications: List[Dict[str, Any]] = []
        self.received_messages: List[Dict[str, Any]] = []

        # Statistics
        self.decisions_made = 0
        self.doctrines_applied = 0

    def process_observation(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process observation through cognitive layer.

        Returns enriched observation with:
        - Similar past experiences
        - Relevant knowledge
        - Applicable doctrines
        """
        # Retrieve similar experiences
        similar_episodes = self.episodic_memory.retrieve_similar(
            obs, np.zeros(3), k=3  # Dummy action for retrieval
        )

        # Extract lessons from similar experiences
        lessons = []
        for ep in similar_episodes:
            if ep.outcome == "success":
                lessons.append({
                    'type': 'success_pattern',
                    'action': ep.action,
                    'reward': ep.reward
                })
            elif ep.outcome == "failure":
                lessons.append({
                    'type': 'avoid_pattern',
                    'action': ep.action,
                    'reward': ep.reward
                })

        return {
            'raw_observation': obs,
            'similar_episodes': similar_episodes,
            'lessons': lessons,
            'success_rate': self.episodic_memory.get_success_rate(self.agent_id),
            'info': info
        }

    def record_experience(self, obs: np.ndarray, action: np.ndarray,
                          reward: float, info: Dict[str, Any]):
        """Record experience to episodic memory"""
        # Determine outcome
        if reward > 50:  # Successful delivery
            outcome = "success"
        elif reward < -10:  # Collision or failure
            outcome = "failure"
        else:
            outcome = "partial"

        context = {
            'position': info.get('position', [0, 0]),
            'carrying': info.get('carrying', False),
            'battery': info.get('battery', 100),
            'nearby_robots': info.get('nearby_robots', 0)
        }

        self.episodic_memory.add(
            robot_id=self.agent_id,
            obs=obs,
            action=action,
            reward=reward,
            outcome=outcome,
            context=context
        )

    def send_message(self, target_id: int, message_type: str, content: Any):
        """Queue a message to send to another agent"""
        self.pending_communications.append({
            'from': self.agent_id,
            'to': target_id,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        })

    def receive_message(self, message: Dict[str, Any]):
        """Receive a message from another agent"""
        self.received_messages.append(message)

    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get and clear pending outgoing messages"""
        messages = self.pending_communications
        self.pending_communications = []
        return messages


# =============================================================================
# Cognitive Trainer (Wrapper around MERAWarehousePPO)
# =============================================================================

class CognitiveQREATrainer:
    """
    Training wrapper that adds cognitive capabilities to MERA-PPO.

    Layers:
    1. Base: MERAWarehousePPO (reactive control)
    2. Cognitive: Memory + Doctrines (learning from experience)
    3. Social: Inter-agent communication (coordination)
    """

    def __init__(self, config_path: str = 'config.yaml',
                 encoder_type: str = 'mera',
                 enable_doctrines: bool = True,
                 enable_communication: bool = True):

        self.config_path = config_path
        self.encoder_type = encoder_type
        self.enable_doctrines = enable_doctrines
        self.enable_communication = enable_communication

        # Initialize base trainer
        self.base_trainer = MERAWarehousePPO(
            config_path=config_path,
            encoder_type=encoder_type
        )

        self.num_robots = self.base_trainer.num_robots

        # Create cognitive agents
        self.cognitive_agents: Dict[int, CognitiveAgent] = {
            i: CognitiveAgent(agent_id=i) for i in range(self.num_robots)
        }

        # Shared doctrine engine
        self.doctrine_engine = DoctrineEngine()

        # Communication buffer
        self.message_buffer: List[Dict[str, Any]] = []

        # Tracking
        self.cognitive_stats = {
            'doctrines_formed': 0,
            'doctrines_applied': 0,
            'messages_sent': 0,
            'experiences_stored': 0
        }

    def _process_step_cognitively(self, robot_id: int, obs: np.ndarray,
                                   action: np.ndarray, reward: float,
                                   info: Dict[str, Any]):
        """Process a step through the cognitive layer"""
        agent = self.cognitive_agents[robot_id]

        # Record experience
        agent.record_experience(obs, action, reward, info)
        self.cognitive_stats['experiences_stored'] += 1

        # Add to doctrine pattern buffer
        context = {
            'robot_id': robot_id,
            'reward': reward,
            'carrying': info.get('carrying', False),
            'battery': info.get('battery', 100),
            'num_packages': len([p for p in self.base_trainer.env.packages if not p.is_delivered])
        }

        outcome = {
            'success': reward > 0,
            'reward': reward,
            'collision': 'collision' in str(info.get('event', '')).lower()
        }

        self.doctrine_engine.add_observation(context, outcome)

        # Attempt doctrine formation periodically
        if self.enable_doctrines and np.random.random() < 0.01:
            new_doctrine = self.doctrine_engine.attempt_doctrine_formation()
            if new_doctrine:
                self.cognitive_stats['doctrines_formed'] += 1
                print(f"  [Cognitive] New doctrine formed: {new_doctrine.id}")

    def _process_communications(self):
        """Process inter-agent communications"""
        if not self.enable_communication:
            return

        # Collect all pending messages
        all_messages = []
        for agent in self.cognitive_agents.values():
            all_messages.extend(agent.get_pending_messages())

        # Deliver messages
        for msg in all_messages:
            target_id = msg['to']
            if target_id in self.cognitive_agents:
                self.cognitive_agents[target_id].receive_message(msg)
                self.cognitive_stats['messages_sent'] += 1

    def _apply_doctrines(self, robot_id: int, context: Dict[str, Any]) -> List[Doctrine]:
        """Get and apply applicable doctrines"""
        if not self.enable_doctrines:
            return []

        applicable = self.doctrine_engine.get_applicable_doctrines(context)

        if applicable:
            self.cognitive_stats['doctrines_applied'] += len(applicable)

        return applicable

    def train(self, epochs: Optional[int] = None):
        """
        Train with cognitive enhancements.

        Wraps base trainer's train() with cognitive processing.
        """
        if epochs:
            self.base_trainer.num_epochs = epochs

        print("=" * 70)
        print("Cognitive QREA Training")
        print("=" * 70)
        print(f"Base encoder: {self.encoder_type}")
        print(f"Robots: {self.num_robots}")
        print(f"Doctrines: {'enabled' if self.enable_doctrines else 'disabled'}")
        print(f"Communication: {'enabled' if self.enable_communication else 'disabled'}")
        print("=" * 70)

        # Hook into base trainer's rollout
        original_collect = self.base_trainer.collect_rollout

        def cognitive_collect():
            """Wrapper that adds cognitive processing to rollout"""
            result = original_collect()
            transitions, epoch_stats = result

            # Process transitions through cognitive layer
            for robot_id, robot_transitions in transitions.items():
                for t in robot_transitions:
                    info = {
                        'position': t.robot_position.tolist() if t.robot_position is not None else [0, 0],
                        'carrying': False,  # Would need to track this
                        'battery': 100
                    }
                    self._process_step_cognitively(robot_id, t.obs, t.action, t.reward, info)

            # Process communications
            self._process_communications()

            return result

        self.base_trainer.collect_rollout = cognitive_collect

        # Run training
        start_time = time.time()
        results = self.base_trainer.train()
        total_time = time.time() - start_time

        # Print cognitive stats
        print("\n" + "=" * 70)
        print("Cognitive Layer Statistics")
        print("=" * 70)
        print(f"Experiences stored: {self.cognitive_stats['experiences_stored']}")
        print(f"Doctrines formed: {self.cognitive_stats['doctrines_formed']}")
        print(f"Doctrines applied: {self.cognitive_stats['doctrines_applied']}")
        print(f"Messages sent: {self.cognitive_stats['messages_sent']}")

        # Print active doctrines
        if self.doctrine_engine.doctrines:
            print(f"\nActive Doctrines ({len(self.doctrine_engine.doctrines)}):")
            for d_id, doctrine in list(self.doctrine_engine.doctrines.items())[:5]:
                print(f"  - {d_id}: conf={doctrine.confidence:.2f}, "
                      f"eff={doctrine.effectiveness:.2f}")

        # Add cognitive results
        results['cognitive'] = {
            'stats': self.cognitive_stats,
            'doctrines': len(self.doctrine_engine.doctrines),
            'training_time': total_time
        }

        return results

    def save_cognitive_state(self, path: Path):
        """Save cognitive layer state"""
        state = {
            'doctrines': {
                d_id: asdict(d) for d_id, d in self.doctrine_engine.doctrines.items()
            },
            'stats': self.cognitive_stats,
            'agent_memories': {
                agent_id: {
                    'episode_count': len(agent.episodic_memory.episodes),
                    'success_rate': agent.episodic_memory.get_success_rate()
                }
                for agent_id, agent in self.cognitive_agents.items()
            }
        }

        with open(path / 'cognitive_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"Cognitive state saved to {path / 'cognitive_state.json'}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cognitive QREA Training")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--encoder', type=str, default='mera',
                        choices=['mera', 'mera_uprt', 'gru', 'transformer', 'mlp'])
    parser.add_argument('--no-doctrines', action='store_true',
                        help='Disable doctrine learning')
    parser.add_argument('--no-communication', action='store_true',
                        help='Disable inter-agent communication')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with 5 epochs')

    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 5

    trainer = CognitiveQREATrainer(
        config_path=args.config,
        encoder_type=args.encoder,
        enable_doctrines=not args.no_doctrines,
        enable_communication=not args.no_communication
    )

    results = trainer.train(epochs=args.epochs)

    # Save cognitive state
    trainer.save_cognitive_state(trainer.base_trainer.output_dir)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
