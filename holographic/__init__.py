"""
QREA Holographic Training Module
================================

Advanced cognitive upgrades for the QREA framework:

Phase 3 Upgrades:
- LearnedEpisodicMemory: Contrastive learning embeddings for better retrieval

Phase 4 Upgrades:
- NeuralDoctrineEngine: Neural pattern recognition for doctrine formation

Usage:
    # Use learned embeddings (Phase 3)
    python qrea_cognitive.py --learned_embeddings

    # Use neural doctrines (Phase 4)
    python qrea_cognitive.py --neural_doctrines

    # Both upgrades
    python qrea_cognitive.py --learned_embeddings --neural_doctrines
"""

from .learned_memory import LearnedEpisodicMemory, create_episodic_memory
from .neural_doctrine_engine import NeuralDoctrineEngine, create_doctrine_engine

__all__ = [
    'LearnedEpisodicMemory',
    'create_episodic_memory',
    'NeuralDoctrineEngine',
    'create_doctrine_engine',
]
