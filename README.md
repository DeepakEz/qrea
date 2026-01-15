# QREA: Quantum-Resonant Evolutionary Agents

Multi-agent reinforcement learning for warehouse coordination using MERA (Multi-scale Entanglement Renormalization Ansatz) tensor networks.

## Architecture

```
warehouse_env.py          → Multi-robot warehouse environment
    ↓
warehouse_uprt.py         → Spatial activity fields (UPRT)
    ↓
mera_enhanced.py          → MERA tensor network encoder
    ↓
mera_ppo_warehouse.py     → PPO training with MERA/GRU/Transformer encoders
    ↓
run_experiments.py        → Multi-seed experiment runner
```

## Quick Start

```bash
# Install dependencies
pip install torch numpy pyyaml gymnasium

# Run training (50 epochs, MERA encoder)
python mera_ppo_warehouse.py --epochs 50 --encoder mera

# Compare encoders
python run_experiments.py --compare all --seeds 3 --epochs 100
```

## Encoders

| Encoder | Description |
|---------|-------------|
| `mera` | MERA tensor network (temporal hierarchy) |
| `mera_uprt` | MERA + UPRT spatial fields |
| `gru` | GRU baseline |
| `transformer` | Transformer baseline |
| `mlp` | MLP baseline |

## Key Metrics

- **Pickups**: Packages picked up per episode
- **Deliveries**: Packages delivered per episode
- **Collisions**: Robot-robot collisions
- **Throughput**: Packages delivered per hour
- **Φ_Q**: Hierarchical entropy metric (correlation-based diagnostic)

## Configuration

Edit `config.yaml` to modify:

```yaml
environment:
  num_robots: 8
  grid_size: [50, 50]
  num_packages: 100
  max_episode_steps: 1000

learning:
  learning_rate: 0.0003
  actor_critic:
    discount: 0.99
    lambda_gae: 0.95
```

## Reward Structure

| Reward | Value | Condition |
|--------|-------|-----------|
| Delivery | +100/200/300 | Package delivered (by priority) |
| Pickup | +25 | Package picked up |
| Approach (not carrying) | 0.3/step | Moving toward nearest package |
| Carrying progress | 0.5/step | Moving toward delivery station |
| Pre-pickup | 20.0 max | Slowing down near package |
| Collision | -2.0 | Robot-robot collision |

## Task Design

- **Package spawning**: Scattered across grid (not just at pickup station)
- **Pickup requirements**: Speed < 1.0, Distance < 1.5m, Gripper action > 0.5
- **Delivery stations**: Fixed positions at grid corners
- **No auto-pickup**: Agents must learn gripper control naturally

## Training Tips

1. **Pickups but no deliveries?** Increase carrying progress reward
2. **No pickups?** Check package spawn locations, increase approach reward
3. **Φ_Q stuck at 0?** Increase MERA noise initialization (0.5 recommended)
4. **High collisions?** Agents are learning coordination, will decrease over time

## File Structure

```
qrea/
├── config.yaml              # Main configuration
├── warehouse_env.py         # Gym-style warehouse environment
├── warehouse_uprt.py        # Spatial activity fields
├── mera_enhanced.py         # MERA tensor network
├── mera_ppo_warehouse.py    # PPO training loop
├── run_experiments.py       # Multi-seed experiments
├── qrea_cognitive.py        # Cognitive architecture (optional)
└── results_*/               # Training outputs
```

## Research Notes

- **Φ_Q** is a diagnostic probe, not optimized directly
- Dense rewards used for learning, sparse for evaluation
- MERA provides hierarchical temporal encoding
- UPRT provides spatial coordination context

## Citation

```bibtex
@article{qrea2024,
  title={QREA: Multi-Agent Coordination via Tensor Network Encoders},
  author={...},
  year={2024}
}
```

## License

MIT License
