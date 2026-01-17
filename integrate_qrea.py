#!/usr/bin/env python3
"""
QREA Single Codebase Integration Script
========================================

Integrates all files into ONE unified codebase with all features.

Usage:
    python integrate_qrea.py
    
This will:
1. Organize all files properly
2. Fix all import statements
3. Create holographic/ module
4. Add missing files (config.yaml, etc.)
5. Test the integration

Result: ONE working codebase with everything.
"""

import os
import shutil
from pathlib import Path
import re


def fix_imports_in_file(filepath: Path, replacements: dict):
    """Fix import statements in a file"""
    if not filepath.exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified = False
        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)
                modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Fixed imports in {filepath.name}")
            return True
        return False
    except Exception as e:
        print(f"  ❌ Error fixing {filepath.name}: {e}")
        return False


def integrate_codebase():
    """Main integration function"""
    print("=" * 70)
    print("QREA SINGLE CODEBASE INTEGRATION")
    print("=" * 70)
    print("\nThis will create ONE unified codebase with all features.")
    print("Working directory:", Path.cwd())
    
    # File mapping: what to rename/keep
    file_actions = {
        # Files to rename (complete → regular names)
        'renames': {
            'warehouse_env_complete.py': 'warehouse_env.py',
            'mera_ppo_complete.py': 'mera_ppo_curriculum.py',  # Keep both PPO versions
            'visualize_complete.py': 'visualize.py',
            'analysis_complete.py': 'analysis.py',
        },
        # Files to keep as-is
        'keep': [
            'mera_enhanced.py',
            'mera_ppo_warehouse.py',  # Original PPO (simpler)
            'warehouse_uprt.py',
            'qrea_cognitive.py',
            'run_experiments.py',
        ],
        # Files to move to holographic/
        'move_to_holographic': [
            'learned_memory.py',
            'neural_doctrine_engine.py',
        ]
    }
    
    # Step 1: Create holographic directory
    print("\n" + "=" * 70)
    print("Step 1: Create holographic/ module")
    print("=" * 70)
    
    holo_dir = Path('holographic')
    holo_dir.mkdir(exist_ok=True)
    print(f"✅ Created {holo_dir}/")
    
    # Create __init__.py if it doesn't exist
    init_file = holo_dir / '__init__.py'
    if not init_file.exists():
        init_content = '''"""
Holographic Module - Advanced QREA Upgrades
===========================================

Phase 3 & 4 upgrades for QREA system.
"""

from .learned_memory import LearnedEpisodicMemory, create_episodic_memory, Episode
from .neural_doctrine_engine import (
    NeuralDoctrineEngine,
    create_doctrine_engine,
    Doctrine,
    DoctrineType
)

__version__ = '0.1.0'

__all__ = [
    'LearnedEpisodicMemory',
    'create_episodic_memory',
    'Episode',
    'NeuralDoctrineEngine',
    'create_doctrine_engine',
    'Doctrine',
    'DoctrineType',
]
'''
        with open(init_file, 'w') as f:
            f.write(init_content)
        print(f"✅ Created {init_file}")
    
    # Step 2: Move files to holographic/
    print("\n" + "=" * 70)
    print("Step 2: Move files to holographic/")
    print("=" * 70)
    
    for filename in file_actions['move_to_holographic']:
        src = Path(filename)
        dst = holo_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✅ Copied {filename} → holographic/{filename}")
        else:
            print(f"⚠️  {filename} not found (skipping)")
    
    # Step 3: Rename complete files
    print("\n" + "=" * 70)
    print("Step 3: Rename complete files")
    print("=" * 70)
    
    for old_name, new_name in file_actions['renames'].items():
        src = Path(old_name)
        dst = Path(new_name)
        if src.exists():
            # Check if destination already exists
            if dst.exists():
                backup = dst.with_suffix(dst.suffix + '.backup')
                shutil.copy2(dst, backup)
                print(f"  ⚠️  {new_name} exists, backed up to {backup.name}")
            
            shutil.copy2(src, dst)
            print(f"✅ {old_name} → {new_name}")
        else:
            print(f"⚠️  {old_name} not found (skipping)")
    
    # Step 4: Fix imports
    print("\n" + "=" * 70)
    print("Step 4: Fix import statements")
    print("=" * 70)
    
    # Import fixes for each file
    import_fixes = {
        'warehouse_env.py': {
            # No fixes needed (this is the target file)
        },
        'mera_ppo_curriculum.py': {
            'from warehouse_env_fixed import': 'from warehouse_env import',
        },
        'visualize.py': {
            'from warehouse_env_fixed import': 'from warehouse_env import',
            'from mera_ppo_complete import': 'from mera_ppo_curriculum import',
        },
        'analysis.py': {
            'from warehouse_env_fixed import': 'from warehouse_env import',
            'from mera_ppo_complete import': 'from mera_ppo_curriculum import',
        },
        'qrea_cognitive.py': {
            # Already correct, but ensure holographic imports work
        }
    }
    
    for filename, replacements in import_fixes.items():
        filepath = Path(filename)
        if replacements:
            fix_imports_in_file(filepath, replacements)
    
    # Step 5: Create config.yaml if missing
    print("\n" + "=" * 70)
    print("Step 5: Ensure config.yaml exists")
    print("=" * 70)
    
    config_file = Path('config.yaml')
    if not config_file.exists():
        config_content = '''# QREA Configuration
environment:
  grid_size: [50, 50]
  num_robots: 8
  num_packages: 20
  max_episode_steps: 1000
  dt: 0.1
  sparse_rewards: false
  
  robot:
    radius: 0.5
    max_speed: 2.0
    max_acceleration: 1.0
    max_angular_velocity: 1.57
    battery_capacity: 1000.0
    battery_drain_idle: 0.1
    battery_drain_moving: 0.5
    battery_drain_carrying: 1.0
    charging_rate: 5.0
  
  sensors:
    lidar:
      num_rays: 32
      max_range: 10.0
      noise_std: 0.1
  
  package_spawn_rate: 0.1
  package_types:
    standard:
      weight: 5.0
      priority: 1
      reward: 100
    express:
      weight: 5.0
      priority: 2
      reward: 200
    fragile:
      weight: 3.0
      priority: 3
      reward: 300
  
  stations:
    pickup: [5, 5]
    delivery_1: [45, 5]
    delivery_2: [45, 45]
    charging: [5, 45]

agent:
  world_model:
    latent_dim: 256

learning:
  learning_rate: 0.0003
  grad_clip: 0.5
  actor_critic:
    discount: 0.99
    lambda_gae: 0.95
    entropy_weight: 0.01

uprt:
  field:
    grid_resolution: [20, 20]
    diffusion_coeff: 0.1
    decay_rate: 0.01
    update_rate: 0.1
  symbols:
    embedding_dim: 128
    num_prototypes: 50
    detection_threshold: 0.6
    emergence_threshold: 0.7
'''
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"✅ Created {config_file}")
    else:
        print(f"✅ {config_file} already exists")
    
    # Step 6: Create requirements.txt if missing
    print("\n" + "=" * 70)
    print("Step 6: Ensure requirements.txt exists")
    print("=" * 70)
    
    req_file = Path('requirements.txt')
    if not req_file.exists():
        req_content = '''# QREA Requirements
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0
pyyaml>=6.0
matplotlib>=3.7.0
scipy>=1.10.0
imageio>=2.31.0
seaborn>=0.12.0
pandas>=2.0.0
'''
        with open(req_file, 'w') as f:
            f.write(req_content)
        print(f"✅ Created {req_file}")
    else:
        print(f"✅ {req_file} already exists")
    
    # Step 7: Verify structure
    print("\n" + "=" * 70)
    print("Step 7: Verify final structure")
    print("=" * 70)
    
    expected_files = {
        'Core': [
            'config.yaml',
            'requirements.txt',
            'mera_enhanced.py',
            'warehouse_env.py',
            'warehouse_uprt.py',
        ],
        'Training': [
            'mera_ppo_warehouse.py',  # Original (simple)
            'mera_ppo_curriculum.py',  # Enhanced (curriculum)
            'run_experiments.py',
        ],
        'Advanced': [
            'qrea_cognitive.py',
            'visualize.py',
            'analysis.py',
        ],
        'Holographic': [
            'holographic/__init__.py',
            'holographic/learned_memory.py',
            'holographic/neural_doctrine_engine.py',
        ]
    }
    
    print("\nFinal structure:")
    all_present = True
    for category, files in expected_files.items():
        print(f"\n{category}:")
        for filename in files:
            filepath = Path(filename)
            if filepath.exists():
                print(f"  ✅ {filename}")
            else:
                print(f"  ❌ {filename} MISSING")
                all_present = False
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE!")
    print("=" * 70)
    
    if all_present:
        print("\n✅ All files in place!")
    else:
        print("\n⚠️  Some files missing (see above)")
    
    print("\nYour unified codebase structure:")
    print("""
qrea_project/
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
│
# Core
├── mera_enhanced.py              # MERA tensor network
├── warehouse_env.py              # Enhanced environment (curriculum support)
├── warehouse_uprt.py             # UPRT fields
│
# Training
├── mera_ppo_warehouse.py         # Basic PPO (clean, simple)
├── mera_ppo_curriculum.py        # Advanced PPO (curriculum learning)
├── run_experiments.py            # Multi-seed experiments
│
# Advanced
├── qrea_cognitive.py             # Cognitive layer
├── visualize.py                  # Visualization system
├── analysis.py                   # Statistical analysis
│
# Holographic
└── holographic/
    ├── __init__.py
    ├── learned_memory.py
    └── neural_doctrine_engine.py
    """)
    
    print("\nQuick test:")
    print("  python -c \"from warehouse_env import WarehouseEnv; print('✅ warehouse_env')\"")
    print("  python -c \"from holographic.learned_memory import LearnedEpisodicMemory; print('✅ holographic')\"")
    
    print("\nUsage:")
    print("  # Basic training")
    print("  python mera_ppo_warehouse.py --quick_test")
    print()
    print("  # Curriculum training")
    print("  python mera_ppo_curriculum.py --curriculum --level 1")
    print()
    print("  # Visualization")
    print("  python visualize.py --checkpoint model.pt --save video.mp4")
    print()
    print("  # Experiments")
    print("  python run_experiments.py --compare all --seeds 5")


if __name__ == "__main__":
    integrate_codebase()
