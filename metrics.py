"""
Comprehensive Metrics Tracking for QREA Warehouse System
Tracks performance, learning progress, safety, and emergent behaviors
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class MetricBuffer:
    """Circular buffer for metric history"""
    maxlen: int = 1000
    values: deque = field(default_factory=deque)
    
    def __post_init__(self):
        self.values = deque(maxlen=self.maxlen)
    
    def append(self, value: float):
        self.values.append(value)
    
    def mean(self) -> float:
        return np.mean(self.values) if self.values else 0.0
    
    def std(self) -> float:
        return np.std(self.values) if self.values else 0.0
    
    def min(self) -> float:
        return np.min(self.values) if self.values else 0.0
    
    def max(self) -> float:
        return np.max(self.values) if self.values else 0.0
    
    def latest(self) -> float:
        return self.values[-1] if self.values else 0.0


class MetricsTracker:
    """Comprehensive metrics tracking system"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics = defaultdict(lambda: MetricBuffer(buffer_size))
        self.episode_metrics = []
        self.step_count = 0
        self.episode_count = 0
        
    def update(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Update metrics with new values"""
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
            
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(float(value))
    
    def get_summary(self, keys: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics"""
        if keys is None:
            keys = list(self.metrics.keys())
        
        summary = {}
        for key in keys:
            if key in self.metrics:
                buffer = self.metrics[key]
                summary[key] = {
                    'mean': buffer.mean(),
                    'std': buffer.std(),
                    'min': buffer.min(),
                    'max': buffer.max(),
                    'latest': buffer.latest()
                }
        return summary
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric"""
        return self.metrics[key].latest()
    
    def get_mean(self, key: str, window: Optional[int] = None) -> float:
        """Get mean of metric over window"""
        if key not in self.metrics:
            return 0.0
        
        values = list(self.metrics[key].values)
        if not values:
            return 0.0
        
        if window is not None:
            values = values[-window:]
        
        return np.mean(values)
    
    def end_episode(self, episode_info: Dict[str, float]):
        """Record end-of-episode metrics"""
        episode_info['episode'] = self.episode_count
        episode_info['step'] = self.step_count
        self.episode_metrics.append(episode_info)
        self.episode_count += 1
    
    def save(self, path: str):
        """Save metrics to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'episode_metrics': self.episode_metrics,
            'current_metrics': {
                key: {
                    'values': list(buffer.values),
                    'mean': buffer.mean(),
                    'std': buffer.std()
                }
                for key, buffer in self.metrics.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.step_count = data['step_count']
        self.episode_count = data['episode_count']
        self.episode_metrics = data['episode_metrics']
        
        for key, metric_data in data['current_metrics'].items():
            buffer = MetricBuffer(self.buffer_size)
            buffer.values.extend(metric_data['values'])
            self.metrics[key] = buffer


class PerformanceMetrics:
    """Track warehouse-specific performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.packages_delivered = 0
        self.packages_attempted = 0
        self.total_distance = 0.0
        self.collisions = 0
        self.deadline_misses = 0
        self.energy_consumed = 0.0
        self.idle_time = 0.0
        self.active_time = 0.0
        
        # SLA tracking
        self.priority_1_delivered = 0
        self.priority_2_delivered = 0
        self.priority_3_delivered = 0
        self.priority_1_total = 0
        self.priority_2_total = 0
        self.priority_3_total = 0
        
        # Timing
        self.delivery_times = []
        self.wait_times = []
    
    def update_delivery(self, priority: int, delivery_time: float, 
                       deadline: float, distance: float):
        """Update metrics for a delivery"""
        self.packages_delivered += 1
        self.total_distance += distance
        self.delivery_times.append(delivery_time)
        
        # Track by priority
        if priority == 1:
            self.priority_1_delivered += 1
            self.priority_1_total += 1
        elif priority == 2:
            self.priority_2_delivered += 1
            self.priority_2_total += 1
        else:
            self.priority_3_delivered += 1
            self.priority_3_total += 1
        
        # Check deadline
        if delivery_time > deadline:
            self.deadline_misses += 1
    
    def update_attempt(self, priority: int):
        """Update when package delivery is attempted"""
        self.packages_attempted += 1
        if priority == 1:
            self.priority_1_total += 1
        elif priority == 2:
            self.priority_2_total += 1
        else:
            self.priority_3_total += 1
    
    def add_collision(self):
        """Record a collision"""
        self.collisions += 1
    
    def add_energy(self, energy: float):
        """Add energy consumption"""
        self.energy_consumed += energy
    
    def add_time(self, active: float, idle: float):
        """Add time tracking"""
        self.active_time += active
        self.idle_time += idle
    
    def get_throughput(self, time_elapsed: float) -> float:
        """Packages per unit time"""
        return self.packages_delivered / max(time_elapsed, 1e-6)
    
    def get_success_rate(self) -> float:
        """Delivery success rate"""
        if self.packages_attempted == 0:
            return 0.0
        return self.packages_delivered / self.packages_attempted
    
    def get_sla_compliance(self) -> Dict[str, float]:
        """SLA compliance by priority"""
        return {
            'priority_1': (self.priority_1_delivered / max(self.priority_1_total, 1)),
            'priority_2': (self.priority_2_delivered / max(self.priority_2_total, 1)),
            'priority_3': (self.priority_3_delivered / max(self.priority_3_total, 1)),
            'overall': self.get_success_rate()
        }
    
    def get_efficiency(self) -> float:
        """Energy efficiency (packages per energy unit)"""
        if self.energy_consumed == 0:
            return 0.0
        return self.packages_delivered / self.energy_consumed
    
    def get_avg_delivery_time(self) -> float:
        """Average delivery time"""
        return np.mean(self.delivery_times) if self.delivery_times else 0.0
    
    def get_deadline_compliance(self) -> float:
        """Percentage of deliveries meeting deadline"""
        if self.packages_delivered == 0:
            return 0.0
        return 1.0 - (self.deadline_misses / self.packages_delivered)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'packages_delivered': self.packages_delivered,
            'packages_attempted': self.packages_attempted,
            'success_rate': self.get_success_rate(),
            'total_distance': self.total_distance,
            'collisions': self.collisions,
            'collision_rate': self.collisions / max(self.packages_attempted, 1),
            'deadline_misses': self.deadline_misses,
            'deadline_compliance': self.get_deadline_compliance(),
            'energy_consumed': self.energy_consumed,
            'energy_efficiency': self.get_efficiency(),
            'idle_time': self.idle_time,
            'active_time': self.active_time,
            'utilization': self.active_time / max(self.active_time + self.idle_time, 1e-6),
            'avg_delivery_time': self.get_avg_delivery_time(),
            **{f'sla_{k}': v for k, v in self.get_sla_compliance().items()}
        }


class LearningMetrics:
    """Track learning-specific metrics"""
    
    def __init__(self):
        self.losses = defaultdict(list)
        self.gradients = defaultdict(list)
        self.learning_rates = []
        self.intrinsic_rewards = {
            'novelty': [],
            'competence': [],
            'empowerment': []
        }
    
    def update_loss(self, loss_dict: Dict[str, float]):
        """Update loss values"""
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.losses[key].append(value)
    
    def update_gradients(self, grad_dict: Dict[str, float]):
        """Update gradient norms"""
        for key, value in grad_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.gradients[key].append(value)
    
    def update_intrinsic(self, novelty: float, competence: float, empowerment: float):
        """Update intrinsic motivation metrics"""
        self.intrinsic_rewards['novelty'].append(novelty)
        self.intrinsic_rewards['competence'].append(competence)
        self.intrinsic_rewards['empowerment'].append(empowerment)
    
    def get_recent_loss(self, key: str, window: int = 100) -> float:
        """Get recent average loss"""
        if key not in self.losses or not self.losses[key]:
            return 0.0
        return np.mean(self.losses[key][-window:])
    
    def get_intrinsic_balance(self) -> Dict[str, float]:
        """Get balance of intrinsic rewards"""
        return {
            key: np.mean(values[-100:]) if values else 0.0
            for key, values in self.intrinsic_rewards.items()
        }


class EvolutionMetrics:
    """Track evolutionary progress"""
    
    def __init__(self):
        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []
        self.hgt_events = []
        self.mutation_rates = []
        self.best_genomes = []
    
    def update_generation(self, fitness_values: np.ndarray, diversity: float,
                         mutation_rate: float, best_genome: Optional[Dict] = None):
        """Update generation metrics"""
        self.generation += 1
        self.fitness_history.append({
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values),
            'max': np.max(fitness_values),
            'min': np.min(fitness_values),
            'median': np.median(fitness_values)
        })
        self.diversity_history.append(diversity)
        self.mutation_rates.append(mutation_rate)
        
        if best_genome is not None:
            self.best_genomes.append(best_genome)
    
    def record_hgt_event(self, donor_fitness: float, recipient_fitness: float,
                        improvement: float):
        """Record horizontal gene transfer event"""
        self.hgt_events.append({
            'generation': self.generation,
            'donor_fitness': donor_fitness,
            'recipient_fitness': recipient_fitness,
            'improvement': improvement
        })
    
    def get_improvement_rate(self, window: int = 10) -> float:
        """Get recent fitness improvement rate"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        recent = self.fitness_history[-window:]
        if len(recent) < 2:
            return 0.0
        
        start_fitness = recent[0]['max']
        end_fitness = recent[-1]['max']
        
        return (end_fitness - start_fitness) / len(recent)


class SafetyMetrics:
    """Track safety-related metrics"""
    
    def __init__(self):
        self.safety_violations = defaultdict(int)
        self.constraint_values = defaultdict(list)
        self.patch_gate_rejections = 0
        self.patch_gate_accepts = 0
        self.near_misses = 0
        
    def record_violation(self, violation_type: str):
        """Record a safety violation"""
        self.safety_violations[violation_type] += 1
    
    def record_constraint(self, constraint_name: str, value: float):
        """Record constraint value"""
        self.constraint_values[constraint_name].append(value)
    
    def record_patch_decision(self, accepted: bool):
        """Record PatchGate decision"""
        if accepted:
            self.patch_gate_accepts += 1
        else:
            self.patch_gate_rejections += 1
    
    def record_near_miss(self):
        """Record near-miss collision"""
        self.near_misses += 1
    
    def get_violation_rate(self) -> Dict[str, float]:
        """Get violation rates"""
        total = sum(self.safety_violations.values())
        if total == 0:
            return {}
        return {
            key: count / total 
            for key, count in self.safety_violations.items()
        }
    
    def get_patch_acceptance_rate(self) -> float:
        """Get PatchGate acceptance rate"""
        total = self.patch_gate_accepts + self.patch_gate_rejections
        if total == 0:
            return 0.0
        return self.patch_gate_accepts / total
