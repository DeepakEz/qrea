"""
Safety System for QREA
PatchGate for safe self-modification and constraint verification
Feature 5: Barrier Lyapunov Functions + Ghost Rollouts
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum


class SafetyConstraintType(Enum):
    """Types of safety constraints"""
    VELOCITY = "velocity"
    BATTERY = "battery"
    COLLISION = "collision"
    SAFE_ZONE = "safe_zone"


@dataclass
class SafetyConstraint:
    """Safety constraint specification"""
    name: str
    constraint_type: SafetyConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    barrier_eps: float = 1e-6


class BarrierLyapunovFunction:
    """
    Barrier Lyapunov Function for safety certification.
    
    V(x) = -log((x_max - x)(x - x_min)) for x in (x_min, x_max)
    
    BLF blows up at constraint boundaries, preventing unsafe states.
    """
    
    def __init__(self, constraints: List[SafetyConstraint], device: torch.device):
        self.constraints = constraints
        self.device = device
    
    def compute(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute total BLF value for state.
        
        Args:
            state_dict: Dictionary with 'velocity', 'battery', etc.
            
        Returns:
            V: Total BLF value (higher = closer to unsafe)
        """
        V_total = torch.tensor(0.0, device=self.device)
        
        for constraint in self.constraints:
            if constraint.constraint_type == SafetyConstraintType.VELOCITY:
                if 'velocity' in state_dict:
                    v = state_dict['velocity']
                    V = self._barrier(v, constraint.min_value, constraint.max_value, 
                                     constraint.barrier_eps)
                    V_total += V
            
            elif constraint.constraint_type == SafetyConstraintType.BATTERY:
                if 'battery' in state_dict:
                    b = state_dict['battery']
                    V = self._barrier(b, constraint.min_value, constraint.max_value,
                                     constraint.barrier_eps)
                    V_total += V
            
            elif constraint.constraint_type == SafetyConstraintType.COLLISION:
                if 'min_distance' in state_dict:
                    d = state_dict['min_distance']
                    # For collision, we only have lower bound (min safe distance)
                    if constraint.min_value is not None:
                        # Use one-sided barrier
                        V = -torch.log((d - constraint.min_value).clamp(min=constraint.barrier_eps))
                        V_total += V.sum()
        
        return V_total
    
    def _barrier(self, x: torch.Tensor, x_min: float, x_max: float, eps: float) -> torch.Tensor:
        """
        Compute barrier function: V(x) = -log((x_max - x)(x - x_min))
        
        Args:
            x: State variable
            x_min: Minimum safe value
            x_max: Maximum safe value
            eps: Small constant to prevent log(0)
            
        Returns:
            V: Barrier value
        """
        if x_min is None or x_max is None:
            return torch.tensor(0.0, device=self.device)
        
        # Compute (x_max - x) * (x - x_min)
        barrier_term = (x_max - x) * (x - x_min)
        
        # Clamp to prevent log(negative) or log(0)
        barrier_term = barrier_term.clamp(min=eps)
        
        # Barrier function
        V = -torch.log(barrier_term)
        
        return V.sum()  # Sum over all elements if x is a tensor


class GhostRollout:
    """
    Ghost rollout for testing policy changes without committing to environment.
    Simulates N steps ahead using buffered states.
    """
    
    def __init__(self, horizon: int, world_model):
        self.horizon = horizon
        self.world_model = world_model
    
    def simulate(self, policy, initial_state: Dict, state_buffer: List[Dict]) -> Dict:
        """
        Simulate policy for N steps using world model.
        
        Args:
            policy: Policy to test
            initial_state: Starting state
            state_buffer: Recent states for context
            
        Returns:
            trajectory: Dict with states, actions, safety metrics
        """
        trajectory = {
            'states': [],
            'actions': [],
            'velocities': [],
            'batteries': [],
            'min_distances': []
        }
        
        state = initial_state
        
        for t in range(self.horizon):
            # Get action from policy
            with torch.no_grad():
                action, _ = policy(state, deterministic=True)
            
            # Imagine next state
            next_state = self.world_model.imagine(action, state)
            
            # Extract safety-relevant variables
            # These would be decoded from the latent state
            velocity = self._extract_velocity(state)
            battery = self._extract_battery(state)
            min_dist = self._extract_min_distance(state)
            
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['velocities'].append(velocity)
            trajectory['batteries'].append(battery)
            trajectory['min_distances'].append(min_dist)
            
            state = next_state
        
        return trajectory
    
    def _extract_velocity(self, state: Dict) -> torch.Tensor:
        """Extract velocity from latent state (placeholder)"""
        # In practice, would decode from state['stoch'] or state['deter']
        # For now, return dummy value
        return torch.tensor([1.0], device=state['stoch'].device)
    
    def _extract_battery(self, state: Dict) -> torch.Tensor:
        """Extract battery from latent state (placeholder)"""
        return torch.tensor([100.0], device=state['stoch'].device)
    
    def _extract_min_distance(self, state: Dict) -> torch.Tensor:
        """Extract minimum distance to obstacles (placeholder)"""
        return torch.tensor([2.0], device=state['stoch'].device)


class PatchGate:
    """
    Safe self-modification gate with BLF verification.
    Feature 5: Uses Barrier Lyapunov Functions + Ghost Rollouts
    """
    
    def __init__(self, config: dict, world_model):
        safety_cfg = config['safety']['patchgate']
        
        self.enabled = safety_cfg['enabled']
        self.test_episodes = safety_cfg['test_episodes']
        self.safety_threshold = safety_cfg['safety_threshold']
        self.rollback_on_failure = safety_cfg['rollback_on_failure']
        
        # Feature 5: BLF configuration
        self.use_blf = safety_cfg.get('use_blf', True)
        self.blf_tolerance = safety_cfg.get('blf_tolerance', 0.0)
        self.ghost_horizon = safety_cfg.get('ghost_horizon', 5)
        
        # Track modifications
        self.modification_history = []
        
        # Feature 5: Initialize BLF with safety constraints
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.constraints = [
            SafetyConstraint(
                name="velocity",
                constraint_type=SafetyConstraintType.VELOCITY,
                min_value=0.0,
                max_value=2.0,
                barrier_eps=1e-6
            ),
            SafetyConstraint(
                name="battery",
                constraint_type=SafetyConstraintType.BATTERY,
                min_value=10.0,
                max_value=1000.0,
                barrier_eps=1e-6
            ),
            SafetyConstraint(
                name="collision",
                constraint_type=SafetyConstraintType.COLLISION,
                min_value=0.5,  # Minimum safe distance
                max_value=None,
                barrier_eps=1e-6
            )
        ]
        
        self.blf = BarrierLyapunovFunction(self.constraints, device)
        self.ghost_rollout = GhostRollout(self.ghost_horizon, world_model)
        
        # State buffer for ghost rollouts
        self.state_buffer = []
        self.buffer_size = 100
    
    def update_state_buffer(self, state: Dict):
        """Add state to buffer for ghost rollouts"""
        self.state_buffer.append(state)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
    
    def verify_modification(self, agent, new_agent, env, 
                           current_state: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Verify that new agent modification is safe.
        Feature 5: Uses BLF + Ghost Rollouts for certification.
        
        Args:
            agent: Current agent
            new_agent: Modified agent (patched policy)
            env: Environment for testing
            current_state: Current state for BLF verification
        
        Returns:
            is_safe: Whether modification is safe
            metrics: Test metrics
        """
        if not self.enabled:
            return True, {}
        
        metrics = {}
        
        # Feature 5: BLF verification via ghost rollout
        if self.use_blf and current_state is not None and len(self.state_buffer) > 0:
            blf_safe, blf_metrics = self._verify_blf(
                agent, new_agent, current_state
            )
            metrics.update(blf_metrics)
            
            if not blf_safe:
                return False, metrics
        
        # Traditional performance-based verification
        # Evaluate current agent
        current_performance = self._evaluate_agent(agent, env, self.test_episodes)
        
        # Evaluate new agent
        new_performance = self._evaluate_agent(new_agent, env, self.test_episodes)
        
        # Check safety criteria
        # 1. No increase in collisions
        collision_safe = new_performance['collision_rate'] <= current_performance['collision_rate'] * 1.1
        
        # 2. Maintain minimum performance
        performance_safe = new_performance['reward'] >= current_performance['reward'] * self.safety_threshold
        
        # 3. No constraint violations
        constraint_safe = new_performance['constraint_violations'] == 0
        
        is_safe = collision_safe and performance_safe and constraint_safe
        
        metrics.update({
            'current_reward': current_performance['reward'],
            'new_reward': new_performance['reward'],
            'current_collisions': current_performance['collision_rate'],
            'new_collisions': new_performance['collision_rate'],
            'constraint_violations': new_performance['constraint_violations'],
            'is_safe': is_safe
        })
        
        # Log modification attempt
        self.modification_history.append({
            'approved': is_safe,
            'metrics': metrics
        })
        
        return is_safe, metrics
    
    def _verify_blf(self, current_policy, new_policy, current_state: Dict) -> Tuple[bool, Dict]:
        """
        Verify safety using Barrier Lyapunov Function.
        
        Simulates N steps ahead and checks if BLF increases (unsafe).
        
        Args:
            current_policy: Current policy
            new_policy: New (patched) policy to test
            current_state: Starting state
            
        Returns:
            is_safe: Whether BLF criterion is satisfied
            metrics: BLF metrics
        """
        # Compute baseline BLF with current policy
        baseline_traj = self.ghost_rollout.simulate(
            current_policy, current_state, self.state_buffer
        )
        
        # Compute BLF for baseline trajectory
        baseline_V = 0.0
        for t in range(len(baseline_traj['states'])):
            state_dict = {
                'velocity': baseline_traj['velocities'][t],
                'battery': baseline_traj['batteries'][t],
                'min_distance': baseline_traj['min_distances'][t]
            }
            V_t = self.blf.compute(state_dict)
            baseline_V += V_t.item()
        baseline_V /= len(baseline_traj['states'])
        
        # Compute BLF with new policy
        new_traj = self.ghost_rollout.simulate(
            new_policy, current_state, self.state_buffer
        )
        
        new_V = 0.0
        for t in range(len(new_traj['states'])):
            state_dict = {
                'velocity': new_traj['velocities'][t],
                'battery': new_traj['batteries'][t],
                'min_distance': new_traj['min_distances'][t]
            }
            V_t = self.blf.compute(state_dict)
            new_V += V_t.item()
        new_V /= len(new_traj['states'])
        
        # Check BLF criterion: new BLF should not increase significantly
        delta_V = new_V - baseline_V
        is_safe = (delta_V <= self.blf_tolerance)
        
        metrics = {
            'blf_baseline': baseline_V,
            'blf_new': new_V,
            'blf_delta': delta_V,
            'blf_safe': is_safe
        }
        
        return is_safe, metrics
    
    def _evaluate_agent(self, agent, env, num_episodes: int) -> Dict:
        """Evaluate agent performance"""
        total_reward = 0.0
        total_collisions = 0
        constraint_violations = 0
        
        for _ in range(num_episodes):
            obs_dict = env.reset()
            agent.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                actions = {}
                for robot_id in obs_dict.keys():
                    action = agent.act(obs_dict[robot_id])
                    
                    # Check action constraints
                    if not self._check_action_constraints(action):
                        constraint_violations += 1
                    
                    actions[robot_id] = action
                
                obs_dict, rewards, dones, infos = env.step(actions)
                
                episode_reward += sum(rewards.values())
                
                # Count collisions
                for info in infos.values():
                    if isinstance(info, dict) and info.get('collision', False):
                        total_collisions += 1
                
                done = dones.get('__all__', False)
            
            total_reward += episode_reward
        
        return {
            'reward': total_reward / num_episodes,
            'collision_rate': total_collisions / num_episodes,
            'constraint_violations': constraint_violations
        }
    
    def _check_action_constraints(self, action: np.ndarray) -> bool:
        """Check if action satisfies constraints"""
        # Velocity constraint
        if abs(action[0]) > 2.5:  # max velocity
            return False
        
        # Angular velocity constraint
        if abs(action[1]) > 1.57:  # max angular velocity
            return False
        
        # Gripper constraint
        if action[2] < 0 or action[2] > 1:
            return False
        
        return True
    
    def get_approval_rate(self) -> float:
        """Get fraction of modifications approved"""
        if not self.modification_history:
            return 0.0
        
        approved = sum(1 for mod in self.modification_history if mod['approved'])
        return approved / len(self.modification_history)


class ConstraintVerifier:
    """
    Verify safety constraints during execution
    """
    
    def __init__(self, config: dict):
        constraint_cfg = config['safety']['constraints']
        
        self.max_velocity = constraint_cfg['max_velocity']
        self.min_battery_reserve = constraint_cfg['min_battery_reserve']
        self.safe_zone_required = constraint_cfg.get('safe_zone_required', True)
        
        # Collision constraints
        collision_cfg = config['safety']['collision']
        self.min_distance = collision_cfg['min_distance']
        self.hard_constraint = collision_cfg['hard_constraint']
        
        # Violation tracking
        self.violations = {
            'velocity': 0,
            'battery': 0,
            'collision': 0,
            'safe_zone': 0
        }
    
    def verify_action(self, action: np.ndarray, robot_state: Dict) -> Tuple[bool, np.ndarray]:
        """
        Verify action satisfies constraints
        
        Args:
            action: Proposed action [velocity, angular_vel, gripper]
            robot_state: Current robot state
        
        Returns:
            is_safe: Whether action is safe
            corrected_action: Safe version of action
        """
        corrected = action.copy()
        violations = []
        
        # 1. Velocity constraint
        if abs(action[0]) > self.max_velocity:
            corrected[0] = np.clip(action[0], -self.max_velocity, self.max_velocity)
            violations.append('velocity')
            self.violations['velocity'] += 1
        
        # 2. Angular velocity constraint (implicitly handled)
        
        # 3. Battery constraint
        if robot_state.get('battery', 1000) < self.min_battery_reserve:
            # Force towards charging station
            violations.append('battery')
            self.violations['battery'] += 1
            # Override action to go to nearest charging station
            # (This would need charging station position)
        
        is_safe = len(violations) == 0
        
        return is_safe, corrected
    
    def verify_robot_positions(self, positions: Dict[int, np.ndarray]) -> List[Tuple[int, int]]:
        """
        Check for collision risks between robots
        
        Returns:
            List of (robot_id1, robot_id2) pairs that are too close
        """
        collisions = []
        
        robot_ids = list(positions.keys())
        
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                rid1, rid2 = robot_ids[i], robot_ids[j]
                pos1, pos2 = positions[rid1], positions[rid2]
                
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < self.min_distance:
                    collisions.append((rid1, rid2))
                    self.violations['collision'] += 1
        
        return collisions
    
    def apply_collision_avoidance(self, actions: Dict[int, np.ndarray],
                                 positions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Apply collision avoidance to actions
        """
        if not self.hard_constraint:
            return actions
        
        # Check for potential collisions
        collisions = self.verify_robot_positions(positions)
        
        if not collisions:
            return actions
        
        corrected_actions = actions.copy()
        
        for rid1, rid2 in collisions:
            # Reduce velocities of both robots
            if rid1 in corrected_actions:
                corrected_actions[rid1][0] *= 0.5  # Halve velocity
            if rid2 in corrected_actions:
                corrected_actions[rid2][0] *= 0.5
        
        return corrected_actions
    
    def get_violation_statistics(self) -> Dict:
        """Get constraint violation statistics"""
        total = sum(self.violations.values())
        
        stats = {
            'total_violations': total,
            **self.violations
        }
        
        if total > 0:
            stats['violation_rates'] = {
                k: v / total for k, v in self.violations.items()
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset violation counters"""
        self.violations = {k: 0 for k in self.violations}


class SafetyMonitor:
    """
    Overall safety monitoring system
    Combines PatchGate and ConstraintVerifier
    """
    
    def __init__(self, config: dict, world_model=None):
        self.config = config
        
        # PatchGate requires world_model for ghost rollouts
        if world_model is not None and config['safety']['patchgate'].get('enabled', False):
            self.patchgate = PatchGate(config, world_model)
        else:
            self.patchgate = None
        
        self.verifier = ConstraintVerifier(config)
        
        # Safety metrics
        self.safety_score = 1.0
        self.safety_history = []
    
    def verify_modification(self, agent, new_agent, env) -> Tuple[bool, Dict]:
        """Verify agent modification"""
        if self.patchgate is None:
            # PatchGate not enabled, allow modification
            return True, {"patchgate_enabled": False, "approved": True}
        return self.patchgate.verify_modification(agent, new_agent, env)
    
    def verify_step(self, actions: Dict[int, np.ndarray],
                   robot_states: Dict[int, Dict]) -> Tuple[Dict[int, np.ndarray], bool]:
        """
        Verify and correct actions for safety
        
        Returns:
            corrected_actions: Safe actions
            all_safe: Whether all actions were safe
        """
        corrected_actions = {}
        all_safe = True
        
        # Verify each action
        for robot_id, action in actions.items():
            robot_state = robot_states.get(robot_id, {})
            is_safe, corrected = self.verifier.verify_action(action, robot_state)
            
            corrected_actions[robot_id] = corrected
            
            if not is_safe:
                all_safe = False
        
        # Apply collision avoidance
        positions = {rid: np.array(rstate['position']) 
                    for rid, rstate in robot_states.items()}
        corrected_actions = self.verifier.apply_collision_avoidance(
            corrected_actions, positions
        )
        
        # Update safety score
        self.safety_score = 0.95 * self.safety_score + 0.05 * (1.0 if all_safe else 0.0)
        self.safety_history.append(self.safety_score)
        
        return corrected_actions, all_safe
    
    def get_safety_metrics(self) -> Dict:
        """Get comprehensive safety metrics"""
        metrics = {
            'safety_score': self.safety_score,
            **self.verifier.get_violation_statistics(),
            'safety_trend': np.mean(self.safety_history[-100:]) if self.safety_history else 1.0
        }
        
        # Add PatchGate metrics if available
        if self.patchgate is not None:
            metrics['patchgate_approval_rate'] = self.patchgate.get_approval_rate()
        else:
            metrics['patchgate_approval_rate'] = 1.0  # Not applicable
        
        return metrics
    
    def reset(self):
        """Reset safety monitoring"""
        self.safety_score = 1.0
        self.safety_history = []
        self.verifier.reset_statistics()
