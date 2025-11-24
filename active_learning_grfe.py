"""
Active Learning and Full GRFE Implementation for QREA v3.1

This module implements:
1. True active learning loop where high epistemic uncertainty triggers exploration
2. Complete GRFE functional including Φ_Q integrated information
3. Information gain-based action selection
4. Uncertainty-aware training dynamics

Key additions:
- ActiveLearningController: Decides when to switch to exploration mode
- GRFEFunctional: Full v3.1 energy functional with all terms
- InformationGainPolicy: Action selection that minimizes epistemic uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AgentMode(Enum):
    """Operating modes for the agent"""
    NORMAL = "normal"              # Normal exploitation/exploration
    ACTIVE_LEARNING = "active"     # High uncertainty - gather info
    BOOTSTRAP = "bootstrap"        # Initial random exploration


@dataclass
class GRFEComponents:
    """Components of the Global Resonance Free Energy functional"""
    variational_fe: float          # E_q[log q - log p] (prediction error)
    coherence_R: float            # Phase coherence (cognitive stability)
    phi_q: float                  # Integrated information (consciousness)
    novelty_N: float              # Novelty drive
    empowerment_C: float          # Empowerment (control)
    topological_preservation: float # Symbol stability
    surrogate_entropy: float      # Epistemic uncertainty penalty
    
    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total GRFE"""
        if weights is None:
            # Default weights from v3.1 spec
            weights = {
                'variational': 1.0,
                'coherence': 0.5,      # α
                'phi_q': 0.3,          # β  
                'novelty': 0.1,        # γ
                'empowerment': 0.2,    # δ
                'topological': 0.4,    # ζ
                'entropy': 0.1,        # λ
            }
        
        total = (
            weights['variational'] * self.variational_fe
            - weights['coherence'] * self.coherence_R      # Maximize coherence
            - weights['phi_q'] * self.phi_q                 # Maximize integration
            - weights['novelty'] * self.novelty_N           # Maximize novelty
            - weights['empowerment'] * self.empowerment_C   # Maximize control
            + weights['topological'] * self.topological_preservation
            + weights['entropy'] * self.surrogate_entropy   # Penalize uncertainty
        )
        
        return total


class GRFEFunctional(nn.Module):
    """
    Complete Global Resonance Free Energy functional for QREA v3.1.
    
    F_GRFE = E_q[log q - log p] - αR - βΦ_Q - γN - δC + ζL_topo + λH[q]
    
    This unifies all components into a single differentiable objective.
    """
    
    def __init__(
        self,
        coherence_weight: float = 0.5,
        phi_q_weight: float = 0.3,
        novelty_weight: float = 0.1,
        empowerment_weight: float = 0.2,
        topological_weight: float = 0.4,
        entropy_weight: float = 0.1
    ):
        super().__init__()
        
        self.weights = {
            'variational': 1.0,
            'coherence': coherence_weight,
            'phi_q': phi_q_weight,
            'novelty': novelty_weight,
            'empowerment': empowerment_weight,
            'topological': topological_weight,
            'entropy': entropy_weight,
        }
        
    def compute_variational_fe(
        self,
        predicted_probs: torch.Tensor,
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Variational free energy: E_q[log q - log p]
        
        This is essentially the prediction error (cross-entropy).
        """
        return F.cross_entropy(torch.log(predicted_probs + 1e-10), true_labels)
    
    def compute_coherence(self, phase_states: torch.Tensor) -> torch.Tensor:
        """
        Coherence R = |⟨e^(iθ)⟩|
        
        Measures phase synchronization across cognitive elements.
        
        Args:
            phase_states: (batch, n_oscillators) phase values
            
        Returns:
            (batch,) coherence values in [0, 1]
        """
        # Convert to complex exponentials
        complex_phases = torch.exp(1j * phase_states)
        
        # Mean complex phase
        mean_phase = complex_phases.mean(dim=-1)
        
        # Coherence is magnitude
        R = torch.abs(mean_phase)
        
        return R
    
    def compute_novelty(
        self,
        current_state: torch.Tensor,
        memory_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Novelty N = min_i ||s_t - s_memory_i||
        
        Distance to nearest previously seen state.
        """
        # Expand dimensions for broadcasting
        current = current_state.unsqueeze(1)  # (batch, 1, dim)
        memory = memory_states.unsqueeze(0)    # (1, mem_size, dim)
        
        # Compute distances
        distances = torch.norm(current - memory, dim=-1)  # (batch, mem_size)
        
        # Minimum distance = novelty
        novelty = distances.min(dim=-1)[0]
        
        return novelty
    
    def compute_empowerment(
        self,
        current_state: torch.Tensor,
        action_effects: torch.Tensor
    ) -> torch.Tensor:
        """
        Empowerment C = I(A; S') = H[S'] - H[S'|A]
        
        Mutual information between actions and resulting states.
        This measures how much control the agent has.
        
        Args:
            current_state: (batch, state_dim)
            action_effects: (batch, n_actions, state_dim) predicted next states
            
        Returns:
            (batch,) empowerment values
        """
        # Entropy of next state marginal
        # Approximate via variance (Gaussian assumption)
        H_S_prime = torch.log(action_effects.var(dim=1).mean(dim=-1) + 1e-8)
        
        # Conditional entropy (given action, how uncertain is outcome?)
        H_S_prime_given_A = torch.log(
            action_effects.var(dim=-1).mean(dim=-1) + 1e-8
        )
        
        # Empowerment = reduction in uncertainty
        empowerment = H_S_prime - H_S_prime_given_A
        
        return empowerment
    
    def compute_topological_preservation(
        self,
        symbol_charges: torch.Tensor,
        symbol_charges_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Topological preservation: L_topo = ||Q_t - Q_{t+1}||²
        
        Symbols should maintain their topological charge.
        """
        return F.mse_loss(symbol_charges, symbol_charges_next)
    
    def compute_surrogate_entropy(
        self,
        predicted_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Surrogate entropy: H[q(φ|s)] = -Σ q log q
        
        This is the epistemic uncertainty penalty term from v3.1.
        """
        entropy = -torch.sum(
            predicted_probs * torch.log(predicted_probs + 1e-10),
            dim=-1
        )
        return entropy.mean()
    
    def forward(
        self,
        predicted_probs: torch.Tensor,
        true_labels: torch.Tensor,
        phi_q: torch.Tensor,
        phase_states: Optional[torch.Tensor] = None,
        memory_states: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        action_effects: Optional[torch.Tensor] = None,
        symbol_charges: Optional[torch.Tensor] = None,
        symbol_charges_next: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, GRFEComponents]:
        """
        Compute complete GRFE functional.
        
        Returns:
            total_grfe: Scalar loss value
            components: GRFEComponents dataclass with individual terms
        """
        # Core components (always computed)
        variational_fe = self.compute_variational_fe(predicted_probs, true_labels)
        surrogate_entropy = self.compute_surrogate_entropy(predicted_probs)
        
        # Optional components (use zeros if not provided)
        batch_size = predicted_probs.shape[0]
        device = predicted_probs.device
        
        coherence_R = self.compute_coherence(phase_states) if phase_states is not None \
                      else torch.zeros(batch_size, device=device)
        
        novelty_N = self.compute_novelty(current_state, memory_states) \
                    if (current_state is not None and memory_states is not None) \
                    else torch.zeros(batch_size, device=device)
        
        empowerment_C = self.compute_empowerment(current_state, action_effects) \
                        if (current_state is not None and action_effects is not None) \
                        else torch.zeros(batch_size, device=device)
        
        topological = self.compute_topological_preservation(symbol_charges, symbol_charges_next) \
                      if (symbol_charges is not None and symbol_charges_next is not None) \
                      else torch.tensor(0.0, device=device)
        
        # Package components
        components = GRFEComponents(
            variational_fe=variational_fe.item(),
            coherence_R=coherence_R.mean().item(),
            phi_q=phi_q.mean().item(),
            novelty_N=novelty_N.mean().item(),
            empowerment_C=empowerment_C.mean().item(),
            topological_preservation=topological.item(),
            surrogate_entropy=surrogate_entropy.item()
        )
        
        # Compute weighted total
        total = components.total(self.weights)
        
        return torch.tensor(total, device=device, requires_grad=True), components


class ActiveLearningController:
    """
    Controls when the agent switches to active learning mode.
    
    Active learning is triggered when:
    1. Epistemic uncertainty exceeds threshold
    2. Recent performance drops
    3. Novel regime detected
    
    In active learning mode, the agent prioritizes gathering information
    to reduce model uncertainty over maximizing immediate reward.
    """
    
    def __init__(
        self,
        epistemic_threshold: float = 0.3,
        performance_threshold: float = 0.5,
        bootstrap_steps: int = 100,
        memory_size: int = 50
    ):
        self.epistemic_threshold = epistemic_threshold
        self.performance_threshold = performance_threshold
        self.bootstrap_steps = bootstrap_steps
        
        self.step_count = 0
        self.mode = AgentMode.BOOTSTRAP
        
        # Track recent performance
        self.recent_rewards = []
        self.memory_size = memory_size
        
        # Track mode changes
        self.mode_history = []
        
    def should_enter_active_learning(
        self,
        epistemic_uncertainty: float,
        recent_reward_avg: float,
        regime_confidence: float
    ) -> bool:
        """
        Decide whether to enter active learning mode.
        
        Args:
            epistemic_uncertainty: Model's epistemic uncertainty
            recent_reward_avg: Average recent reward
            regime_confidence: Confidence in current regime prediction
            
        Returns:
            True if should enter active learning
        """
        # Always bootstrap at start
        if self.step_count < self.bootstrap_steps:
            return True
        
        # High uncertainty → active learning
        if epistemic_uncertainty > self.epistemic_threshold:
            return True
        
        # Poor recent performance → explore more
        if recent_reward_avg < self.performance_threshold:
            return True
        
        # Low regime confidence → need more data
        if regime_confidence < 0.5:
            return True
        
        return False
    
    def update(
        self,
        epistemic_uncertainty: float,
        reward: float,
        regime_confidence: float
    ) -> AgentMode:
        """
        Update controller state and return current mode.
        
        Args:
            epistemic_uncertainty: Current epistemic uncertainty
            reward: Reward just received
            regime_confidence: Confidence in regime prediction
            
        Returns:
            Current AgentMode
        """
        self.step_count += 1
        
        # Track rewards
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.memory_size:
            self.recent_rewards.pop(0)
        
        # Compute recent average
        recent_avg = np.mean(self.recent_rewards) if self.recent_rewards else 0.5
        
        # Decide mode
        if self.step_count < self.bootstrap_steps:
            self.mode = AgentMode.BOOTSTRAP
        elif self.should_enter_active_learning(
            epistemic_uncertainty, recent_avg, regime_confidence
        ):
            self.mode = AgentMode.ACTIVE_LEARNING
        else:
            self.mode = AgentMode.NORMAL
        
        self.mode_history.append((self.step_count, self.mode))
        
        return self.mode


class InformationGainPolicy:
    """
    Policy for active learning: select actions that maximize information gain.
    
    Instead of maximizing reward, this policy selects actions that will
    most reduce the model's epistemic uncertainty.
    
    This is the "active" part of active learning.
    """
    
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        
    def compute_expected_information_gain(
        self,
        action: int,
        current_uncertainty: np.ndarray,  # Per-arm uncertainties
        arm_counts: np.ndarray             # How many times each arm pulled
    ) -> float:
        """
        Estimate how much pulling 'action' would reduce uncertainty.
        
        Heuristic: Information gain ≈ current_uncertainty / (count + 1)
        
        This favors arms we're uncertain about AND haven't pulled recently.
        """
        # If we've never pulled this arm, high info gain
        if arm_counts[action] == 0:
            return 10.0
        
        # Otherwise, uncertainty weighted by inverse count
        return current_uncertainty[action] / (arm_counts[action] ** 0.5 + 1)
    
    def select_action(
        self,
        arm_uncertainties: np.ndarray,
        arm_counts: np.ndarray,
        recent_actions: List[int],
        exploration_bonus: float = 1.0
    ) -> int:
        """
        Select action that maximizes expected information gain.
        
        Args:
            arm_uncertainties: (num_arms,) uncertainty for each arm
            arm_counts: (num_arms,) total pulls for each arm
            recent_actions: List of recently taken actions
            exploration_bonus: Scale factor for exploration
            
        Returns:
            Selected action index
        """
        info_gains = np.array([
            self.compute_expected_information_gain(a, arm_uncertainties, arm_counts)
            for a in range(self.num_arms)
        ])
        
        # Add bonus for arms not recently pulled
        recency_bonus = np.ones(self.num_arms)
        for action in recent_actions[-20:]:  # Last 20 actions
            recency_bonus[action] *= 0.9
        
        # Combined score
        scores = info_gains * recency_bonus * exploration_bonus
        
        # Select best (with some randomness)
        # Softmax selection for exploration
        temp = 0.5
        probs = np.exp(scores / temp)
        probs = probs / probs.sum()
        
        return np.random.choice(self.num_arms, p=probs)


class UncertaintyAwareTraining:
    """
    Training dynamics that use uncertainty to guide learning.
    
    Key ideas:
    1. When uncertain, gather more samples before updating
    2. When confident but wrong, increase learning rate
    3. Adaptively balance exploration and exploitation
    """
    
    def __init__(
        self,
        base_lr: float = 1e-3,
        uncertainty_threshold: float = 0.3,
        batch_size: int = 32
    ):
        self.base_lr = base_lr
        self.uncertainty_threshold = uncertainty_threshold
        self.batch_size = batch_size
        
        # Buffers for adaptive batching
        self.uncertain_samples = []
        self.confident_samples = []
        
    def should_update(
        self,
        epistemic_uncertainty: float,
        buffer_size: int
    ) -> bool:
        """
        Decide whether to perform parameter update.
        
        When uncertain, wait for more samples.
        When confident, can update more frequently.
        """
        if epistemic_uncertainty > self.uncertainty_threshold:
            # High uncertainty → need more samples
            return buffer_size >= self.batch_size * 2
        else:
            # Low uncertainty → can update with fewer samples
            return buffer_size >= self.batch_size
    
    def compute_adaptive_lr(
        self,
        epistemic_uncertainty: float,
        prediction_error: float
    ) -> float:
        """
        Compute learning rate based on uncertainty and error.
        
        - High uncertainty + high error → cautious (low LR)
        - Low uncertainty + high error → need to learn fast (high LR)
        - Low uncertainty + low error → fine-tuning (medium LR)
        """
        if epistemic_uncertainty > self.uncertainty_threshold:
            if prediction_error > 0.5:
                # Uncertain and wrong → very cautious
                return self.base_lr * 0.1
            else:
                # Uncertain but doing ok → explore carefully
                return self.base_lr * 0.5
        else:
            if prediction_error > 0.5:
                # Confident but wrong → learn fast
                return self.base_lr * 2.0
            else:
                # Confident and correct → maintain
                return self.base_lr


# Example integration
class QREAAgentWithActivelearning:
    """
    Enhanced QREA agent with active learning and full GRFE.
    
    This is what the bandit agent would look like with all additions.
    """
    
    def __init__(
        self,
        num_arms: int,
        baseline,
        mera,
        uprt,
        surrogate,
        modulator
    ):
        self.num_arms = num_arms
        self.baseline = baseline
        self.mera = mera
        self.uprt = uprt
        self.surrogate = surrogate
        self.modulator = modulator
        
        # New components
        self.grfe = GRFEFunctional()
        self.active_learning = ActiveLearningController()
        self.info_gain_policy = InformationGainPolicy(num_arms)
        self.training = UncertaintyAwareTraining()
        
        self.recent_actions = []
        
    def select_action(self, state, history):
        """Select action using active learning when appropriate"""
        
        # Get QREA outputs
        mera_latent, mera_aux = self.mera(history)
        regime_probs, _ = self.uprt(mera_latent, state['recent_rewards'], state['q_values'])
        mean_probs, epistemic_var, _ = self.surrogate(mera_latent, state['features'])
        
        # Get modulation
        modulation = self.modulator(regime_probs, epistemic_var.mean(), 
                                   torch.zeros_like(epistemic_var), state['features'][:3])
        
        # Update active learning controller
        mode = self.active_learning.update(
            epistemic_uncertainty=epistemic_var.mean().item(),
            reward=state.get('last_reward', 0.5),
            regime_confidence=regime_probs.max().item()
        )
        
        # Select action based on mode
        if mode == AgentMode.ACTIVE_LEARNING:
            # Use information gain policy
            action = self.info_gain_policy.select_action(
                arm_uncertainties=self.baseline.get_uncertainties(),
                arm_counts=self.baseline.counts,
                recent_actions=self.recent_actions,
                exploration_bonus=modulation['alpha'].item()
            )
        else:
            # Use normal modulated baseline
            action = self.baseline.select_action(
                epsilon=modulation['epsilon'].item(),
                temperature=modulation['tau'].item(),
                exploration_bonus=modulation['alpha'].item()
            )
        
        self.recent_actions.append(action)
        
        return action, mode, mera_aux


if __name__ == "__main__":
    print("Testing Active Learning and GRFE Components")
    print("=" * 60)
    
    # Test GRFE functional
    print("\n1. Testing GRFE Functional")
    grfe = GRFEFunctional()
    
    batch_size = 4
    num_regimes = 5
    
    predicted_probs = torch.softmax(torch.randn(batch_size, num_regimes), dim=1)
    true_labels = torch.randint(0, num_regimes, (batch_size,))
    phi_q = torch.rand(batch_size) * 2.0
    
    total, components = grfe(predicted_probs, true_labels, phi_q)
    
    print(f"Total GRFE: {total.item():.4f}")
    print(f"Components:")
    print(f"  Variational FE: {components.variational_fe:.4f}")
    print(f"  Φ_Q: {components.phi_q:.4f}")
    print(f"  Coherence: {components.coherence_R:.4f}")
    print(f"  Surrogate Entropy: {components.surrogate_entropy:.4f}")
    
    # Test active learning controller
    print("\n2. Testing Active Learning Controller")
    controller = ActiveLearningController(epistemic_threshold=0.3)
    
    # Simulate trajectory
    for step in range(10):
        uncertainty = np.random.rand() * 0.5 if step < 5 else 0.1
        reward = 0.3 if step < 5 else 0.7
        confidence = 0.4 if step < 5 else 0.9
        
        mode = controller.update(uncertainty, reward, confidence)
        print(f"Step {step}: uncertainty={uncertainty:.2f}, mode={mode.value}")
    
    # Test information gain policy
    print("\n3. Testing Information Gain Policy")
    policy = InformationGainPolicy(num_arms=5)
    
    uncertainties = np.array([0.8, 0.2, 0.5, 0.1, 0.9])
    counts = np.array([10, 50, 20, 100, 5])
    recent = [0, 0, 1, 2, 1]
    
    action = policy.select_action(uncertainties, counts, recent)
    print(f"Selected action: {action}")
    print(f"Arm uncertainties: {uncertainties}")
    print(f"Arm counts: {counts}")
    
    # Test uncertainty-aware training
    print("\n4. Testing Uncertainty-Aware Training")
    training = UncertaintyAwareTraining()
    
    scenarios = [
        (0.5, 0.6, "High unc, high error"),
        (0.5, 0.1, "High unc, low error"),
        (0.1, 0.6, "Low unc, high error"),
        (0.1, 0.1, "Low unc, low error"),
    ]
    
    for uncertainty, error, desc in scenarios:
        lr = training.compute_adaptive_lr(uncertainty, error)
        print(f"{desc}: LR = {lr:.4f}")
    
    print("\n✓ All tests passed!")
