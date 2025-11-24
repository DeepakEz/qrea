"""
Emergent Communication and Language System
Agents develop shared communication protocols
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageEncoder(nn.Module):
    def __init__(self, state_dim: int, vocab_size: int, message_length: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, message_length * vocab_size),
        )

    def forward(self, state: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        logits = self.encoder(state).view(batch_size, self.message_length, self.vocab_size)
        logits = logits / temperature
        message_dist = F.gumbel_softmax(logits, tau=temperature, hard=True)
        message_indices = message_dist.argmax(dim=-1)
        return logits, message_indices


class MessageDecoder(nn.Module):
    def __init__(self, message_length: int, vocab_size: int, embedding_dim: int, action_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(messages).permute(1, 0, 2)
        attended, _ = self.attention(embedded, embedded, embedded)
        attended = attended.permute(1, 0, 2)
        pooled = attended.mean(dim=1)
        return self.decoder(pooled)


class CommunicationChannel:
    def __init__(self, config: dict):
        comm_cfg = config["communication"]
        self.range = comm_cfg["channel"]["range"]
        self.bandwidth = comm_cfg["channel"]["bandwidth"]
        self.noise_level = comm_cfg["channel"]["noise_level"]

        self.messages: Dict[int, List[Dict]] = {}
        self.message_count = defaultdict(int)

    def can_communicate(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        return np.linalg.norm(pos1 - pos2) <= self.range

    def send_message(self, sender_id: int, message: torch.Tensor, sender_pos: np.ndarray, receiver_positions: Dict[int, np.ndarray]):
        if sender_id not in self.messages:
            self.messages[sender_id] = []

        for receiver_id, receiver_pos in receiver_positions.items():
            if receiver_id == sender_id or not self.can_communicate(sender_pos, receiver_pos):
                continue
            if self.message_count[receiver_id] >= self.bandwidth:
                continue

            noisy_message = message.clone()
            if self.noise_level > 0:
                noise = torch.randn_like(noisy_message.float()) * self.noise_level
                noisy_message = (noisy_message.float() + noise).long().clamp(0, 99)

            if receiver_id not in self.messages:
                self.messages[receiver_id] = []
            self.messages[receiver_id].append({
                "sender": sender_id,
                "message": noisy_message,
                "distance": float(np.linalg.norm(sender_pos - receiver_pos)),
            })
            self.message_count[receiver_id] += 1

    def receive_messages(self, receiver_id: int) -> List[Dict]:
        return self.messages.get(receiver_id, [])

    def reset(self):
        self.messages = {}
        self.message_count = defaultdict(int)


class EmergentLanguage:
    def __init__(self, config: dict, state_dim: int, action_dim: int, device: torch.device):
        self.config = config
        self.device = device

        lang_cfg = config["communication"]["language"]
        self.vocab_size = lang_cfg["vocab_size"]
        self.message_length = lang_cfg["message_length"]
        self.embedding_dim = lang_cfg["embedding_dim"]

        self.encoder = MessageEncoder(state_dim, self.vocab_size, self.message_length, self.embedding_dim).to(device)
        self.decoder = MessageDecoder(self.message_length, self.vocab_size, self.embedding_dim, action_dim).to(device)
        self.channel = CommunicationChannel(config)

        self.message_history: List[torch.Tensor] = []
        self.success_rates: List[float] = []

        prag = config["communication"].get("pragmatics", {})
        self.success_threshold: float = float(prag.get("success_threshold", 0.05))
        self.update_rate: float = float(prag.get("update_rate", 0.1))

    def encode_message(self, state: torch.Tensor) -> torch.Tensor:
        _, message = self.encoder(state, temperature=1.0)
        return message

    def decode_message(self, message: torch.Tensor) -> torch.Tensor:
        return self.decoder(message)

    def communicate(self, robot_states: Dict[int, Dict], observations: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        self.channel.reset()
        positions = {rid: np.array(rstate["position"]) for rid, rstate in robot_states.items()}

        for robot_id, obs in observations.items():
            state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            message = self.encode_message(state).squeeze(0)
            self.channel.send_message(robot_id, message, positions[robot_id], positions)
            try:
                self.message_history.append(message.detach().to("cpu"))
                if len(self.message_history) > 5000:
                    self.message_history = self.message_history[-2500:]
            except Exception:
                pass

        communication_influences: Dict[int, torch.Tensor] = {}
        for robot_id in observations.keys():
            received = self.channel.receive_messages(robot_id)
            if received:
                messages = [msg["message"] for msg in received]
                messages_tensor = torch.stack(messages).to(self.device)
                influences = self.decoder(messages_tensor).mean(dim=0)
                communication_influences[robot_id] = influences
            else:
                communication_influences[robot_id] = torch.zeros(3, device=self.device)

        return communication_influences

    def train_language(self, episodes: List[Dict], optimizer: torch.optim.Optimizer) -> Dict:
        if not episodes:
            return {}

        total_loss = 0.0
        num_batches = 0

        for episode in episodes:
            obs = episode["observations"]
            actions = episode["actions"]

            if len(obs) < 10:
                continue

            start_idx = np.random.randint(0, len(obs) - 10)
            states = torch.as_tensor(np.array(obs[start_idx:start_idx+10]), dtype=torch.float32, device=self.device)
            target_actions = torch.as_tensor(np.array(actions[start_idx:start_idx+10]), dtype=torch.float32, device=self.device)

            message_logits, messages = self.encoder(states)
            influences = self.decoder(messages)

            comm_loss = F.mse_loss(influences, target_actions)
            message_probs = F.softmax(message_logits, dim=-1)
            entropy = -(message_probs * (message_probs + 1e-8).log()).sum()
            loss = comm_loss - 0.01 * entropy

            total_loss += float(loss.item())
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.success_rates.append(float(comm_loss.detach().item() < self.success_threshold))
            if len(self.success_rates) > 20000:
                self.success_rates = self.success_rates[-10000:]

        avg_loss = total_loss / max(num_batches, 1)
        recent_success = float(np.mean(self.success_rates[-100:])) if self.success_rates else 0.0
        return {
            "communication_loss": avg_loss,
            "message_diversity": self._compute_message_diversity(),
            "vocab_utilization": self._compute_vocab_utilization(),
            "avg_message_length_used": self._compute_avg_message_length(),
            "success_rate_recent": recent_success,
        }

    def _compute_message_diversity(self) -> float:
        if len(self.message_history) < 2:
            return 0.0
        recent = self.message_history[-100:]
        try:
            unique = len(set([tuple(msg.flatten().tolist()) for msg in recent]))
        except Exception:
            unique = len(recent)
        return unique / max(len(recent), 1)

    def get_language_statistics(self) -> Dict:
        return {
            "vocab_utilization": self._compute_vocab_utilization(),
            "message_diversity": self._compute_message_diversity(),
            "avg_message_length_used": self._compute_avg_message_length(),
            "success_rate_recent": float(np.mean(self.success_rates[-100:])) if self.success_rates else 0.0,
        }

    def _compute_vocab_utilization(self) -> float:
        if not self.message_history:
            return 0.0
        recent = self.message_history[-1000:]
        all_tokens = torch.cat([msg.flatten() for msg in recent]) if recent else torch.tensor([])
        if all_tokens.numel() == 0:
            return 0.0
        unique_tokens = torch.unique(all_tokens)
        return float(len(unique_tokens)) / float(self.vocab_size or 1)

    def _compute_avg_message_length(self) -> float:
        if not self.message_history:
            return 0.0
        recent = self.message_history[-100:]
        lengths = [(msg != 0).sum().item() for msg in recent]
        return float(np.mean(lengths)) if lengths else 0.0