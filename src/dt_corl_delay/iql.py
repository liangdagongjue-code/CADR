"""A compact Implicit Q-Learning implementation for continuous control."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = _mlp(input_dim, 1, hidden_dims)
        self.q2 = _mlp(input_dim, 1, hidden_dims)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((states, actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.network = _mlp(state_dim, 1, hidden_dims)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.backbone = _mlp(state_dim, hidden_dims[-1], hidden_dims[:-1])
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

    def distribution_parameters(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.backbone(states))
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-5.0, 2.0)
        return mean, log_std

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.distribution_parameters(states)
        clipped_actions = actions.clamp(-0.999999, 0.999999)
        pre_tanh = torch.atanh(clipped_actions)
        inverse_std = torch.exp(-log_std)
        gaussian_log_prob = -0.5 * (
            ((pre_tanh - mean) * inverse_std).pow(2)
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        correction = torch.log(1.0 - clipped_actions.pow(2) + 1e-6)
        return (gaussian_log_prob - correction).sum(dim=-1, keepdim=True)

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a reparameterized squashed-Gaussian action and its log density."""

        mean, log_std = self.distribution_parameters(states)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.rsample()
        actions = torch.tanh(pre_tanh)
        correction = torch.log(1.0 - actions.pow(2) + 1e-6)
        log_prob = (normal.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)
        return actions, log_prob

    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(states)
        return torch.tanh(mean)


@dataclass(frozen=True)
class IQLHyperparameters:
    gamma: float = 0.99
    target_tau: float = 0.005
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    max_advantage_weight: float = 100.0


class IQLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        learning_rate: float,
        hyperparameters: IQLHyperparameters,
        device: torch.device,
    ) -> None:
        self.device = device
        self.hyperparameters = hyperparameters
        self.q = TwinQ(state_dim, action_dim, hidden_dims).to(device)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.value = ValueFunction(state_dim, hidden_dims).to(device)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(device)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        states, actions, rewards, next_states, terminals = batch
        hp = self.hyperparameters

        with torch.no_grad():
            target_q1, target_q2 = self.target_q(states, actions)
            target_q = torch.minimum(target_q1, target_q2)
        values = self.value(states)
        advantages = target_q - values
        expectile_weight = torch.where(
            advantages > 0,
            hp.expectile,
            1.0 - hp.expectile,
        )
        value_loss = (expectile_weight * advantages.pow(2)).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            next_values = self.value(next_states)
            q_target = rewards + hp.gamma * (1.0 - terminals) * next_values
        q1, q2 = self.q(states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            advantage_weights = torch.exp(hp.advantage_temperature * advantages)
            advantage_weights.clamp_(max=hp.max_advantage_weight)
        policy_log_prob = self.policy.log_prob(states, actions)
        policy_loss = -(advantage_weights * policy_log_prob).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_q.parameters(), self.q.parameters()):
                target_parameter.lerp_(parameter, hp.target_tau)

        return {
            "q_loss": q_loss.detach(),
            "value_loss": value_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "q_mean": target_q.mean().detach(),
            "value_mean": values.mean().detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_weight_mean": advantage_weights.mean().detach(),
        }

    @torch.inference_mode()
    def act(self, observation: np.ndarray, state_mean: torch.Tensor, state_std: torch.Tensor) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        normalized = (observation_tensor - state_mean) / state_std
        return self.act_normalized(normalized).squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def act_normalized(self, normalized_observations: torch.Tensor) -> torch.Tensor:
        """Act on already normalized states without a CPU synchronization."""

        return self.policy.deterministic(normalized_observations)

    def load_checkpoint(self, checkpoint: dict[str, Any], load_optimizers: bool = False) -> None:
        self.q.load_state_dict(checkpoint["q"])
        self.target_q.load_state_dict(checkpoint["target_q"])
        self.value.load_state_dict(checkpoint["value"])
        self.policy.load_state_dict(checkpoint["policy"])
        if load_optimizers:
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])

    def checkpoint(self) -> dict[str, Any]:
        return {
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "value": self.value.state_dict(),
            "policy": self.policy.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "hyperparameters": self.hyperparameters.__dict__,
        }
