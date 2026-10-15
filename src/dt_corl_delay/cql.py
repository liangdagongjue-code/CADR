"""Conservative Q-Learning for continuous clean-data offline control."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .iql import GaussianPolicy, TwinQ


@dataclass(frozen=True)
class CQLHyperparameters:
    gamma: float = 0.99
    target_tau: float = 0.005
    cql_alpha: float = 5.0
    cql_temperature: float = 1.0
    cql_n_actions: int = 10
    bc_steps: int = 10_000
    initial_entropy_alpha: float = 1.0


class CQLAgent:
    """SAC-style CQL with automatic entropy tuning and deterministic evaluation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        learning_rate: float,
        hyperparameters: CQLHyperparameters,
        device: torch.device,
    ) -> None:
        if hyperparameters.cql_n_actions < 1:
            raise ValueError("cql_n_actions must be positive")
        if hyperparameters.cql_temperature <= 0.0:
            raise ValueError("cql_temperature must be positive")
        self.device = device
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        self.q = TwinQ(state_dim, action_dim, hidden_dims).to(device)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(device)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        initial_log_alpha = math.log(hyperparameters.initial_entropy_alpha)
        self.log_alpha = torch.tensor(
            initial_log_alpha,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -float(action_dim)
        self.total_updates = 0

    def _repeat_states(self, states: torch.Tensor, count: int) -> torch.Tensor:
        return states[:, None, :].expand(-1, count, -1)

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        states, actions, rewards, next_states, terminals = batch
        hp = self.hyperparameters
        self.total_updates += 1

        sampled_actions, sampled_log_prob = self.policy.sample(states)
        alpha = self.log_alpha.exp().detach()
        alpha_loss = -(self.log_alpha * (sampled_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            next_actions, next_log_prob = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_q(next_states, next_actions)
            next_q = torch.minimum(next_q1, next_q2) - self.log_alpha.exp() * next_log_prob
            q_target = rewards + hp.gamma * (1.0 - terminals) * next_q

        q1, q2 = self.q(states, actions)
        bellman_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        batch_size = states.shape[0]
        count = hp.cql_n_actions
        repeated_states = self._repeat_states(states, count)
        repeated_next_states = self._repeat_states(next_states, count)
        random_actions = torch.empty(
            batch_size,
            count,
            self.action_dim,
            device=self.device,
        ).uniform_(-1.0, 1.0)
        current_actions, current_log_prob = self.policy.sample(repeated_states)
        next_policy_actions, next_policy_log_prob = self.policy.sample(repeated_next_states)
        current_actions = current_actions.detach()
        current_log_prob = current_log_prob.detach()
        next_policy_actions = next_policy_actions.detach()
        next_policy_log_prob = next_policy_log_prob.detach()

        q1_random, q2_random = self.q(repeated_states, random_actions)
        q1_current, q2_current = self.q(repeated_states, current_actions)
        # Candidate actions drawn at next states are deliberately evaluated at
        # current states, matching the continuous CQL action-mixture objective.
        q1_next, q2_next = self.q(repeated_states, next_policy_actions)
        random_log_density = -self.action_dim * math.log(2.0)
        q1_candidates = torch.cat(
            (
                q1_random - random_log_density,
                q1_current - current_log_prob,
                q1_next - next_policy_log_prob,
            ),
            dim=1,
        )
        q2_candidates = torch.cat(
            (
                q2_random - random_log_density,
                q2_current - current_log_prob,
                q2_next - next_policy_log_prob,
            ),
            dim=1,
        )
        temperature = hp.cql_temperature
        cql1 = (
            torch.logsumexp(q1_candidates / temperature, dim=1).mul(temperature) - q1
        ).mean()
        cql2 = (
            torch.logsumexp(q2_candidates / temperature, dim=1).mul(temperature) - q2
        ).mean()
        cql_penalty = hp.cql_alpha * (cql1 + cql2)
        q_loss = bellman_loss + cql_penalty
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_prob = self.policy.sample(states)
        if self.total_updates <= hp.bc_steps:
            policy_loss = (self.log_alpha.exp().detach() * log_prob - self.policy.log_prob(states, actions)).mean()
        else:
            for parameter in self.q.parameters():
                parameter.requires_grad_(False)
            policy_q1, policy_q2 = self.q(states, new_actions)
            policy_loss = (
                self.log_alpha.exp().detach() * log_prob
                - torch.minimum(policy_q1, policy_q2)
            ).mean()
            for parameter in self.q.parameters():
                parameter.requires_grad_(True)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_q.parameters(), self.q.parameters()):
                target_parameter.lerp_(parameter, hp.target_tau)

        return {
            "q_loss": q_loss.detach(),
            "bellman_loss": bellman_loss.detach(),
            "cql_penalty": cql_penalty.detach(),
            "policy_loss": policy_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "entropy_alpha": self.log_alpha.exp().detach(),
            "q_mean": torch.minimum(q1, q2).mean().detach(),
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
        return self.policy.deterministic(normalized_observations)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "policy": self.policy.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_updates": self.total_updates,
            "hyperparameters": self.hyperparameters.__dict__,
        }

    def load_checkpoint(self, checkpoint: dict[str, Any], load_optimizers: bool = False) -> None:
        self.q.load_state_dict(checkpoint["q"])
        self.target_q.load_state_dict(checkpoint["target_q"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.total_updates = int(checkpoint.get("total_updates", 0))
        if load_optimizers:
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
