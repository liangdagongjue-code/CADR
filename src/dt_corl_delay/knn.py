"""Non-parametric offline policy used to audit manipulation data usability."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class KNNPolicyAgent:
    """Retrieve actions from nearest normalized offline states on the GPU."""

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        neighbors: int = 1,
        temperature: float = 0.1,
    ) -> None:
        if states.ndim != 2 or actions.ndim != 2 or states.shape[0] != actions.shape[0]:
            raise ValueError("KNN states and actions must be aligned matrices")
        if not 1 <= neighbors <= states.shape[0]:
            raise ValueError("neighbors is outside the dataset range")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.device = states.device
        self.states = states.detach()
        self.actions = actions.detach()
        self.state_squared_norm = self.states.pow(2).sum(dim=1)
        self.neighbors = neighbors
        self.temperature = temperature
        # Match the minimal frozen-agent interface used by delayed evaluation.
        self.policy = self

    def eval(self) -> "KNNPolicyAgent":
        return self

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
        if normalized_observations.ndim != 2 or normalized_observations.shape[1] != self.states.shape[1]:
            raise ValueError("query states do not match the KNN state dimension")
        query_norm = normalized_observations.pow(2).sum(dim=1, keepdim=True)
        distances = (
            query_norm
            + self.state_squared_norm.unsqueeze(0)
            - 2.0 * normalized_observations @ self.states.T
        ).clamp_min_(0.0)
        values, indices = torch.topk(
            distances,
            k=self.neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        neighbor_actions = self.actions[indices]
        if self.neighbors == 1:
            return neighbor_actions[:, 0, :].clamp(-1.0, 1.0)
        weights = torch.softmax(-values / self.temperature, dim=1)
        return (neighbor_actions * weights.unsqueeze(-1)).sum(dim=1).clamp(-1.0, 1.0)

    def provenance(self) -> dict[str, Any]:
        return {
            "policy": "normalized_state_knn_action_retrieval",
            "neighbors": self.neighbors,
            "temperature": self.temperature,
            "reference_transitions": int(self.states.shape[0]),
            "state_dim": int(self.states.shape[1]),
            "action_dim": int(self.actions.shape[1]),
        }
