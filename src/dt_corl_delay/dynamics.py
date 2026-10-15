"""Clean-data probabilistic dynamics used for latent-state reconstruction."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _network(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), nn.SiLU()))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class DeltaDynamicsEnsemble(nn.Module):
    """Bootstrap ensemble predicting normalized one-step state deltas."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        ensemble_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least one")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.device = device
        self.members = nn.ModuleList(
            [_network(state_dim + action_dim, state_dim, hidden_dims) for _ in range(ensemble_size)]
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        target_deltas = next_states - states
        member_losses: list[torch.Tensor] = []
        batch_size = states.shape[0]
        for member in self.members:
            bootstrap = torch.randint(
                batch_size,
                (batch_size,),
                device=states.device,
                generator=generator,
            )
            inputs = torch.cat((states[bootstrap], actions[bootstrap]), dim=-1)
            prediction = member(inputs)
            member_losses.append(F.mse_loss(prediction, target_deltas[bootstrap]))
        loss = torch.stack(member_losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()
        return {
            "dynamics_loss": loss.detach(),
            "member_loss_min": torch.stack(member_losses).min().detach(),
            "member_loss_max": torch.stack(member_losses).max().detach(),
        }

    @torch.inference_mode()
    def predict_next_members(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return shape ``[ensemble, batch, state]`` one-step predictions."""

        inputs = torch.cat((states, actions), dim=-1)
        return torch.stack(
            [states + member(inputs) for member in self.members],
            dim=0,
        )

    @torch.inference_mode()
    def rollout_members(
        self,
        initial_state: torch.Tensor,
        actions: Sequence[torch.Tensor],
        normalized_state_clip: float = 20.0,
    ) -> torch.Tensor:
        """Roll each ensemble member through the same candidate action suffix."""

        if initial_state.shape != (1, self.state_dim):
            raise ValueError(f"expected initial state shape (1, {self.state_dim})")
        states = initial_state.expand(self.ensemble_size, -1).clone()
        for action in actions:
            if action.shape != (1, self.action_dim):
                raise ValueError(f"expected action shape (1, {self.action_dim})")
            next_states: list[torch.Tensor] = []
            for index, member in enumerate(self.members):
                inputs = torch.cat((states[index : index + 1], action), dim=-1)
                next_states.append(states[index : index + 1] + member(inputs))
            states = torch.cat(next_states, dim=0).clamp(
                -normalized_state_clip,
                normalized_state_clip,
            )
        return states

    @torch.inference_mode()
    def rollout_age_candidates(
        self,
        initial_state: torch.Tensor,
        action_history: Sequence[torch.Tensor],
        maximum_age: int,
        normalized_state_clip: float = 20.0,
    ) -> torch.Tensor:
        """Batch all observation-age hypotheses.

        Returns ``[age, ensemble, state]`` and is numerically equivalent to
        calling :meth:`rollout_members` once per action suffix, while reducing
        model calls from quadratic to linear in ``maximum_age``.
        """

        if initial_state.shape != (1, self.state_dim):
            raise ValueError(f"expected initial state shape (1, {self.state_dim})")
        if maximum_age < 0 or maximum_age > len(action_history):
            raise ValueError("maximum_age must be covered by action_history")
        age_count = maximum_age + 1
        states = initial_state.reshape(1, 1, self.state_dim).expand(
            age_count, self.ensemble_size, self.state_dim
        ).clone()
        history_length = len(action_history)
        for phase in range(maximum_age):
            active_age_values = list(range(phase + 1, age_count))
            active_ages = torch.arange(
                phase + 1,
                age_count,
                device=self.device,
                dtype=torch.long,
            )
            phase_actions = torch.cat(
                [
                    action_history[history_length - age + phase]
                    for age in active_age_values
                ],
                dim=0,
            )
            updated = states.clone()
            for member_index, member in enumerate(self.members):
                member_states = states[active_ages, member_index]
                inputs = torch.cat((member_states, phase_actions), dim=-1)
                predicted = (member_states + member(inputs)).clamp(
                    -normalized_state_clip,
                    normalized_state_clip,
                )
                updated[active_ages, member_index] = predicted
            states = updated
        return states

    def checkpoint(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "ensemble_size": self.ensemble_size,
            "members": self.members.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any], load_optimizer: bool = False) -> None:
        self.members.load_state_dict(checkpoint["members"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
