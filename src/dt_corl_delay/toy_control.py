"""Dependency-free control problem for end-to-end infrastructure checks."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from .channel import Ack


@dataclass
class LinearPlant:
    rng: random.Random
    transition_gain: float = 1.03
    action_gain: float = 0.18
    process_noise_std: float = 0.015
    action_limit: float = 2.0
    state: float = 0.0

    def reset(self) -> float:
        self.state = self.rng.uniform(-2.0, 2.0)
        return self.state

    def step(self, action: float) -> tuple[float, float]:
        clipped_action = max(-self.action_limit, min(self.action_limit, action))
        reward = -(self.state * self.state + 0.05 * clipped_action * clipped_action)
        noise = self.rng.gauss(0.0, self.process_noise_std)
        self.state = (
            self.transition_gain * self.state + self.action_gain * clipped_action + noise
        )
        return self.state, reward


@dataclass(frozen=True)
class LinearCheckpoint:
    gain: float
    training_samples: int
    seed: int


class FrozenLinearPolicy:
    """A fitted scalar controller whose gain is immutable at deployment."""

    def __init__(self, checkpoint: LinearCheckpoint) -> None:
        self.checkpoint = checkpoint
        self.ack_count = 0
        self.ack_delay_sum = 0

    def reset(self) -> None:
        self.ack_count = 0
        self.ack_delay_sum = 0

    def act(self, observation: float, acknowledgements: Sequence[Ack]) -> float:
        # ACK statistics are tracked to verify the information path. The smoke
        # policy intentionally does not adapt parameters or claim a new method.
        self.ack_count += len(acknowledgements)
        self.ack_delay_sum += sum(item.delay for item in acknowledgements)
        return -self.checkpoint.gain * observation


def fit_clean_offline_policy(
    seed: int,
    episodes: int,
    horizon: int,
) -> LinearCheckpoint:
    """Fit behavior cloning on clean, delay-free scalar trajectories."""

    rng = random.Random(seed)
    numerator = 0.0
    denominator = 0.0
    samples = 0
    for _ in range(episodes):
        plant = LinearPlant(rng=random.Random(rng.randrange(2**31)))
        state = plant.reset()
        for _ in range(horizon):
            expert_action = max(-2.0, min(2.0, -1.45 * state + rng.gauss(0.0, 0.08)))
            numerator += -state * expert_action
            denominator += state * state
            samples += 1
            state, _ = plant.step(expert_action)
    if denominator <= 0.0:
        raise RuntimeError("degenerate clean training data")
    return LinearCheckpoint(gain=numerator / denominator, training_samples=samples, seed=seed)

