"""Online delay inference from delayed ACKs and right-censored packets.

The estimator never receives the age of the observation used for the current
decision.  It only uses ACKs revealed by the channel and the fact that one
sensor packet is generated after every control decision.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .channel import Ack


class CensoredAckDelayEstimator:
    """Exponentially weighted discrete delay estimator with censoring.

    At decision ``t``, a missing ACK for the packet generated at ``g`` implies
    ``delay > t - g - ack_lag``.  An EM update uses that survival information
    rather than incorrectly treating recently missing ACKs as short delays.
    """

    def __init__(
        self,
        max_delay: int,
        ack_lag: int = 1,
        memory_steps: int = 128,
        decay_time: float = 48.0,
        prior_strength: float = 2.0,
        prior_probabilities: Sequence[float] | None = None,
        em_iterations: int = 8,
    ) -> None:
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        if ack_lag < 1:
            raise ValueError("ack_lag must be at least one")
        if memory_steps < max_delay + ack_lag + 1:
            raise ValueError("memory_steps must cover max_delay and ack_lag")
        if decay_time <= 0.0 or prior_strength <= 0.0:
            raise ValueError("decay_time and prior_strength must be positive")
        if em_iterations < 1:
            raise ValueError("em_iterations must be positive")

        self.max_delay = max_delay
        self.ack_lag = ack_lag
        self.memory_steps = memory_steps
        self.decay_time = decay_time
        self.prior_strength = prior_strength
        self.em_iterations = em_iterations
        if prior_probabilities is None:
            # A weak short-delay preference makes the cold start usable while
            # retaining non-zero support for every OOD delay in the model.
            support = np.arange(max_delay + 1, dtype=np.float64)
            prior = np.exp(-support / max(1.0, max_delay / 3.0))
        else:
            prior = np.asarray(prior_probabilities, dtype=np.float64)
            if prior.shape != (max_delay + 1,):
                raise ValueError("prior_probabilities has the wrong length")
        if not np.isfinite(prior).all() or np.any(prior <= 0.0):
            raise ValueError("prior probabilities must be finite and positive")
        self._prior = prior / prior.sum()
        self._probabilities = self._prior.copy()
        self._exact_delays: dict[int, int] = {}
        self._last_step = -1

    def reset(self) -> None:
        self._probabilities = self._prior.copy()
        self._exact_delays.clear()
        self._last_step = -1

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities.copy()

    @property
    def entropy(self) -> float:
        probabilities = np.clip(self._probabilities, 1e-12, 1.0)
        return float(-(probabilities * np.log(probabilities)).sum())

    def update(self, decision_step: int, acknowledgements: Sequence[Ack]) -> np.ndarray:
        if decision_step <= self._last_step:
            raise ValueError("decision steps must be strictly increasing")
        self._last_step = decision_step
        for acknowledgement in acknowledgements:
            if acknowledgement.revealed_step > decision_step:
                raise ValueError("an ACK cannot be used before its reveal step")
            delay = int(acknowledgement.delay)
            if delay < 0 or delay > self.max_delay:
                raise ValueError(
                    f"observed delay {delay} is outside configured support 0..{self.max_delay}"
                )
            self._exact_delays[int(acknowledgement.generated_step)] = delay

        first_generated = max(1, decision_step - self.memory_steps + 1)
        relevant_steps = range(first_generated, decision_step + 1)
        probabilities = self._probabilities.copy()
        support = np.arange(self.max_delay + 1)
        for _ in range(self.em_iterations):
            counts = self.prior_strength * self._prior
            for generated_step in relevant_steps:
                age = decision_step - generated_step
                weight = math.exp(-age / self.decay_time)
                exact_delay = self._exact_delays.get(generated_step)
                if exact_delay is not None:
                    counts[exact_delay] += weight
                    continue

                censoring_threshold = age - self.ack_lag
                if censoring_threshold < 0:
                    continue
                allowed = support > censoring_threshold
                mass = float(probabilities[allowed].sum())
                if mass <= 1e-15:
                    # This can only happen under severe model mismatch. Keep a
                    # small amount of support at the configured upper bound.
                    counts[-1] += weight
                else:
                    counts[allowed] += weight * probabilities[allowed] / mass
            probabilities = counts / counts.sum()
        self._probabilities = probabilities

        # Entries older than the active likelihood window cannot affect future
        # estimates and are discarded to keep deployment memory bounded.
        stale = [step for step in self._exact_delays if step < first_generated]
        for generated_step in stale:
            del self._exact_delays[generated_step]
        return self.probabilities

    def effective_age_probabilities(self, decision_step: int) -> np.ndarray:
        """Convert packet-delay probabilities into current observation ages.

        The freshest received packet has age ``a`` when its delay is no more
        than ``a`` and every newer packet has not arrived.  Probability beyond
        the episode's current age is folded into the oldest feasible state.
        """

        if decision_step < 0:
            raise ValueError("decision_step must be non-negative")
        maximum_age = min(self.max_delay, decision_step)
        cdf = np.cumsum(self._probabilities)
        full = np.zeros(self.max_delay + 1, dtype=np.float64)
        newer_missing = 1.0
        for age in range(self.max_delay + 1):
            full[age] = newer_missing * cdf[age]
            newer_missing *= 1.0 - cdf[age]
        # Numerical residue and ages impossible near reset belong to the
        # oldest feasible initial observation.
        result = full[: maximum_age + 1].copy()
        if maximum_age < self.max_delay:
            result[-1] += full[maximum_age + 1 :].sum()
        result[-1] += max(0.0, 1.0 - float(full.sum()))
        result /= result.sum()
        return result
