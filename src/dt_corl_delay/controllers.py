"""Deployment controllers that obey the latent-delay information boundary."""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import torch

from .channel import PolicyInput
from .delay_belief import CensoredAckDelayEstimator
from .dynamics import DeltaDynamicsEnsemble


def state_disagreement_components(
    age_states: torch.Tensor,
    age_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose posterior state spread into epistemic and age terms.

    ``age_states`` has shape ``[age, ensemble, state]`` and ``age_weights``
    is a normalized posterior over age. The returned root-mean-square terms
    obey the law of total variance (up to floating-point round-off): total
    variance is epistemic variance plus between-age variance.
    """

    if age_states.ndim != 3:
        raise ValueError("age_states must have shape [age, ensemble, state]")
    if age_weights.ndim != 1 or age_weights.shape[0] != age_states.shape[0]:
        raise ValueError("age_weights must match the age dimension")
    if not torch.isclose(age_weights.sum(), age_weights.new_tensor(1.0)):
        raise ValueError("age_weights must sum to one")
    age_means = age_states.mean(dim=1)
    posterior_mean = (age_means * age_weights[:, None]).sum(dim=0)
    epistemic_variance = (
        (age_states - age_means[:, None, :]).pow(2).mean(dim=-1).mean(dim=1)
        * age_weights
    ).sum()
    age_variance = (
        (age_means - posterior_mean).pow(2).mean(dim=-1) * age_weights
    ).sum()
    total_variance = (epistemic_variance + age_variance).clamp_min(0.0)
    return (
        torch.sqrt(total_variance),
        torch.sqrt(epistemic_variance.clamp_min(0.0)),
        torch.sqrt(age_variance.clamp_min(0.0)),
    )


def reconstruction_reliability(
    state_disagreement: float,
    risk_blend: float,
    risk_threshold: float,
) -> float:
    """Return the smooth dead-zone reliability used by robust blending."""

    excess_disagreement = max(0.0, state_disagreement - risk_threshold)
    return math.exp(-risk_blend * excess_disagreement)


class DirectObservationController:
    """Frozen IQL baseline that treats the received observation as current."""

    def __init__(
        self,
        agent: Any,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        self.agent = agent
        self.state_mean = state_mean
        self.state_std = state_std

    def reset(self, initial_observation: np.ndarray) -> None:
        del initial_observation

    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        del decision_step
        return self.agent.act(
            np.asarray(policy_input.observation),
            self.state_mean,
            self.state_std,
        )

    def observe_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        del action, reward, terminated, truncated

    def diagnostics(self) -> dict[str, float]:
        return {}


class CadrController:
    """Censored-ACK Delay Reconstruction (CADR) policy wrapper.

    The controller marginalizes the frozen actor over candidate current states
    reconstructed for every plausible observation age and dynamics member.
    Current packet delay, generation time, and channel internals are never
    passed to this class.
    """

    def __init__(
        self,
        agent: Any,
        dynamics: DeltaDynamicsEnsemble,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        max_delay: int,
        ack_lag: int,
        memory_steps: int = 128,
        decay_time: float = 48.0,
        prior_strength: float = 2.0,
        risk_blend: float = 0.0,
        risk_threshold: float = 0.0,
        age_decision_rule: str = "posterior_mean",
        entropy_gate_threshold: float = 1.4,
        use_content_conditioning: bool = True,
        posterior_updates: bool = True,
    ) -> None:
        if not math.isfinite(risk_blend) or risk_blend < 0.0:
            raise ValueError("risk_blend must be finite and non-negative")
        if not math.isfinite(risk_threshold) or risk_threshold < 0.0:
            raise ValueError("risk_threshold must be finite and non-negative")
        if age_decision_rule not in {"posterior_mean", "map", "entropy_gated"}:
            raise ValueError("unsupported age_decision_rule")
        if entropy_gate_threshold < 0.0:
            raise ValueError("entropy_gate_threshold must be non-negative")
        self.agent = agent
        self.dynamics = dynamics
        self.state_mean = state_mean
        self.state_std = state_std
        self.risk_blend = risk_blend
        self.risk_threshold = risk_threshold
        self.age_decision_rule = age_decision_rule
        self.entropy_gate_threshold = entropy_gate_threshold
        self.use_content_conditioning = bool(use_content_conditioning)
        self.posterior_updates = bool(posterior_updates)
        self.estimator = CensoredAckDelayEstimator(
            max_delay=max_delay,
            ack_lag=ack_lag,
            memory_steps=memory_steps,
            decay_time=decay_time,
            prior_strength=prior_strength,
        )
        self._actions: list[torch.Tensor] = []
        self._diagnostics: dict[str, list[float]] = {}
        self._known_selected_generated_step = 0
        self._previous_observation: np.ndarray | None = None

    def reset(self, initial_observation: np.ndarray) -> None:
        self.estimator.reset()
        self._actions.clear()
        self._known_selected_generated_step = 0
        self._previous_observation = np.asarray(initial_observation).copy()
        self._diagnostics = {
            "posterior_delay_mean": [],
            "effective_age_mean": [],
            "posterior_entropy": [],
            "state_disagreement": [],
            "epistemic_state_disagreement": [],
            "age_hypothesis_state_disagreement": [],
            "action_disagreement": [],
            "exact_age_from_unchanged_observation": [],
            "observation_changed": [],
            "selected_age": [],
            "conditioned_age_entropy": [],
            "map_rule_used": [],
            "reconstruction_reliability": [],
        }

    def _content_conditioned_age_probabilities(
        self,
        observation: np.ndarray,
        decision_step: int,
        base_probabilities: np.ndarray,
        policy_input: PolicyInput[np.ndarray],
    ) -> np.ndarray:
        for acknowledgement in policy_input.acknowledgements:
            self._known_selected_generated_step = max(
                self._known_selected_generated_step,
                int(acknowledgement.generated_step),
            )
        changed = (
            self._previous_observation is not None
            and not np.array_equal(observation, self._previous_observation)
        )
        exact_age = False
        if decision_step == 0:
            probabilities = np.asarray([1.0], dtype=np.float64)
            exact_age = True
        elif not changed:
            age = min(
                self.estimator.max_delay,
                max(0, decision_step - self._known_selected_generated_step),
            )
            probabilities = np.zeros(age + 1, dtype=np.float64)
            probabilities[age] = 1.0
            exact_age = True
        else:
            # A changed value proves that the selected packet is newer than
            # the latest packet whose generation step has already been
            # revealed. It does not reveal which newer packet arrived.
            maximum_possible_age = min(
                len(base_probabilities) - 1,
                max(0, decision_step - self._known_selected_generated_step - 1),
            )
            probabilities = base_probabilities[: maximum_possible_age + 1].copy()
            mass = float(probabilities.sum())
            if mass <= 1e-15:
                probabilities = np.full(
                    maximum_possible_age + 1,
                    1.0 / (maximum_possible_age + 1),
                    dtype=np.float64,
                )
            else:
                probabilities /= mass
        self._previous_observation = observation.copy()
        self._diagnostics["exact_age_from_unchanged_observation"].append(float(exact_age))
        self._diagnostics["observation_changed"].append(float(changed))
        return probabilities

    @torch.inference_mode()
    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        if self.posterior_updates:
            delay_probabilities = self.estimator.update(
                decision_step,
                policy_input.acknowledgements,
            )
        else:
            delay_probabilities = self.estimator.probabilities
        age_probabilities = self.estimator.effective_age_probabilities(decision_step)
        raw_observation = np.asarray(policy_input.observation)
        if self.use_content_conditioning:
            age_probabilities = self._content_conditioned_age_probabilities(
                raw_observation,
                decision_step,
                age_probabilities,
                policy_input,
            )
        selected_age = int(np.argmax(age_probabilities))
        positive_probabilities = age_probabilities[age_probabilities > 0.0]
        conditioned_age_entropy = float(
            -(positive_probabilities * np.log(positive_probabilities)).sum()
        )
        use_map = self.age_decision_rule == "map" or (
            self.age_decision_rule == "entropy_gated"
            and self.estimator.entropy >= self.entropy_gate_threshold
        )
        if use_map:
            age_probabilities = np.zeros_like(age_probabilities)
            age_probabilities[selected_age] = 1.0
        observation = torch.as_tensor(
            raw_observation,
            dtype=torch.float32,
            device=self.agent.device,
        ).unsqueeze(0)
        normalized_observation = (observation - self.state_mean) / self.state_std

        if len(age_probabilities) == 1 and age_probabilities[0] == 1.0:
            # Preserve bit-level equivalence with the direct policy in the
            # proven zero-delay case; continuous-control trajectories can be
            # chaotic under even round-off-sized action differences.
            action = self.agent.act_normalized(normalized_observation)
            state_disagreement = torch.zeros((), device=self.agent.device)
            epistemic_state_disagreement = torch.zeros(
                (), device=self.agent.device
            )
            age_hypothesis_state_disagreement = torch.zeros(
                (), device=self.agent.device
            )
            action_disagreement = torch.zeros((), device=self.agent.device)
        else:
            age_states = self.dynamics.rollout_age_candidates(
                normalized_observation,
                self._actions,
                maximum_age=len(age_probabilities) - 1,
            )
            states = age_states.reshape(-1, self.dynamics.state_dim)
            age_weights = torch.as_tensor(
                age_probabilities,
                dtype=torch.float32,
                device=self.agent.device,
            )
            weights = age_weights.repeat_interleave(self.dynamics.ensemble_size)
            weights = weights / self.dynamics.ensemble_size
            candidate_actions = self.agent.act_normalized(states)
            action = (candidate_actions * weights[:, None]).sum(dim=0, keepdim=True)

            (
                state_disagreement,
                epistemic_state_disagreement,
                age_hypothesis_state_disagreement,
            ) = state_disagreement_components(
                age_states,
                age_weights,
            )
            action_mean = (candidate_actions * weights[:, None]).sum(dim=0)
            action_disagreement = torch.sqrt(
                ((candidate_actions - action_mean).pow(2).mean(dim=-1) * weights).sum()
            )
        reliability = 1.0
        if self.risk_blend > 0.0:
            direct_action = self.agent.act_normalized(normalized_observation)
            reliability = reconstruction_reliability(
                float(state_disagreement),
                self.risk_blend,
                self.risk_threshold,
            )
            action = reliability * action + (1.0 - reliability) * direct_action
        action = action.clamp(-1.0, 1.0)
        self._actions.append(action.detach())

        support = np.arange(delay_probabilities.size, dtype=np.float64)
        ages = np.arange(age_probabilities.size, dtype=np.float64)
        self._diagnostics["posterior_delay_mean"].append(float(delay_probabilities @ support))
        self._diagnostics["effective_age_mean"].append(float(age_probabilities @ ages))
        self._diagnostics["posterior_entropy"].append(self.estimator.entropy)
        self._diagnostics["state_disagreement"].append(float(state_disagreement))
        self._diagnostics["epistemic_state_disagreement"].append(
            float(epistemic_state_disagreement)
        )
        self._diagnostics["age_hypothesis_state_disagreement"].append(
            float(age_hypothesis_state_disagreement)
        )
        self._diagnostics["action_disagreement"].append(float(action_disagreement))
        self._diagnostics["selected_age"].append(float(selected_age))
        self._diagnostics["conditioned_age_entropy"].append(conditioned_age_entropy)
        self._diagnostics["map_rule_used"].append(float(use_map))
        self._diagnostics["reconstruction_reliability"].append(float(reliability))
        return action.squeeze(0).cpu().numpy()

    def observe_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        del action, reward, terminated, truncated

    def diagnostics(self) -> dict[str, float]:
        return {
            f"mean_{name}": statistics.fmean(values) if values else 0.0
            for name, values in self._diagnostics.items()
        }
