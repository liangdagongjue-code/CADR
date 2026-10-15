"""Pinned official ManiSkill Diffusion Policy loading and delayed deployment."""

from __future__ import annotations

from collections import deque
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

from .channel import PolicyInput
from .controllers import (
    CadrController,
    reconstruction_reliability,
    state_disagreement_components,
)


EXPECTED_MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
_DP_RELATIVE_ROOT = Path("examples/baselines/diffusion_policy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_maniskill_source_provenance(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_MANISKILL_COMMIT:
        raise ValueError(f"unexpected ManiSkill source commit: {commit}")
    relative_files = [
        _DP_RELATIVE_ROOT / "train.py",
        _DP_RELATIVE_ROOT / "diffusion_policy/conditional_unet1d.py",
        _DP_RELATIVE_ROOT / "diffusion_policy/evaluate.py",
        Path("scripts/data_generation/replay_for_il_baselines.sh"),
    ]
    return {
        "repository": "https://github.com/haosulab/ManiSkill.git",
        "tag": "v3.0.1",
        "commit": commit,
        "file_sha256": {
            str(relative).replace("\\", "/"): _sha256(source_root / relative)
            for relative in relative_files
        },
    }


def _load_official_train_module(source_root: Path) -> Any:
    dp_root = source_root.resolve() / _DP_RELATIVE_ROOT
    train_path = dp_root / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    dp_root_text = str(dp_root)
    if dp_root_text not in sys.path:
        sys.path.insert(0, dp_root_text)
    module_name = "_dt_corl_delay_official_maniskill_dp_train"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import official DP source from {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _dataset_dimensions(dataset_path: Path) -> tuple[int, int]:
    with h5py.File(dataset_path, "r") as handle:
        first_name = min(handle.keys(), key=lambda name: int(name.rsplit("_", 1)[1]))
        trajectory = handle[first_name]
        observation_shape = trajectory["obs"].shape
        action_shape = trajectory["actions"].shape
    if len(observation_shape) != 2 or len(action_shape) != 2:
        raise ValueError("official state DP requires flat observations and actions")
    return int(observation_shape[-1]), int(action_shape[-1])


def load_official_diffusion_agent(
    checkpoint_path: Path,
    dataset_path: Path,
    source_root: Path,
    device: torch.device,
    obs_horizon: int = 2,
    act_horizon: int = 8,
    pred_horizon: int = 16,
) -> tuple[Any, dict[str, Any]]:
    provenance = official_maniskill_source_provenance(source_root)
    dataset_path = dataset_path.resolve()
    observation_dim, action_dim = _dataset_dimensions(dataset_path)
    module = _load_official_train_module(source_root)

    class _ShapeOnlyEnvironment:
        single_observation_space = SimpleNamespace(
            shape=(obs_horizon, observation_dim)
        )
        single_action_space = SimpleNamespace(
            shape=(action_dim,),
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
        )

    model_args = SimpleNamespace(
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        pred_horizon=pred_horizon,
        diffusion_step_embed_dim=64,
        unet_dims=[64, 128, 256],
        n_groups=8,
    )
    agent = module.Agent(_ShapeOnlyEnvironment(), model_args).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema_agent" not in payload:
        raise ValueError("official ManiSkill DP checkpoint has no ema_agent state")
    agent.load_state_dict(payload["ema_agent"], strict=True)
    agent.eval()
    return agent, {
        "source": provenance,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path.resolve()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "obs_horizon": obs_horizon,
        "act_horizon": act_horizon,
        "pred_horizon": pred_horizon,
    }


class OfficialDiffusionPolicyController:
    """Execute official DP chunks using only the received observation stream."""

    def __init__(
        self,
        agent: Any,
        device: torch.device,
        obs_horizon: int = 2,
        act_horizon: int = 8,
    ) -> None:
        self.agent = agent
        self.device = device
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self._episode_seed = 0
        self._observations: deque[np.ndarray] = deque(maxlen=obs_horizon)
        self._action_queue: deque[np.ndarray] = deque()
        self._last_decision_step = -1
        self._replans = 0
        self._actions = 0

    def set_episode_seed(self, seed: int) -> None:
        self._episode_seed = int(seed)

    def reset(self, initial_observation: np.ndarray) -> None:
        observation = np.asarray(initial_observation, dtype=np.float32).copy()
        self._observations.clear()
        self._observations.extend(observation.copy() for _ in range(self.obs_horizon))
        self._action_queue.clear()
        self._last_decision_step = -1
        self._replans = 0
        self._actions = 0

    @torch.inference_mode()
    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        if decision_step <= self._last_decision_step:
            raise ValueError("decision steps must be strictly increasing")
        if decision_step > 0:
            self._observations.append(
                np.asarray(policy_input.observation, dtype=np.float32).copy()
            )
        self._last_decision_step = decision_step
        if not self._action_queue:
            # A per-replan seed gives paired diffusion noise across delay
            # scenarios while retaining distinct samples over an episode.
            sample_seed = self._episode_seed + 1_000_003 * self._replans
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            observation_sequence = torch.as_tensor(
                np.stack(tuple(self._observations)),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_sequence = self.agent.get_action(observation_sequence)[0]
            action_sequence = action_sequence[: self.act_horizon].detach().cpu().numpy()
            self._action_queue.extend(
                np.asarray(action, dtype=np.float32).copy() for action in action_sequence
            )
            self._replans += 1
        self._actions += 1
        return self._action_queue.popleft()

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
            "diffusion_replans": float(self._replans),
            "mean_executed_chunk_length": (
                float(self._actions) / self._replans if self._replans else 0.0
            ),
        }


class CadrDiffusionPolicyController(CadrController):
    """CADR state reconstruction followed by the frozen official DP actor.

    The posterior-mean state is updated every control step, including while an
    eight-action DP chunk is being executed. The DP actor is called only at the
    same chunk boundaries as the direct official baseline.
    """

    def __init__(
        self,
        agent: Any,
        dynamics: Any,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        device: torch.device,
        max_delay: int,
        ack_lag: int,
        obs_horizon: int = 2,
        act_horizon: int = 8,
        memory_steps: int = 64,
        decay_time: float = 24.0,
        prior_strength: float = 2.0,
        risk_blend: float = 0.0,
        risk_threshold: float = 0.0,
        age_decision_rule: str = "posterior_mean",
        entropy_gate_threshold: float = 1.4,
        age_source: str = "posterior",
        use_content_conditioning: bool = True,
        posterior_updates: bool = True,
    ) -> None:
        if age_source not in {"posterior", "oracle"}:
            raise ValueError("age_source must be 'posterior' or 'oracle'")
        super().__init__(
            agent=agent,
            dynamics=dynamics,
            state_mean=state_mean,
            state_std=state_std,
            max_delay=max_delay,
            ack_lag=ack_lag,
            memory_steps=memory_steps,
            decay_time=decay_time,
            prior_strength=prior_strength,
            risk_blend=risk_blend,
            risk_threshold=risk_threshold,
            age_decision_rule=age_decision_rule,
            entropy_gate_threshold=entropy_gate_threshold,
            use_content_conditioning=use_content_conditioning,
            posterior_updates=posterior_updates,
        )
        self.device = device
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.age_source = age_source
        self._episode_seed = 0
        self._oracle_age: int | None = None
        self._state_history: deque[np.ndarray] = deque(maxlen=obs_horizon)
        self._action_queue: deque[np.ndarray] = deque()
        self._last_decision_step = -1
        self._replans = 0
        self._executed_actions = 0
        self._oracle_age = None

    def set_oracle_age(self, age: int) -> None:
        """Supply evaluator-only packet age for the privileged upper bound."""

        if self.age_source != "oracle":
            raise RuntimeError("oracle age can only be set for an oracle controller")
        if age < 0 or age > self.estimator.max_delay:
            raise ValueError("oracle age is outside the configured delay support")
        self._oracle_age = int(age)

    def set_episode_seed(self, seed: int) -> None:
        self._episode_seed = int(seed)

    def reset(self, initial_observation: np.ndarray) -> None:
        super().reset(initial_observation)
        initial = np.asarray(initial_observation, dtype=np.float32).copy()
        self._state_history.clear()
        self._state_history.extend(initial.copy() for _ in range(self.obs_horizon))
        self._action_queue.clear()
        self._last_decision_step = -1
        self._replans = 0
        self._executed_actions = 0

    @torch.inference_mode()
    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        if decision_step <= self._last_decision_step:
            raise ValueError("decision steps must be strictly increasing")
        self._last_decision_step = decision_step
        if self.posterior_updates:
            delay_probabilities = self.estimator.update(
                decision_step,
                policy_input.acknowledgements,
            )
        else:
            delay_probabilities = self.estimator.probabilities
        raw_observation = np.asarray(policy_input.observation, dtype=np.float32)
        if self.age_source == "oracle":
            if self._oracle_age is None:
                raise RuntimeError("set_oracle_age must be called before every action")
            selected_age = self._oracle_age
            self._oracle_age = None
            age_probabilities = np.zeros(selected_age + 1, dtype=np.float64)
            age_probabilities[selected_age] = 1.0
            conditioned_age_entropy = 0.0
            use_map = True
        else:
            age_probabilities = self.estimator.effective_age_probabilities(decision_step)
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
            device=self.device,
        ).unsqueeze(0)
        normalized_observation = (observation - self.state_mean) / self.state_std
        exact_current_observation = (
            len(age_probabilities) == 1 and age_probabilities[0] == 1.0
        )
        if exact_current_observation:
            reconstructed_raw = raw_observation.copy()
            state_disagreement = torch.zeros((), device=self.device)
            epistemic_state_disagreement = torch.zeros((), device=self.device)
            age_hypothesis_state_disagreement = torch.zeros((), device=self.device)
            reliability = 1.0
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
                device=self.device,
            )
            weights = age_weights.repeat_interleave(self.dynamics.ensemble_size)
            weights = weights / self.dynamics.ensemble_size
            reconstructed_normalized = (states * weights[:, None]).sum(dim=0)
            (
                state_disagreement,
                epistemic_state_disagreement,
                age_hypothesis_state_disagreement,
            ) = state_disagreement_components(
                age_states,
                age_weights,
            )
            reliability = 1.0
            if self.age_source == "posterior" and self.risk_blend > 0.0:
                reliability = reconstruction_reliability(
                    float(state_disagreement),
                    self.risk_blend,
                    self.risk_threshold,
                )
                reconstructed_normalized = (
                    reliability * reconstructed_normalized
                    + (1.0 - reliability) * normalized_observation.squeeze(0)
                )
            reconstructed_raw = (
                reconstructed_normalized.unsqueeze(0) * self.state_std
                + self.state_mean
            ).squeeze(0).cpu().numpy()
        if decision_step > 0:
            self._state_history.append(
                np.asarray(reconstructed_raw, dtype=np.float32).copy()
            )

        if not self._action_queue:
            sample_seed = self._episode_seed + 1_000_003 * self._replans
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            observation_sequence = torch.as_tensor(
                np.stack(tuple(self._state_history)),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_sequence = self.agent.get_action(observation_sequence)[0]
            action_sequence = action_sequence[: self.act_horizon].detach().cpu().numpy()
            self._action_queue.extend(
                np.asarray(action, dtype=np.float32).copy() for action in action_sequence
            )
            self._replans += 1
        action = np.clip(self._action_queue.popleft(), -1.0, 1.0).astype(np.float32)
        self._actions.append(
            torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        )
        self._executed_actions += 1

        support = np.arange(delay_probabilities.size, dtype=np.float64)
        ages = np.arange(age_probabilities.size, dtype=np.float64)
        self._diagnostics["posterior_delay_mean"].append(
            float(delay_probabilities @ support)
        )
        self._diagnostics["effective_age_mean"].append(
            float(age_probabilities @ ages)
        )
        self._diagnostics["posterior_entropy"].append(self.estimator.entropy)
        self._diagnostics["state_disagreement"].append(float(state_disagreement))
        self._diagnostics["epistemic_state_disagreement"].append(
            float(epistemic_state_disagreement)
        )
        self._diagnostics["age_hypothesis_state_disagreement"].append(
            float(age_hypothesis_state_disagreement)
        )
        self._diagnostics["action_disagreement"].append(0.0)
        self._diagnostics["selected_age"].append(float(selected_age))
        self._diagnostics["conditioned_age_entropy"].append(
            conditioned_age_entropy
        )
        self._diagnostics["map_rule_used"].append(float(use_map))
        self._diagnostics["reconstruction_reliability"].append(float(reliability))
        return action

    def diagnostics(self) -> dict[str, float]:
        diagnostics = super().diagnostics()
        diagnostics.update(
            {
                "diffusion_replans": float(self._replans),
                "mean_executed_chunk_length": (
                    float(self._executed_actions) / self._replans
                    if self._replans
                    else 0.0
                ),
            }
        )
        return diagnostics
