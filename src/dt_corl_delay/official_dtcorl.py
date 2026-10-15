"""Maintained evaluation adapter for the official DT-CORL reconstructor.

The neural architecture is imported directly from the pinned upstream source;
this module only adapts its input/output boundary to the maintained Minari and
Gymnasium experiment stack used by this project.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import statistics
from types import ModuleType
from typing import Any

import numpy as np
import torch

from .channel import PolicyInput
from .delay_belief import CensoredAckDelayEstimator


OFFICIAL_DT_CORL_COMMIT = "7e44adf934b8aef91ca6fb9e057a1935c6a167c9"
OFFICIAL_DT_CORL_NETWORK_SHA256 = (
    "f1ead7f83ef231ca0b2786c24a47b3cbef1179393f6b754e84cb98a08992f81f"
)


def official_network_path() -> Path:
    project_root = Path(os.environ.get("DT_CORL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    path = project_root / "external" / "DT-CORL" / "utils" / "network.py"
    if not path.is_file():
        raise FileNotFoundError(f"pinned DT-CORL network source not found: {path}")
    return path.resolve()


def official_source_provenance() -> dict[str, str]:
    path = official_network_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_DT_CORL_NETWORK_SHA256:
        raise ValueError(
            "pinned DT-CORL network source checksum mismatch: "
            f"{digest} != {OFFICIAL_DT_CORL_NETWORK_SHA256}"
        )
    return {
        "repository": "https://github.com/simonjzh/DT-CORL",
        "commit": OFFICIAL_DT_CORL_COMMIT,
        "network_source": str(path),
        "network_source_sha256": digest,
    }


_NETWORK_MODULE: ModuleType | None = None


def load_official_network_module() -> ModuleType:
    global _NETWORK_MODULE
    if _NETWORK_MODULE is None:
        path = official_network_path()
        specification = importlib.util.spec_from_file_location(
            "dt_corl_pinned_official_network",
            path,
        )
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load official DT-CORL network source: {path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        _NETWORK_MODULE = module
    return _NETWORK_MODULE


def build_official_reconstructor(
    state_dim: int,
    action_dim: int,
    maximum_delay: int,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    attention_dropout: float,
    residual_dropout: float,
    hidden_dropout: float,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if maximum_delay <= 0:
        raise ValueError("maximum_delay must be positive")
    official = load_official_network_module()
    autoencoder = official.AutoEncoder(
        input_dim=state_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    dynamics = official.TRANS_Dynamic(
        latent_dim=latent_dim,
        condition_dim=action_dim,
        seq_len=maximum_delay,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        attention_dropout=attention_dropout,
        residual_dropout=residual_dropout,
        hidden_dropout=hidden_dropout,
    ).to(device)
    return autoencoder, dynamics


class OfficialDTCorlAckMapController:
    """Run the official transformer reconstructor with an ACK-derived MAP age.

    Upstream DT-CORL evaluation supplies the sampled delay directly. This
    adapter removes that privileged signal: age is inferred only from delayed
    observations and acknowledgements, matching this benchmark's information
    boundary. Immediate actions and rewards are retained because they are
    explicit inputs of the upstream reconstruction model.
    """

    def __init__(
        self,
        agent: Any,
        autoencoder: torch.nn.Module,
        dynamics: torch.nn.Module,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        maximum_delay: int,
        ack_lag: int,
        memory_steps: int = 128,
        decay_time: float = 48.0,
        prior_strength: float = 2.0,
        age_source: str = "ack_map",
    ) -> None:
        if age_source not in {"ack_map", "oracle"}:
            raise ValueError("age_source must be 'ack_map' or 'oracle'")
        self.agent = agent
        self.autoencoder = autoencoder
        self.dynamics = dynamics
        self.state_mean = state_mean
        self.state_std = state_std
        self.maximum_delay = maximum_delay
        self.age_source = age_source
        self.estimator = CensoredAckDelayEstimator(
            max_delay=maximum_delay,
            ack_lag=ack_lag,
            memory_steps=memory_steps,
            decay_time=decay_time,
            prior_strength=prior_strength,
        )
        self._actions: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._diagnostics: dict[str, list[float]] = {}
        self._oracle_age: int | None = None

    def reset(self, initial_observation: np.ndarray) -> None:
        del initial_observation
        self.estimator.reset()
        self._actions.clear()
        self._rewards.clear()
        self._oracle_age = None
        self._diagnostics = {
            "estimated_age": [],
            "posterior_delay_mean": [],
            "posterior_entropy": [],
        }

    @torch.inference_mode()
    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        delay_probabilities = self.estimator.update(
            decision_step,
            policy_input.acknowledgements,
        )
        age_probabilities = self.estimator.effective_age_probabilities(decision_step)
        if self.age_source == "oracle":
            if self._oracle_age is None:
                raise RuntimeError("oracle age was not provided before the decision")
            selected_age = self._oracle_age
        else:
            selected_age = int(np.argmax(age_probabilities))
        age = min(selected_age, self.maximum_delay, len(self._actions), len(self._rewards))
        observation = torch.as_tensor(
            np.asarray(policy_input.observation),
            dtype=torch.float32,
            device=self.agent.device,
        ).unsqueeze(0)
        normalized_observation = (observation - self.state_mean) / self.state_std
        if age == 0:
            reconstructed = normalized_observation
        else:
            actions = torch.stack(self._actions[-age:], dim=0).unsqueeze(0)
            rewards = torch.as_tensor(
                self._rewards[-age:],
                dtype=torch.float32,
                device=self.agent.device,
            ).view(1, age, 1)
            action_padding = torch.zeros(
                (1, self.maximum_delay - age, actions.shape[-1]),
                dtype=actions.dtype,
                device=self.agent.device,
            )
            reward_padding = torch.zeros(
                (1, self.maximum_delay - age, 1),
                dtype=rewards.dtype,
                device=self.agent.device,
            )
            padding_mask = torch.cat(
                (
                    torch.zeros(age, dtype=torch.bool, device=self.agent.device),
                    torch.ones(
                        self.maximum_delay - age,
                        dtype=torch.bool,
                        device=self.agent.device,
                    ),
                )
            ).unsqueeze(0)
            timesteps = torch.arange(
                self.maximum_delay,
                dtype=torch.long,
                device=self.agent.device,
            )
            start_latent = self.autoencoder.encode(normalized_observation.unsqueeze(1))
            predicted_latents = self.dynamics(
                latents=start_latent,
                actions=torch.cat((actions, action_padding), dim=1),
                rewards=torch.cat((rewards, reward_padding), dim=1),
                timesteps=timesteps,
                masks=padding_mask,
            )
            reconstructed = self.autoencoder.decode(predicted_latents[:, age - 1, :])
        action = self.agent.act_normalized(reconstructed).clamp(-1.0, 1.0)

        support = np.arange(delay_probabilities.size, dtype=np.float64)
        self._diagnostics["estimated_age"].append(float(age))
        self._diagnostics["posterior_delay_mean"].append(
            float(delay_probabilities @ support)
        )
        self._diagnostics["posterior_entropy"].append(self.estimator.entropy)
        return action.squeeze(0).cpu().numpy()

    def set_oracle_age(self, age: int) -> None:
        if self.age_source != "oracle":
            raise RuntimeError("oracle age can only be set for an oracle controller")
        if age < 0:
            raise ValueError("oracle age must be non-negative")
        self._oracle_age = int(age)

    def observe_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        del terminated, truncated
        self._actions.append(
            torch.as_tensor(
                action,
                dtype=torch.float32,
                device=self.agent.device,
            ).detach()
        )
        self._rewards.append(float(reward))

    def diagnostics(self) -> dict[str, float]:
        return {
            f"mean_{name}": statistics.fmean(values) if values else 0.0
            for name, values in self._diagnostics.items()
        }


def official_dynamic_parameter_count(
    autoencoder: torch.nn.Module,
    dynamics: torch.nn.Module,
) -> int:
    return sum(parameter.numel() for parameter in autoencoder.parameters()) + sum(
        parameter.numel() for parameter in dynamics.parameters()
    )


def reconstruction_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(float((predictions - targets).pow(2).mean()))
