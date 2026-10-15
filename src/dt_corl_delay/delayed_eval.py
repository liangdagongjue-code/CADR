"""Frozen-policy evaluation under latent, non-stationary observation delays."""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch

from .channel import DelayedObservationChannel, PolicyInput
from .controllers import DirectObservationController
from .delay_process import build_delay_process


def _cvar(values: list[float], alpha: float) -> float:
    count = max(1, math.ceil(len(values) * alpha))
    return statistics.fmean(sorted(values)[:count])


def _changepoint_recovery_steps(
    rewards: list[float],
    change_step: int,
    reference_window: int = 50,
    recovery_window: int = 20,
) -> int:
    """Steps until post-change rolling reward returns near its pre-change level."""

    if change_step <= 0 or change_step >= len(rewards):
        return 0
    reference_start = max(0, change_step - reference_window)
    reference = statistics.fmean(rewards[reference_start:change_step])
    target = reference - 0.1 * max(1.0, abs(reference))
    final_start = len(rewards) - recovery_window
    for start in range(change_step, final_start + 1):
        if statistics.fmean(rewards[start : start + recovery_window]) >= target:
            return start - change_step
    return len(rewards) - change_step


def _episode_success_summary(
    samples: list[tuple[int, bool]],
    scenario: Mapping[str, Any],
) -> dict[str, float] | None:
    if not samples:
        return None
    values = [float(value) for _, value in samples]
    result = {
        "once": float(any(value for _, value in samples)),
        "at_end": values[-1],
        "occupancy": statistics.fmean(values),
    }
    if scenario.get("type") == "changepoint":
        change_step = int(scenario["change_step"])
        post_change = [
            float(value) for step, value in samples if step >= change_step
        ]
        if post_change:
            result["post_change_occupancy"] = statistics.fmean(post_change)
    return result


class DeploymentController(Protocol):
    def reset(self, initial_observation: np.ndarray) -> None: ...

    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray: ...

    def observe_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def diagnostics(self) -> dict[str, float]: ...


def evaluate_frozen_policy(
    agent: Any,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    dataset_id: str,
    scenario: Mapping[str, Any],
    episodes: int,
    seed: int,
    ack_lag: int,
    cvar_alpha: float,
    method: str = "direct_observation_iql",
    environment_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    agent.policy.eval()
    return evaluate_controller(
        controller_factory=lambda: DirectObservationController(agent, state_mean, state_std),
        dataset_id=dataset_id,
        scenario=scenario,
        episodes=episodes,
        seed=seed,
        ack_lag=ack_lag,
        cvar_alpha=cvar_alpha,
        method=method,
        environment_config=environment_config,
    )


def _recover_environment(
    dataset_id: str,
    environment_config: Mapping[str, Any] | None,
) -> Any:
    if environment_config is None:
        import minari

        dataset = minari.load_dataset(dataset_id, download=False)
        return dataset.recover_environment(eval_env=True)
    environment_type = str(environment_config["type"]).lower()
    if environment_type != "maniskill":
        raise ValueError(f"unsupported evaluation environment type: {environment_type}")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers task IDs
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    if str(environment_config.get("sim_backend", "physx_cpu")) != "physx_cpu":
        raise ValueError("the verified ManiSkill evaluation path currently requires physx_cpu")
    environment_kwargs: dict[str, Any] = {
        "num_envs": 1,
        "obs_mode": str(environment_config.get("obs_mode", "state")),
        "control_mode": str(environment_config["control_mode"]),
        "reward_mode": str(environment_config.get("reward_mode", "normalized_dense")),
        "sim_backend": "physx_cpu",
        "render_mode": None,
        "render_backend": "none",
        # Official ManiSkill imitation-learning evaluation reconfigures every
        # episode so task randomization is not silently held fixed.
        "reconfiguration_freq": 1,
    }
    if environment_config.get("max_episode_steps") is not None:
        # Passing the override only to our outer loop is insufficient: the
        # Gymnasium TimeLimit wrapper would still truncate at the task's
        # shorter default horizon.
        environment_kwargs["max_episode_steps"] = int(
            environment_config["max_episode_steps"]
        )
    environment = gym.make(
        str(environment_config["env_id"]),
        **environment_kwargs,
    )
    environment = CPUGymWrapper(
        environment,
        ignore_terminations=True,
        record_metrics=True,
    )
    return gym.wrappers.RescaleAction(environment, min_action=-1.0, max_action=1.0)


def _environment_horizon(
    environment: Any,
    environment_config: Mapping[str, Any] | None,
) -> int:
    """Resolve a finite episode horizon across Gymnasium wrapper stacks."""

    if environment_config is not None and environment_config.get("max_episode_steps") is not None:
        horizon = int(environment_config["max_episode_steps"])
        if horizon <= 0:
            raise ValueError("max_episode_steps must be positive")
        return horizon
    candidates = [environment, getattr(environment, "unwrapped", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        spec = getattr(candidate, "spec", None)
        horizon = getattr(spec, "max_episode_steps", None)
        if horizon is not None and int(horizon) > 0:
            return int(horizon)
        horizon = getattr(candidate, "_max_episode_steps", None)
        if horizon is not None and int(horizon) > 0:
            return int(horizon)
    raise ValueError(
        "unable to resolve a finite evaluation horizon; set environment.max_episode_steps"
    )


def evaluate_controller(
    controller_factory: Callable[[], DeploymentController],
    dataset_id: str,
    scenario: Mapping[str, Any],
    episodes: int,
    seed: int,
    ack_lag: int,
    cvar_alpha: float,
    method: str,
    environment_config: Mapping[str, Any] | None = None,
    privileged_selected_age: bool = False,
) -> dict[str, Any]:
    env = _recover_environment(dataset_id, environment_config)
    returns: list[float] = []
    lengths: list[int] = []
    generated_delays: list[int] = []
    ack_counts: list[int] = []
    inference_times_ms: list[float] = []
    controller_diagnostics: dict[str, list[float]] = {}
    episode_successes: list[float] = []
    episode_successes_at_end: list[float] = []
    episode_success_occupancies: list[float] = []
    episode_post_change_success_occupancies: list[float] = []
    episode_seeds: list[int] = []
    recovery_steps: list[int] = []
    horizon = _environment_horizon(env, environment_config)

    for episode in range(episodes):
        episode_seed = seed + 100_003 * episode
        observation, _ = env.reset(seed=episode_seed)
        channel = DelayedObservationChannel(
            delay_process=build_delay_process(scenario),
            rng=random.Random(episode_seed + 1),
            ack_lag=ack_lag,
        )
        channel.reset(np.asarray(observation).copy())
        controller = controller_factory()
        episode_seed_setter = getattr(controller, "set_episode_seed", None)
        if episode_seed_setter is not None:
            episode_seed_setter(episode_seed)
        controller.reset(np.asarray(observation).copy())
        total_return = 0.0
        episode_rewards: list[float] = []
        episode_ack_count = 0
        episode_success_samples: list[tuple[int, bool]] = []
        episode_length = horizon
        for step in range(horizon):
            channel_step = channel.advance(step)
            if privileged_selected_age:
                if channel_step.selected_packet is None:
                    raise RuntimeError("the delayed channel has no selected packet")
                setter = getattr(controller, "set_oracle_age", None)
                if setter is None:
                    raise TypeError("privileged age evaluation requires an oracle controller")
                setter(step - channel_step.selected_packet.generated_step)
            started = time.perf_counter_ns()
            action = controller.act(
                channel_step.policy_input,
                decision_step=step,
            )
            inference_times_ms.append((time.perf_counter_ns() - started) / 1e6)
            observation, reward, terminated, truncated, info = env.step(action)
            controller.observe_transition(
                np.asarray(action).copy(),
                float(reward),
                bool(terminated),
                bool(truncated),
            )
            for success_key in ("success", "is_success", "goal_achieved"):
                if success_key in info:
                    success_value = bool(np.asarray(info[success_key]).any())
                    episode_success_samples.append((step, success_value))
                    break
            packet = channel.send(np.asarray(observation).copy(), generated_step=step + 1)
            generated_delays.append(packet.delay)
            episode_ack_count += len(channel_step.policy_input.acknowledgements)
            total_return += float(reward)
            episode_rewards.append(float(reward))
            if terminated or truncated:
                episode_length = step + 1
                break
        returns.append(total_return)
        lengths.append(episode_length)
        episode_seeds.append(episode_seed)
        if scenario.get("type") == "changepoint":
            recovery_steps.append(
                _changepoint_recovery_steps(
                    episode_rewards,
                    int(scenario["change_step"]),
                )
            )
        ack_counts.append(episode_ack_count)
        success_summary = _episode_success_summary(
            episode_success_samples,
            scenario,
        )
        if success_summary is not None:
            episode_successes.append(success_summary["once"])
            episode_successes_at_end.append(success_summary["at_end"])
            episode_success_occupancies.append(success_summary["occupancy"])
            if "post_change_occupancy" in success_summary:
                episode_post_change_success_occupancies.append(
                    success_summary["post_change_occupancy"]
                )
        for name, value in controller.diagnostics().items():
            controller_diagnostics.setdefault(name, []).append(float(value))
    env.close()

    latency = np.asarray(inference_times_ms)
    result = {
        "method": method,
        "privileged_selected_age": privileged_selected_age,
        "scenario": scenario["name"],
        "episodes": episodes,
        "mean_return": statistics.fmean(returns),
        "stdev_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "worst_return": min(returns),
        "cvar_return": _cvar(returns, cvar_alpha),
        "returns": returns,
        "episode_seeds": episode_seeds,
        "episode_lengths": lengths,
        "mean_episode_length": statistics.fmean(lengths),
        "mean_generated_delay": statistics.fmean(generated_delays),
        "max_generated_delay": max(generated_delays),
        "generated_delay_histogram": {
            str(delay): generated_delays.count(delay)
            for delay in sorted(set(generated_delays))
        },
        "mean_ack_count": statistics.fmean(ack_counts),
        "inference_ms_p50": float(np.percentile(latency, 50)),
        "inference_ms_p95": float(np.percentile(latency, 95)),
    }
    result.update(
        {
            name: statistics.fmean(values)
            for name, values in controller_diagnostics.items()
        }
    )
    if episode_successes:
        # success_rate/successes remain stable aliases for the standard
        # ManiSkill success-once endpoint used by existing artifacts.
        result["success_rate"] = statistics.fmean(episode_successes)
        result["successes"] = episode_successes
        result["success_once_rate"] = statistics.fmean(episode_successes)
        result["successes_once"] = episode_successes
        result["success_at_end_rate"] = statistics.fmean(
            episode_successes_at_end
        )
        result["successes_at_end"] = episode_successes_at_end
        result["mean_success_occupancy"] = statistics.fmean(
            episode_success_occupancies
        )
        result["success_occupancies"] = episode_success_occupancies
        result["success_episodes_reported"] = len(episode_successes)
    if episode_post_change_success_occupancies:
        result["mean_post_change_success_occupancy"] = statistics.fmean(
            episode_post_change_success_occupancies
        )
        result["post_change_success_occupancies"] = (
            episode_post_change_success_occupancies
        )
    if recovery_steps:
        result["recovery_steps"] = recovery_steps
        result["mean_recovery_steps"] = statistics.fmean(recovery_steps)
    return result
