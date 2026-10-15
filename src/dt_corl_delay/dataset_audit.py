"""Audit a Minari dataset before it is admitted to training experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import minari
import numpy as np


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "finite": bool(np.isfinite(values).all()),
    }


def audit(dataset_id: str, output: Path) -> dict[str, Any]:
    dataset = minari.load_dataset(dataset_id, download=False)
    episode_count = 0
    transition_count = 0
    termination_count = 0
    truncation_count = 0
    episode_lengths: list[int] = []
    observation_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    reward_chunks: list[np.ndarray] = []
    digest = hashlib.sha256()

    for episode in dataset.iterate_episodes():
        observations = np.asarray(episode.observations)
        actions = np.asarray(episode.actions)
        rewards = np.asarray(episode.rewards)
        terminations = np.asarray(episode.terminations, dtype=bool)
        truncations = np.asarray(episode.truncations, dtype=bool)
        if observations.shape[0] != actions.shape[0] + 1:
            raise ValueError(
                f"episode {episode_count}: expected T+1 observations, got "
                f"{observations.shape[0]} observations and {actions.shape[0]} actions"
            )
        length = actions.shape[0]
        if rewards.shape[0] != length:
            raise ValueError(f"episode {episode_count}: reward length mismatch")
        if terminations.shape[0] != length or truncations.shape[0] != length:
            raise ValueError(f"episode {episode_count}: done-array length mismatch")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError(f"episode {episode_count}: non-finite state or action")

        episode_count += 1
        transition_count += length
        termination_count += int(terminations.sum())
        truncation_count += int(truncations.sum())
        episode_lengths.append(length)
        observation_chunks.append(observations[:-1])
        action_chunks.append(actions)
        reward_chunks.append(rewards)
        for values in (observations, actions, rewards, terminations, truncations):
            contiguous = np.ascontiguousarray(values)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())

    observations = np.concatenate(observation_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    rewards = np.concatenate(reward_chunks, axis=0)
    env = dataset.recover_environment(eval_env=True)
    initial_observation, _ = env.reset(seed=20260823)
    next_observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    env_spec = {
        "id": env.spec.id if env.spec is not None else None,
        "max_episode_steps": env.spec.max_episode_steps if env.spec is not None else None,
        "reset_observation_shape": list(np.asarray(initial_observation).shape),
        "step_observation_shape": list(np.asarray(next_observation).shape),
        "step_reward_finite": bool(np.isfinite(reward)),
        "step_terminated": bool(terminated),
        "step_truncated": bool(truncated),
    }
    env.close()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "dataset_sha256": digest.hexdigest(),
        "episodes": episode_count,
        "transitions": transition_count,
        "terminations": termination_count,
        "truncations": truncation_count,
        "episode_length": _array_summary(np.asarray(episode_lengths)),
        "observations": _array_summary(observations),
        "actions": _array_summary(actions),
        "rewards": _array_summary(rewards),
        "environment": env_spec,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset, args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

