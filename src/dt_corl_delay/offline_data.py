"""Offline transition loading with explicit clean-data semantics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass
class OfflineTransitions:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminals: torch.Tensor
    episode_ids: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return self.observations.shape[0]

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator,
        subset_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if subset_indices is None:
            indices = torch.randint(
                low=0,
                high=self.size,
                size=(batch_size,),
                device=self.observations.device,
                generator=generator,
            )
        else:
            if subset_indices.ndim != 1 or subset_indices.numel() == 0:
                raise ValueError("subset_indices must be a non-empty vector")
            positions = torch.randint(
                low=0,
                high=subset_indices.numel(),
                size=(batch_size,),
                device=self.observations.device,
                generator=generator,
            )
            indices = subset_indices[positions]
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.terminals[indices],
        )


def load_minari_transitions(dataset_id: str, device: torch.device) -> OfflineTransitions:
    """Load only clean transitions; no delayed view is created during training."""

    import minari

    dataset = minari.load_dataset(dataset_id, download=False)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    truncations = 0
    episodes = 0
    for episode in dataset.iterate_episodes():
        episode_observations = np.asarray(episode.observations)
        episode_actions = np.asarray(episode.actions)
        episode_rewards = np.asarray(episode.rewards)
        episode_terminals = np.asarray(episode.terminations, dtype=np.float32)
        episode_truncations = np.asarray(episode.truncations, dtype=bool)
        if episode_observations.shape[0] != episode_actions.shape[0] + 1:
            raise ValueError("Minari episode does not contain T+1 observations")
        observations.append(episode_observations[:-1])
        next_observations.append(episode_observations[1:])
        actions.append(episode_actions)
        rewards.append(episode_rewards[:, None])
        # Time-limit truncation remains bootstrap-able, matching standard
        # offline MuJoCo practice. Only true terminations stop bootstrapping.
        terminals.append(episode_terminals[:, None])
        episode_ids.append(
            np.full(episode_actions.shape[0], episodes, dtype=np.int64)
        )
        truncations += int(episode_truncations.sum())
        episodes += 1

    observations_np = np.concatenate(observations).astype(np.float32)
    next_observations_np = np.concatenate(next_observations).astype(np.float32)
    actions_np = np.concatenate(actions).astype(np.float32)
    rewards_np = np.concatenate(rewards).astype(np.float32)
    terminals_np = np.concatenate(terminals).astype(np.float32)
    episode_ids_np = np.concatenate(episode_ids)
    state_mean_np = observations_np.mean(axis=0, keepdims=True)
    state_std_np = observations_np.std(axis=0, keepdims=True) + 1e-3
    observations_np = (observations_np - state_mean_np) / state_std_np
    next_observations_np = (next_observations_np - state_mean_np) / state_std_np

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values)).to(device=device)

    return OfflineTransitions(
        observations=tensor(observations_np),
        actions=tensor(actions_np),
        rewards=tensor(rewards_np),
        next_observations=tensor(next_observations_np),
        terminals=tensor(terminals_np),
        episode_ids=torch.from_numpy(episode_ids_np).to(device=device),
        state_mean=tensor(state_mean_np),
        state_std=tensor(state_std_np),
        metadata={
            "dataset_id": dataset_id,
            "episodes": episodes,
            "transitions": observations_np.shape[0],
            "truncations": truncations,
            "state_dim": observations_np.shape[1],
            "action_dim": actions_np.shape[1],
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_maniskill_transitions(
    dataset_path: str | Path,
    dataset_id: str,
    device: torch.device,
    action_low: Sequence[float] | None = None,
    action_high: Sequence[float] | None = None,
) -> OfflineTransitions:
    """Load a replayed ManiSkill state dataset while preserving raw provenance."""

    import h5py

    path = Path(dataset_path)
    if not path.is_absolute():
        project_root = Path(os.environ.get("DT_CORL_PROJECT_ROOT", "."))
        path = project_root / path
    path = path.resolve()
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("ManiSkill replay requires paired HDF5 and JSON files")
    replay_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    truncations = 0
    successes = 0
    with h5py.File(path, "r") as handle:
        trajectory_names = sorted(
            handle.keys(),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        for episode_index, trajectory_name in enumerate(trajectory_names):
            trajectory = handle[trajectory_name]
            episode_observations = np.asarray(trajectory["obs"], dtype=np.float32)
            episode_actions = np.asarray(trajectory["actions"], dtype=np.float32)
            episode_rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
            episode_terminals = np.asarray(trajectory["terminated"], dtype=np.float32)
            episode_truncations = np.asarray(trajectory["truncated"], dtype=bool)
            if episode_observations.shape[0] != episode_actions.shape[0] + 1:
                raise ValueError(f"{trajectory_name} does not contain T+1 observations")
            if episode_rewards.shape[0] != episode_actions.shape[0]:
                raise ValueError(f"{trajectory_name} rewards do not align with actions")
            observations.append(episode_observations[:-1])
            next_observations.append(episode_observations[1:])
            actions.append(episode_actions)
            rewards.append(episode_rewards[:, None])
            terminals.append(episode_terminals[:, None])
            episode_ids.append(
                np.full(episode_actions.shape[0], episode_index, dtype=np.int64)
            )
            truncations += int(episode_truncations.sum())
            if "success" in trajectory:
                successes += int(np.asarray(trajectory["success"], dtype=bool).sum())

    observations_np = np.concatenate(observations)
    next_observations_np = np.concatenate(next_observations)
    actions_np = np.concatenate(actions)
    raw_action_min = actions_np.min(axis=0)
    raw_action_max = actions_np.max(axis=0)
    action_transform: dict[str, Any] = {"type": "identity"}
    if action_low is not None or action_high is not None:
        if action_low is None or action_high is None:
            raise ValueError("action_low and action_high must be provided together")
        action_low_np = np.asarray(action_low, dtype=np.float32)
        action_high_np = np.asarray(action_high, dtype=np.float32)
        if action_low_np.shape != (actions_np.shape[1],) or action_high_np.shape != action_low_np.shape:
            raise ValueError("action bounds do not match the dataset action dimension")
        if np.any(action_high_np <= action_low_np):
            raise ValueError("every action upper bound must exceed its lower bound")
        tolerance = 1e-5
        if np.any(raw_action_min < action_low_np - tolerance) or np.any(
            raw_action_max > action_high_np + tolerance
        ):
            raise ValueError("dataset actions fall outside configured environment bounds")
        actions_np = 2.0 * (actions_np - action_low_np) / (action_high_np - action_low_np) - 1.0
        actions_np = np.clip(actions_np, -1.0, 1.0)
        action_transform = {
            "type": "affine_environment_bounds_to_minus_one_one",
            "environment_low": action_low_np.tolist(),
            "environment_high": action_high_np.tolist(),
            "raw_dataset_min": raw_action_min.tolist(),
            "raw_dataset_max": raw_action_max.tolist(),
        }
    rewards_np = np.concatenate(rewards)
    terminals_np = np.concatenate(terminals)
    episode_ids_np = np.concatenate(episode_ids)
    state_mean_np = observations_np.mean(axis=0, keepdims=True)
    state_std_np = observations_np.std(axis=0, keepdims=True) + 1e-3
    observations_np = (observations_np - state_mean_np) / state_std_np
    next_observations_np = (next_observations_np - state_mean_np) / state_std_np

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values)).to(device=device)

    environment = replay_metadata.get("env_info", {})
    return OfflineTransitions(
        observations=tensor(observations_np),
        actions=tensor(actions_np),
        rewards=tensor(rewards_np),
        next_observations=tensor(next_observations_np),
        terminals=tensor(terminals_np),
        episode_ids=torch.from_numpy(episode_ids_np).to(device=device),
        state_mean=tensor(state_mean_np),
        state_std=tensor(state_std_np),
        metadata={
            "dataset_id": dataset_id,
            "dataset_type": "maniskill_replay_hdf5",
            "dataset_path": str(path),
            "dataset_sha256": _sha256(path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "environment": environment,
            "episodes": len(episode_ids),
            "transitions": observations_np.shape[0],
            "truncations": truncations,
            "successful_transition_labels": successes,
            "state_dim": observations_np.shape[1],
            "action_dim": actions_np.shape[1],
            "action_transform": action_transform,
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
        },
    )


def load_offline_transitions(config: dict[str, Any], device: torch.device) -> OfflineTransitions:
    dataset_type = str(config.get("dataset_type", "minari")).lower()
    if dataset_type == "minari":
        return load_minari_transitions(str(config["dataset_id"]), device)
    if dataset_type == "maniskill":
        return load_maniskill_transitions(
            config["dataset_path"],
            str(config["dataset_id"]),
            device,
            action_low=config.get("action_low"),
            action_high=config.get("action_high"),
        )
    raise ValueError(f"unsupported dataset_type: {dataset_type}")
