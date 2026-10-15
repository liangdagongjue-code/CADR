"""Headless state-control smoke for the pinned ManiSkill environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers environments
import numpy as np
import torch


def _any(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(torch.as_tensor(value).any().item())
    return bool(np.asarray(value).any())


def _finite(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(torch.as_tensor(value)).all().item())
    return bool(np.isfinite(np.asarray(value)).all())


def run(
    output: Path,
    backend: str,
    steps: int,
    seed: int,
    num_envs: int,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend="none",
        sim_backend=backend,
    )
    observation, info = env.reset(seed=seed)
    initial_shape = list(observation.shape)
    successes = 0
    started = time.perf_counter()
    completed_steps = 0
    for _ in range(steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if "success" in info:
            successes += int(_any(info["success"]))
        completed_steps += 1
        if _any(terminated) or _any(truncated):
            observation, info = env.reset()
    elapsed = time.perf_counter() - started
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "PickCube-v1",
        "obs_mode": "state",
        "control_mode": "pd_joint_pos",
        "sim_backend": backend,
        "render_mode": None,
        "render_backend": "none",
        "headless": True,
        "num_envs": num_envs,
        "steps": completed_steps,
        "environment_transitions": completed_steps * num_envs,
        "elapsed_seconds": elapsed,
        "steps_per_second": completed_steps / elapsed,
        "environment_transitions_per_second": completed_steps * num_envs / elapsed,
        "observation_shape": initial_shape,
        "action_shape": list(env.action_space.shape),
        "reward_finite": _finite(reward),
        "success_observations": successes,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    env.close()
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="physx_cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output.resolve(), args.backend, args.steps, args.seed, args.num_envs),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
