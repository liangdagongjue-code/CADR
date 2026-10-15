"""Verify physics-only and EGL offscreen MuJoCo execution on a headless host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

# This must be set before importing MuJoCo through Gymnasium.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio.v3 as iio
import mujoco
import numpy as np

from .channel import DelayedObservationChannel
from .delay_process import build_delay_process


def _physics_throughput(seed: int, steps: int) -> float:
    env = gym.make("HalfCheetah-v5")
    observation, _ = env.reset(seed=seed)
    generator = np.random.default_rng(seed)
    started = time.perf_counter()
    for _ in range(steps):
        action = generator.uniform(env.action_space.low, env.action_space.high)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    elapsed = time.perf_counter() - started
    env.close()
    del observation
    return steps / elapsed


def _render_delayed_rollout(
    output_dir: Path,
    seed: int,
    steps: int,
    render_every: int,
) -> dict[str, float | int | str]:
    env = gym.make(
        "HalfCheetah-v5",
        render_mode="rgb_array",
        width=480,
        height=368,
    )
    observation, _ = env.reset(seed=seed)
    policy_rng = np.random.default_rng(seed + 1)
    projection = policy_rng.normal(
        loc=0.0,
        scale=0.12,
        size=(env.action_space.shape[0], observation.shape[0]),
    )
    delay_process = build_delay_process(
        {
            "type": "markov",
            "regimes": [
                {"delays": [0, 1, 2], "probabilities": [0.65, 0.25, 0.10]},
                {"delays": [3, 5, 7], "probabilities": [0.20, 0.45, 0.35]},
            ],
            "transition": [[0.96, 0.04], [0.10, 0.90]],
        }
    )
    channel = DelayedObservationChannel(
        delay_process=delay_process,
        rng=random.Random(seed + 2),
        ack_lag=1,
    )
    channel.reset(observation.copy())

    frames: list[np.ndarray] = []
    delays: list[int] = []
    total_return = 0.0
    ack_count = 0
    started = time.perf_counter()
    for step in range(steps):
        channel_step = channel.advance(step)
        policy_observation = np.asarray(channel_step.policy_input.observation)
        action = np.tanh(projection @ policy_observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        packet = channel.send(observation.copy(), generated_step=step + 1)
        delays.append(packet.delay)
        total_return += float(reward)
        ack_count += len(channel_step.policy_input.acknowledgements)
        if step % render_every == 0:
            frames.append(np.asarray(env.render()))
        if terminated or truncated:
            break
    render_elapsed = time.perf_counter() - started
    env.close()

    video_path = output_dir / "halfcheetah_egl.mp4"
    first_frame_path = output_dir / "halfcheetah_first_frame.png"
    iio.imwrite(video_path, np.stack(frames), fps=30, codec="libx264", quality=7)
    iio.imwrite(first_frame_path, frames[0])
    return {
        "rollout_steps": len(delays),
        "rendered_frames": len(frames),
        "rollout_return": total_return,
        "mean_generated_delay": float(np.mean(delays)),
        "max_generated_delay": max(delays),
        "revealed_ack_count": ack_count,
        "render_loop_steps_per_second": len(delays) / render_elapsed,
        "video": video_path.name,
        "first_frame": first_frame_path.name,
    }


def run(output_dir: Path, seed: int, physics_steps: int, render_steps: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "display_environment": os.environ.get("DISPLAY"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "versions": {
            "gymnasium": gym.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "physics_steps": physics_steps,
        "physics_steps_per_second": _physics_throughput(seed, physics_steps),
    }
    metrics["delayed_render_rollout"] = _render_delayed_rollout(
        output_dir=output_dir,
        seed=seed + 10_000,
        steps=render_steps,
        render_every=2,
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    (output_dir / "headless_metrics.json").write_bytes(payload.encode("utf-8"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--physics-steps", type=int, default=5000)
    parser.add_argument("--render-steps", type=int, default=240)
    args = parser.parse_args()
    metrics = run(
        output_dir=args.output.resolve(),
        seed=args.seed,
        physics_steps=args.physics_steps,
        render_steps=args.render_steps,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
