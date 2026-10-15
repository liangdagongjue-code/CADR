"""End-to-end clean-training and delayed-deployment smoke experiment."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping

from .channel import DelayedObservationChannel
from .delay_process import build_delay_process
from .toy_control import FrozenLinearPolicy, LinearPlant, fit_clean_offline_policy


def _percentile_tail(values: list[float], alpha: float) -> float:
    count = max(1, math.ceil(len(values) * alpha))
    return statistics.fmean(sorted(values)[:count])


def _canonicalize(value: Any) -> Any:
    """Stabilize serialized floats across operating systems and libm builds."""

    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(_canonicalize(value), indent=2, sort_keys=True) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(_canonicalize(row), sort_keys=True) + "\n"
        for row in rows
    )
    path.write_bytes(payload.encode("utf-8"))


def _svg_polyline(values: list[float], width: int, height: int, maximum: float) -> str:
    if len(values) == 1:
        return f"0,{height / 2:.2f}"
    points: list[str] = []
    for index, value in enumerate(values):
        x = index * width / (len(values) - 1)
        y = height / 2 - (value / maximum) * (height * 0.42)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def render_trajectory_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width, panel_height = 1000, 260
    true_state = [float(row["true_state"]) for row in rows]
    observed_state = [float(row["policy_observation"]) for row in rows]
    delays = [float(row["generated_delay"]) for row in rows]
    state_max = max(1.0, *(abs(value) for value in true_state + observed_state))
    delay_max = max(1.0, *delays)
    state_true_points = _svg_polyline(true_state, width, panel_height, state_max)
    state_obs_points = _svg_polyline(observed_state, width, panel_height, state_max)
    delay_points = _svg_polyline(delays, width, panel_height, delay_max)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="650" viewBox="0 0 1040 650">
<rect width="1040" height="650" fill="#0b1020"/>
<text x="20" y="30" fill="#f4f7ff" font-family="sans-serif" font-size="20">Latent observation-delay smoke trajectory</text>
<g transform="translate(20,55)">
  <rect width="1000" height="260" fill="#121a2e" stroke="#334155"/>
  <line x1="0" y1="130" x2="1000" y2="130" stroke="#475569"/>
  <polyline points="{state_true_points}" fill="none" stroke="#38bdf8" stroke-width="2"/>
  <polyline points="{state_obs_points}" fill="none" stroke="#fb7185" stroke-width="2" opacity="0.9"/>
  <text x="12" y="24" fill="#38bdf8" font-family="sans-serif">true state</text>
  <text x="112" y="24" fill="#fb7185" font-family="sans-serif">policy observation</text>
</g>
<g transform="translate(20,355)">
  <rect width="1000" height="260" fill="#121a2e" stroke="#334155"/>
  <line x1="0" y1="130" x2="1000" y2="130" stroke="#475569"/>
  <polyline points="{delay_points}" fill="none" stroke="#fbbf24" stroke-width="2"/>
  <text x="12" y="24" fill="#fbbf24" font-family="sans-serif">environment-side generated delay (hidden from policy)</text>
</g>
</svg>
"""
    path.write_bytes(svg.encode("utf-8"))


def evaluate_scenario(
    scenario: Mapping[str, Any],
    checkpoint: Any,
    seed: int,
    episodes: int,
    horizon: int,
    ack_lag: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    sample_steps: list[dict[str, Any]] = []
    for episode in range(episodes):
        episode_seed = seed + 100_000 * episode
        plant = LinearPlant(rng=random.Random(episode_seed + 1))
        delay_process = build_delay_process(scenario)
        channel = DelayedObservationChannel(
            delay_process=delay_process,
            rng=random.Random(episode_seed + 2),
            ack_lag=ack_lag,
        )
        policy = FrozenLinearPolicy(checkpoint)
        state = plant.reset()
        channel.reset(state)
        total_return = 0.0
        generated_delays: list[int] = []
        for step in range(horizon):
            channel_step = channel.advance(step)
            policy_observation = channel_step.policy_input.observation
            action = policy.act(
                policy_observation,
                channel_step.policy_input.acknowledgements,
            )
            next_state, reward = plant.step(action)
            total_return += reward
            packet = channel.send(next_state, generated_step=step + 1)
            generated_delays.append(packet.delay)
            if episode == 0:
                selected = channel_step.selected_packet
                sample_steps.append(
                    {
                        "step": step,
                        "true_state": state,
                        "policy_observation": policy_observation,
                        "action": action,
                        "reward": reward,
                        "generated_delay": packet.delay,
                        "selected_observation_age": (
                            step - selected.generated_step if selected is not None else None
                        ),
                        "revealed_ack_delays": [
                            acknowledgement.delay
                            for acknowledgement in channel_step.policy_input.acknowledgements
                        ],
                    }
                )
            state = next_state
        episode_rows.append(
            {
                "scenario": scenario["name"],
                "episode": episode,
                "seed": episode_seed,
                "return": total_return,
                "mean_generated_delay": statistics.fmean(generated_delays),
                "max_generated_delay": max(generated_delays),
                "ack_count": policy.ack_count,
            }
        )
    return episode_rows, sample_steps


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = fit_clean_offline_policy(
        seed=int(config["seed"]),
        episodes=int(config["train_episodes"]),
        horizon=int(config["train_horizon"]),
    )
    _write_json(output_dir / "checkpoint.json", asdict(checkpoint))

    all_episodes: list[dict[str, Any]] = []
    sample_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario in config["scenarios"]:
        rows, sample = evaluate_scenario(
            scenario=scenario,
            checkpoint=checkpoint,
            seed=int(config["seed"]),
            episodes=int(config["eval_episodes"]),
            horizon=int(config["eval_horizon"]),
            ack_lag=int(config["ack_lag"]),
        )
        all_episodes.extend(rows)
        sample_by_scenario[str(scenario["name"])] = sample

    metrics: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "checkpoint": asdict(checkpoint),
        "scenarios": {},
    }
    for scenario in config["scenarios"]:
        name = str(scenario["name"])
        returns = [row["return"] for row in all_episodes if row["scenario"] == name]
        metrics["scenarios"][name] = {
            "mean_return": statistics.fmean(returns),
            "stdev_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
            "worst_return": min(returns),
            "cvar_return": _percentile_tail(returns, float(config["cvar_alpha"])),
        }

    _write_json(output_dir / "metrics.json", metrics)
    _write_jsonl(output_dir / "episodes.jsonl", all_episodes)
    for scenario_name, sample in sample_by_scenario.items():
        _write_jsonl(output_dir / f"trajectory_{scenario_name}.jsonl", sample)
        render_trajectory_svg(sample, output_dir / f"trajectory_{scenario_name}.svg")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = run(args.config.resolve(), args.output.resolve())
    print(json.dumps(metrics["scenarios"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
