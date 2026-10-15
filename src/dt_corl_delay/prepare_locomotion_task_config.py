"""Materialize a frozen Minari configuration for one task and seed.

The maintained locomotion tasks use a 1,000-step horizon.  The exact Minari
Adroit expert-v2 tasks use 200 steps, so temporal scenario landmarks are scaled
to that horizon while every delay law and probability remains unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = {
    "mujoco/halfcheetah/medium-v0": 1000,
    "mujoco/hopper/medium-v0": 1000,
    "mujoco/walker2d/medium-v0": 1000,
    "D4RL/door/expert-v2": 200,
    "D4RL/pen/expert-v2": 200,
    "D4RL/hammer/expert-v2": 200,
    "D4RL/relocate/expert-v2": 200,
}


def materialize(
    template: Path,
    output: Path,
    dataset_id: str,
    seed: int,
) -> dict[str, Any]:
    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported formal Minari dataset: {dataset_id}")
    config = json.loads(template.read_text(encoding="utf-8"))
    config["dataset_id"] = dataset_id
    config["seed"] = seed
    task_horizon = SUPPORTED_DATASETS[dataset_id]
    if dataset_id.startswith("D4RL/") and "required_methods" in config:
        config["required_methods"] = [
            method
            for method in config["required_methods"]
            if method != "direct_observation_cql"
        ]
    if task_horizon != 1000:
        for scenario in config.get("scenarios", []):
            if "change_step" in scenario:
                scenario["change_step"] = task_horizon // 4
            if "horizon_steps" in scenario:
                scenario["horizon_steps"] = task_horizon
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"generated config already exists with different content: {output}")
    output.write_text(serialized, encoding="utf-8", newline="\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = materialize(
        args.template.resolve(),
        args.output.resolve(),
        args.dataset_id,
        args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "dataset_id": config["dataset_id"],
                "seed": config["seed"],
            }
        )
    )


if __name__ == "__main__":
    main()
