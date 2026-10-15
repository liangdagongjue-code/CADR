"""Compose a frozen task config with the shared 50 ms network-trace suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRACE_OVERRIDE_KEYS = (
    "experiment_series",
    "eval_seed",
    "eval_episodes",
    "max_delay",
    "posterior_memory_steps",
    "posterior_decay_time",
    "posterior_prior_strength",
    "scenarios",
)


def materialize(
    task_template: Path,
    trace_template: Path,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    config = json.loads(task_template.read_text(encoding="utf-8"))
    trace_config = json.loads(trace_template.read_text(encoding="utf-8"))
    missing = [key for key in TRACE_OVERRIDE_KEYS if key not in trace_config]
    if missing:
        raise ValueError(f"trace template is missing required keys: {missing}")
    for key in TRACE_OVERRIDE_KEYS:
        config[key] = trace_config[key]
    config["seed"] = seed
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"generated config already exists with different content: {output}")
    output.write_text(serialized, encoding="utf-8", newline="\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-template", type=Path, required=True)
    parser.add_argument("--trace-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = materialize(
        args.task_template.resolve(),
        args.trace_template.resolve(),
        args.output.resolve(),
        args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "seed": result["seed"],
                "environment": result["environment"]["env_id"],
                "scenarios": len(result["scenarios"]),
            }
        )
    )


if __name__ == "__main__":
    main()
