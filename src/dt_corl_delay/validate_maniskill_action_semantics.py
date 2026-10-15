"""Validate the normalized policy-action interface against a real ManiSkill env."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .delayed_eval import _environment_horizon, _recover_environment


def validate(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment_config = config["environment"]
    environment = _recover_environment(str(config["dataset_id"]), environment_config)
    try:
        action_low = np.asarray(environment.action_space.low, dtype=np.float32)
        action_high = np.asarray(environment.action_space.high, dtype=np.float32)
        if not np.allclose(action_low, -1.0) or not np.allclose(action_high, 1.0):
            raise AssertionError("the policy-facing action space is not normalized to [-1, 1]")

        observation, _ = environment.reset(seed=int(config.get("eval_seed", 0)))
        next_observation, reward, terminated, truncated, _ = environment.step(
            np.zeros(environment.action_space.shape, dtype=np.float32)
        )
        if not np.isfinite(np.asarray(observation)).all():
            raise AssertionError("reset observation contains non-finite values")
        if not np.isfinite(np.asarray(next_observation)).all() or not np.isfinite(reward):
            raise AssertionError("one-step transition contains non-finite values")

        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.resolve()),
            "environment": environment_config,
            "observation_shape": list(np.asarray(observation).shape),
            "action_shape": list(environment.action_space.shape),
            "resolved_horizon": _environment_horizon(environment, environment_config),
            "policy_action_low": action_low.tolist(),
            "policy_action_high": action_high.tolist(),
            "zero_action_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "passed": True,
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.config.resolve())
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
