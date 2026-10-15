"""Audit a replayed ManiSkill state dataset and write a checksum-bound manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .offline_data import load_maniskill_transitions


def audit(
    dataset_path: Path,
    dataset_id: str,
    output: Path,
    action_low: list[float] | None = None,
    action_high: list[float] | None = None,
) -> dict[str, object]:
    transitions = load_maniskill_transitions(
        dataset_path,
        dataset_id,
        torch.device("cpu"),
        action_low=action_low,
        action_high=action_high,
    )
    tensors = {
        "observations": transitions.observations,
        "actions": transitions.actions,
        "rewards": transitions.rewards,
        "next_observations": transitions.next_observations,
        "terminals": transitions.terminals,
    }
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": transitions.metadata,
        "all_tensors_finite": all(torch.isfinite(value).all().item() for value in tensors.values()),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "action_min": float(transitions.actions.min()),
        "action_max": float(transitions.actions.max()),
        "terminal_count": int(transitions.terminals.sum()),
        "state_normalization": {
            "epsilon": 1e-3,
            "mean_shape": list(transitions.state_mean.shape),
            "std_min": float(transitions.state_std.min()),
            "std_max": float(transitions.state_std.max()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-low")
    parser.add_argument("--action-high")
    args = parser.parse_args()
    result = audit(
        args.dataset_path.resolve(),
        args.dataset_id,
        args.output.resolve(),
        action_low=[float(value) for value in args.action_low.split(",")] if args.action_low else None,
        action_high=[float(value) for value in args.action_high.split(",")] if args.action_high else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
