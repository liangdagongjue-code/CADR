"""Deterministically score a CADR dynamics checkpoint on its full held-out split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .dynamics import DeltaDynamicsEnsemble
from .offline_data import load_offline_transitions
from .provenance import (
    file_input_provenance,
    package_source_provenance,
    verify_package_source_unchanged,
)
from .train_cadr import (
    _full_multistep_validation_metrics,
    _full_validation_metrics,
)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run(
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 4096,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    local_source_provenance = package_source_provenance()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested but is unavailable")
    dataset = load_offline_transitions(config, device)
    dynamics_state = payload["dynamics"]
    dynamics = DeltaDynamicsEnsemble(
        state_dim=int(dynamics_state["state_dim"]),
        action_dim=int(dynamics_state["action_dim"]),
        hidden_dims=tuple(int(value) for value in config["dynamics_hidden_dims"]),
        ensemble_size=int(dynamics_state["ensemble_size"]),
        learning_rate=float(config["dynamics_learning_rate"]),
        device=device,
    )
    dynamics.load_checkpoint(dynamics_state)
    dynamics.eval()

    split_generator = torch.Generator(device=device).manual_seed(seed + 303)
    episode_ids = torch.unique(dataset.episode_ids, sorted=True)
    episode_permutation = episode_ids[
        torch.randperm(episode_ids.numel(), generator=split_generator, device=device)
    ]
    validation_episode_count = max(1, int(round(episode_ids.numel() * 0.1)))
    validation_episode_ids = episode_permutation[:validation_episode_count]
    validation_indices = torch.nonzero(
        torch.isin(dataset.episode_ids, validation_episode_ids),
        as_tuple=False,
    ).squeeze(1)
    metrics = _full_validation_metrics(
        dynamics,
        dataset,
        validation_indices,
        batch_size,
    )
    metrics.update(
        _full_multistep_validation_metrics(
            dynamics,
            dataset,
            validation_episode_ids,
            int(config["max_delay"]),
            batch_size,
        )
    )
    verify_package_source_unchanged(local_source_provenance)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "deterministic_full_split_dynamics_validation",
        "checkpoint": file_input_provenance(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "dataset_id": str(config["dataset_id"]),
        "validation": metrics,
        "split": {
            "method": "whole_episode_random_90_10",
            "split_seed": seed + 303,
            "validation_episodes": int(validation_episode_ids.numel()),
            "validation_episode_ids": [
                int(value) for value in validation_episode_ids.cpu().tolist()
            ],
        },
        "local_source_provenance": local_source_provenance,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    result = run(
        args.checkpoint.resolve(),
        args.output.resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
