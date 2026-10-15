"""Evaluate a clean offline KNN policy under the common delay channel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .delayed_eval import evaluate_frozen_policy
from .knn import KNNPolicyAgent
from .offline_data import load_offline_transitions


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _write_json(output_dir / "config.json", config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_offline_transitions(config, device)
    agent = KNNPolicyAgent(
        dataset.observations,
        dataset.actions,
        neighbors=int(config.get("neighbors", 1)),
        temperature=float(config.get("neighbor_temperature", 0.1)),
    )
    evaluations: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        evaluations[str(scenario["name"])] = evaluate_frozen_policy(
            agent=agent,
            state_mean=dataset.state_mean,
            state_std=dataset.state_std,
            dataset_id=str(config["dataset_id"]),
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=int(config["eval_seed"]),
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method="clean_offline_knn_action_retrieval",
            environment_config=config.get("environment"),
        )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "KNN_offline_action_retrieval",
        "purpose": "manipulation_dataset_usability_and_nonparametric_baseline",
        "training_data_delay": "none_clean_only",
        "deployment_parameter_updates": False,
        "policy": agent.provenance(),
        "dataset": dataset.metadata,
        "evaluations": evaluations,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
