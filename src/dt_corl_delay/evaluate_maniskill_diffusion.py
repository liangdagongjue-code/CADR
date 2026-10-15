"""Evaluate a frozen official ManiSkill Diffusion Policy under latent delay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .delayed_eval import evaluate_controller
from .maniskill_diffusion import (
    OfficialDiffusionPolicyController,
    load_official_diffusion_agent,
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
    config_path: Path,
    checkpoint_path: Path,
    source_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
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
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    obs_horizon = int(config.get("obs_horizon", 2))
    act_horizon = int(config.get("act_horizon", 8))
    agent, policy_provenance = load_official_diffusion_agent(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        source_root=source_root,
        device=device,
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        pred_horizon=int(config.get("pred_horizon", 16)),
    )
    evaluations: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        evaluations[str(scenario["name"])] = evaluate_controller(
            controller_factory=lambda: OfficialDiffusionPolicyController(
                agent=agent,
                device=device,
                obs_horizon=obs_horizon,
                act_horizon=act_horizon,
            ),
            dataset_id=str(config["dataset_id"]),
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=int(config["eval_seed"]),
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method="official_maniskill_diffusion_policy_direct_delayed_observation",
            environment_config=config["environment"],
        )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "official_ManiSkill_v3.0.1_state_Diffusion_Policy",
        "training_data_delay": "none_clean_demonstrations_only",
        "deployment_parameter_updates": False,
        "policy": policy_provenance,
        "information_boundary": "received_observations_only; latent delay and generation timestamp hidden",
        "primary_metric": "success_rate",
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        source_root=args.source_root.resolve(),
        output_dir=args.output.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
