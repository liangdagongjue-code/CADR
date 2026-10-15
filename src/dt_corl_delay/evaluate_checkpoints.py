"""Evaluate frozen IQL and CADR checkpoints without retraining either model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .controllers import CadrController, DirectObservationController
from .delayed_eval import evaluate_controller
from .dynamics import DeltaDynamicsEnsemble
from .provenance import (
    file_input_provenance,
    package_source_provenance,
    verify_package_source_unchanged,
)
from .train_cadr import _load_agent


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run(
    config_path: Path,
    base_checkpoint_path: Path,
    dynamics_checkpoint_path: Path,
    output_dir: Path,
    resume: bool = False,
    include_cadr: bool = True,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_snapshot = output_dir / "config.json"
    inputs_snapshot = output_dir / "inputs.json"
    summary_path = output_dir / "summary.json"
    partial_path = output_dir / "summary_partial.json"
    local_source_provenance = package_source_provenance()
    if output_dir.exists():
        if not resume:
            raise FileExistsError(output_dir)
        if not config_snapshot.is_file() or json.loads(
            config_snapshot.read_text(encoding="utf-8")
        ) != config:
            raise ValueError("resume configuration does not match the output snapshot")
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        requested_inputs = {
            "base_checkpoint": file_input_provenance(base_checkpoint_path),
            "dynamics_checkpoint": file_input_provenance(dynamics_checkpoint_path),
            "include_cadr": include_cadr,
            "local_source_provenance": local_source_provenance,
        }
        if not inputs_snapshot.is_file() or json.loads(
            inputs_snapshot.read_text(encoding="utf-8")
        ) != requested_inputs:
            raise ValueError("resume checkpoint inputs do not match the output snapshot")
    else:
        requested_inputs = {
            "base_checkpoint": file_input_provenance(base_checkpoint_path),
            "dynamics_checkpoint": file_input_provenance(dynamics_checkpoint_path),
            "include_cadr": include_cadr,
            "local_source_provenance": local_source_provenance,
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(config_snapshot, config)
        _write_json_atomic(inputs_snapshot, requested_inputs)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))

    base_payload = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    dynamics_payload = torch.load(
        dynamics_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if config["dataset_id"] != base_payload["dataset_metadata"]["dataset_id"]:
        raise ValueError("evaluation and base checkpoint datasets differ")
    agent = _load_agent(base_payload, device)
    base_algorithm = str(
        base_payload.get("agent_kind", base_payload["config"].get("algorithm", "iql"))
    ).lower()
    dynamics_config = dynamics_payload["config"]
    dynamics_state = dynamics_payload["dynamics"]
    dynamics = DeltaDynamicsEnsemble(
        state_dim=int(dynamics_state["state_dim"]),
        action_dim=int(dynamics_state["action_dim"]),
        hidden_dims=tuple(int(value) for value in dynamics_config["dynamics_hidden_dims"]),
        ensemble_size=int(dynamics_state["ensemble_size"]),
        learning_rate=float(dynamics_config["dynamics_learning_rate"]),
        device=device,
    )
    dynamics.load_checkpoint(dynamics_state)
    dynamics.eval()
    state_mean = base_payload["state_mean"].to(device)
    state_std = base_payload["state_std"].to(device)

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": (
            f"paired_frozen_{base_algorithm.upper()}_vs_CADR"
            if include_cadr
            else f"frozen_{base_algorithm.upper()}_direct_only"
        ),
        "training_data_delay": "none_clean_only",
        "deployment_parameter_updates": False,
        "local_source_provenance": local_source_provenance,
        "information_boundary": "delayed_observation_and_revealed_acks_only",
        "base_checkpoint": str(base_checkpoint_path),
        "dynamics_checkpoint": str(dynamics_checkpoint_path),
        "evaluations": {},
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        summary["evaluations"] = partial.get("evaluations", {})
    evaluation_started = time.perf_counter()
    for scenario in config["scenarios"]:
        scenario_name = str(scenario["name"])
        if scenario_name in summary["evaluations"]:
            continue
        paired_seed = int(config["eval_seed"])
        scenario_started = time.perf_counter()
        direct = evaluate_controller(
            controller_factory=lambda: DirectObservationController(agent, state_mean, state_std),
            dataset_id=config["dataset_id"], scenario=scenario,
            episodes=int(config["eval_episodes"]), seed=paired_seed,
            ack_lag=int(config["ack_lag"]), cvar_alpha=float(config["cvar_alpha"]),
            method=f"direct_observation_{base_algorithm}",
            environment_config=config.get("environment"),
        )
        scenario_results: dict[str, Any] = {"direct": direct}
        if include_cadr:
            cadr = evaluate_controller(
                controller_factory=lambda: CadrController(
                    agent=agent, dynamics=dynamics, state_mean=state_mean, state_std=state_std,
                    max_delay=int(config["max_delay"]), ack_lag=int(config["ack_lag"]),
                    memory_steps=int(config["posterior_memory_steps"]),
                    decay_time=float(config["posterior_decay_time"]),
                    prior_strength=float(config["posterior_prior_strength"]),
                    risk_blend=float(config.get("risk_blend", 0.0)),
                    risk_threshold=float(config.get("risk_threshold", 0.0)),
                    age_decision_rule=str(config.get("age_decision_rule", "posterior_mean")),
                    entropy_gate_threshold=float(config.get("entropy_gate_threshold", 1.4)),
                    use_content_conditioning=bool(
                        config.get("use_content_conditioning", True)
                    ),
                    posterior_updates=bool(config.get("posterior_updates", True)),
                ),
                dataset_id=config["dataset_id"], scenario=scenario,
                episodes=int(config["eval_episodes"]), seed=paired_seed,
                ack_lag=int(config["ack_lag"]), cvar_alpha=float(config["cvar_alpha"]),
                method=f"cadr_{base_algorithm}",
                environment_config=config.get("environment"),
            )
            scenario_results.update(
                {
                    "cadr": cadr,
                    "paired_mean_return_delta": (
                        cadr["mean_return"] - direct["mean_return"]
                    ),
                }
            )
        scenario_results["wall_seconds"] = time.perf_counter() - scenario_started
        summary["evaluations"][scenario_name] = scenario_results
        _write_json_atomic(partial_path, summary)
    summary["evaluation_seconds"] = time.perf_counter() - evaluation_started
    verify_package_source_unchanged(local_source_provenance)
    _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="evaluate only the direct baseline and do not emit a CADR method",
    )
    args = parser.parse_args()
    result = run(
        args.config.resolve(), args.base_checkpoint.resolve(),
        args.dynamics_checkpoint.resolve(), args.output.resolve(), resume=args.resume,
        include_cadr=not args.direct_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
