"""Compare direct DP with CADR state reconstruction under latent delay."""

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
from .dynamics import DeltaDynamicsEnsemble
from .maniskill_diffusion import (
    CadrDiffusionPolicyController,
    OfficialDiffusionPolicyController,
    load_official_diffusion_agent,
    official_maniskill_source_provenance,
)
from .offline_data import load_offline_transitions
from .provenance import (
    file_input_provenance,
    package_source_provenance,
    verify_package_source_unchanged,
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
    policy_checkpoint_path: Path,
    dynamics_checkpoint_path: Path,
    source_root: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_snapshot = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    progress_path = output_dir / "progress.json"
    inputs_snapshot = output_dir / "inputs.json"
    local_source_provenance = package_source_provenance()
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if not config_snapshot.is_file() or json.loads(
            config_snapshot.read_text(encoding="utf-8")
        ) != config:
            raise ValueError("resume configuration does not match the output snapshot")
        # A completed run is immutable and idempotent. Returning its summary
        # also keeps runs created before inputs.json was introduced readable.
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        dataset_path = Path(config["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        requested_inputs = {
            "policy_checkpoint": file_input_provenance(policy_checkpoint_path),
            "dynamics_checkpoint": file_input_provenance(dynamics_checkpoint_path),
            "dataset": file_input_provenance(dataset_path),
            "source_root": str(source_root.resolve()),
            "official_maniskill_source": official_maniskill_source_provenance(source_root),
            "local_source_provenance": local_source_provenance,
        }
        if not inputs_snapshot.is_file() or json.loads(
            inputs_snapshot.read_text(encoding="utf-8")
        ) != requested_inputs:
            raise ValueError("resume checkpoint/source inputs do not match the output snapshot")
    else:
        dataset_path = Path(config["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        requested_inputs = {
            "policy_checkpoint": file_input_provenance(policy_checkpoint_path),
            "dynamics_checkpoint": file_input_provenance(dynamics_checkpoint_path),
            "dataset": file_input_provenance(dataset_path),
            "source_root": str(source_root.resolve()),
            "official_maniskill_source": official_maniskill_source_provenance(source_root),
            "local_source_provenance": local_source_provenance,
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(config_snapshot, config)
        _write_json(inputs_snapshot, requested_inputs)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    obs_horizon = int(config.get("obs_horizon", 2))
    act_horizon = int(config.get("act_horizon", 8))
    agent, policy_provenance = load_official_diffusion_agent(
        checkpoint_path=policy_checkpoint_path,
        dataset_path=dataset_path,
        source_root=source_root,
        device=device,
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        pred_horizon=int(config.get("pred_horizon", 16)),
    )
    dataset = load_offline_transitions(config, device)
    dynamics_payload = torch.load(
        dynamics_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    dynamics_config = dynamics_payload["config"]
    if str(dynamics_config["dataset_id"]) != str(config["dataset_id"]):
        raise ValueError("dynamics and evaluation datasets differ")
    dynamics = DeltaDynamicsEnsemble(
        state_dim=int(dataset.metadata["state_dim"]),
        action_dim=int(dataset.metadata["action_dim"]),
        hidden_dims=tuple(int(value) for value in dynamics_config["dynamics_hidden_dims"]),
        ensemble_size=int(dynamics_config["ensemble_size"]),
        learning_rate=float(dynamics_config["dynamics_learning_rate"]),
        device=device,
    )
    dynamics.load_checkpoint(dynamics_payload["dynamics"])
    dynamics.eval()

    common_controller_arguments = {
        "agent": agent,
        "device": device,
        "obs_horizon": obs_horizon,
        "act_horizon": act_horizon,
    }
    cadr_controller_arguments = {
        **common_controller_arguments,
        "act_horizon": int(config.get("cadr_act_horizon", act_horizon)),
        "dynamics": dynamics,
        "state_mean": dataset.state_mean,
        "state_std": dataset.state_std,
        "max_delay": int(config["max_delay"]),
        "ack_lag": int(config["ack_lag"]),
        "memory_steps": int(config["posterior_memory_steps"]),
        "decay_time": float(config["posterior_decay_time"]),
        "prior_strength": float(config["posterior_prior_strength"]),
        "risk_blend": float(config.get("risk_blend", 0.0)),
        "risk_threshold": float(config.get("risk_threshold", 0.0)),
        "age_decision_rule": str(config.get("age_decision_rule", "posterior_mean")),
        "entropy_gate_threshold": float(config.get("entropy_gate_threshold", 1.4)),
        "use_content_conditioning": bool(
            config.get("use_content_conditioning", True)
        ),
        "posterior_updates": bool(config.get("posterior_updates", True)),
    }
    evaluations: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        evaluations = progress.get("evaluations", {})
    ablation_variant = config.get("ablation_variant")
    cadr_method_key = (
        f"cadr_dp_{ablation_variant}" if ablation_variant else "cadr_dp"
    )
    cadr_method_name = (
        f"cadr_dp_ablation_{ablation_variant}_plus_official_maniskill_dp"
        if ablation_variant
        else "cadr_posterior_state_reconstruction_plus_official_maniskill_dp"
    )
    for scenario in config["scenarios"]:
        scenario_name = str(scenario["name"])
        scenario_evaluations = evaluations.setdefault(scenario_name, {})
        common_evaluation_arguments = {
            "dataset_id": str(config["dataset_id"]),
            "scenario": scenario,
            "episodes": int(config["eval_episodes"]),
            # Identical environment and diffusion-noise seeds are used for the
            # direct and CADR policies and across delay scenarios.
            "seed": int(config["eval_seed"]),
            "ack_lag": int(config["ack_lag"]),
            "cvar_alpha": float(config["cvar_alpha"]),
            "environment_config": config["environment"],
        }
        if "direct" not in scenario_evaluations:
            scenario_evaluations["direct"] = evaluate_controller(
                controller_factory=lambda: OfficialDiffusionPolicyController(
                    **common_controller_arguments
                ),
                method="official_maniskill_dp_direct_delayed_observation",
                **common_evaluation_arguments,
            )
            _write_json(progress_path, {"evaluations": evaluations})
        if cadr_method_key not in scenario_evaluations:
            scenario_evaluations[cadr_method_key] = evaluate_controller(
                controller_factory=lambda: CadrDiffusionPolicyController(
                    **cadr_controller_arguments
                ),
                method=cadr_method_name,
                **common_evaluation_arguments,
            )
            _write_json(progress_path, {"evaluations": evaluations})
        if (
            bool(config.get("include_oracle_upper_bound", False))
            and "oracle_dp" not in scenario_evaluations
        ):
            scenario_evaluations["oracle_dp"] = evaluate_controller(
                controller_factory=lambda: CadrDiffusionPolicyController(
                    **cadr_controller_arguments,
                    age_source="oracle",
                ),
                method=(
                    "privileged_oracle_age_state_reconstruction_plus_"
                    "official_maniskill_dp_upper_bound"
                ),
                privileged_selected_age=True,
                **common_evaluation_arguments,
            )
            _write_json(progress_path, {"evaluations": evaluations})

    clean = evaluations.get("clean_delay_0")
    clean_equivalence = None
    if clean is not None:
        exact_fields = (
            "returns",
            "episode_lengths",
            "successes",
            "successes_at_end",
            "success_occupancies",
        )
        field_equality = {
            field: clean["direct"].get(field) == clean[cadr_method_key].get(field)
            for field in exact_fields
        }
        clean_equivalence = {
            "field_exact_equality": field_equality,
            "all_required_fields_exactly_equal": all(field_equality.values()),
        }
        if ablation_variant is None and not clean_equivalence[
            "all_required_fields_exactly_equal"
        ]:
            raise ValueError(
                "main CADR-DP failed the preregistered clean no-regression gate: "
                + json.dumps(field_equality, sort_keys=True)
            )
    verify_package_source_unchanged(local_source_provenance)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CADR_posterior_state_reconstruction_plus_official_ManiSkill_DP",
        "training_data_delay": "none_clean_demonstrations_only",
        "deployment_parameter_updates": False,
        "local_source_provenance": local_source_provenance,
        "information_boundary": "received_observations_revealed_ACKs_and_executed_actions_only",
        "included_privileged_oracle_upper_bound": bool(
            config.get("include_oracle_upper_bound", False)
        ),
        "ablation_variant": ablation_variant,
        "cadr_method_key": cadr_method_key,
        "policy": policy_provenance,
        "dynamics_checkpoint": str(dynamics_checkpoint_path.resolve()),
        "dynamics_training_summary": {
            "step": int(dynamics_payload["step"]),
            "whole_episode_split": True,
        },
        "primary_metric": "success_rate",
        "paired_clean_equivalence": clean_equivalence,
        "evaluations": evaluations,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(
        config_path=args.config.resolve(),
        policy_checkpoint_path=args.policy_checkpoint.resolve(),
        dynamics_checkpoint_path=args.dynamics_checkpoint.resolve(),
        source_root=args.source_root.resolve(),
        output_dir=args.output.resolve(),
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
