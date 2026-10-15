"""Evaluate one trained official DT-CORL adapter with ACK-MAP and oracle age."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from .controllers import DirectObservationController
from .delayed_eval import evaluate_controller
from .offline_data import load_offline_transitions
from .official_dtcorl import (
    OfficialDTCorlAckMapController,
    build_official_reconstructor,
    official_source_provenance,
)
from .provenance import (
    file_input_provenance,
    package_source_provenance,
    verify_package_source_unchanged,
)
from .train_cadr import _load_agent


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
    base_checkpoint_path: Path,
    dtcorl_checkpoint_path: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    evaluation_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_snapshot = output_dir / "config.json"
    inputs_snapshot = output_dir / "inputs.json"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    local_source_provenance = package_source_provenance()
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if not config_snapshot.is_file() or json.loads(
            config_snapshot.read_text(encoding="utf-8")
        ) != evaluation_config:
            raise ValueError("resume configuration does not match the output snapshot")
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        requested_inputs = {
            "base_checkpoint": file_input_provenance(base_checkpoint_path),
            "dtcorl_checkpoint": file_input_provenance(dtcorl_checkpoint_path),
            "official_dtcorl_source": official_source_provenance(),
            "local_source_provenance": local_source_provenance,
        }
        if not inputs_snapshot.is_file() or json.loads(
            inputs_snapshot.read_text(encoding="utf-8")
        ) != requested_inputs:
            raise ValueError("resume checkpoint inputs do not match the output snapshot")
    else:
        requested_inputs = {
            "base_checkpoint": file_input_provenance(base_checkpoint_path),
            "dtcorl_checkpoint": file_input_provenance(dtcorl_checkpoint_path),
            "official_dtcorl_source": official_source_provenance(),
            "local_source_provenance": local_source_provenance,
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(config_snapshot, evaluation_config)
        _write_json(inputs_snapshot, requested_inputs)
    device = torch.device(evaluation_config.get("device", "cuda"))
    base_payload = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    dtcorl_payload = torch.load(dtcorl_checkpoint_path, map_location=device, weights_only=False)
    training_config = dtcorl_payload["config"]
    dataset_id = str(evaluation_config["dataset_id"])
    if dataset_id != str(base_payload["dataset_metadata"]["dataset_id"]):
        raise ValueError("evaluation and frozen policy use different datasets")
    if dataset_id != str(training_config["dataset_id"]):
        raise ValueError("evaluation and DT-CORL reconstructor use different datasets")

    agent = _load_agent(base_payload, device)
    dataset = load_offline_transitions(evaluation_config, device)
    horizon = int(training_config["maximum_delay"])
    autoencoder, dynamics = build_official_reconstructor(
        state_dim=int(dataset.metadata["state_dim"]),
        action_dim=int(dataset.metadata["action_dim"]),
        maximum_delay=horizon,
        latent_dim=int(training_config["latent_dim"]),
        hidden_dim=int(training_config["transformer_hidden_dim"]),
        num_layers=int(training_config["num_layers"]),
        num_heads=int(training_config["num_heads"]),
        attention_dropout=float(training_config.get("attention_dropout", 0.1)),
        residual_dropout=float(training_config.get("residual_dropout", 0.1)),
        hidden_dropout=float(training_config.get("hidden_dropout", 0.1)),
        device=device,
    )
    autoencoder.load_state_dict(dtcorl_payload["autoencoder"])
    dynamics.load_state_dict(dtcorl_payload["dynamics"])
    autoencoder.eval()
    dynamics.eval()

    controller_arguments = {
        "agent": agent,
        "autoencoder": autoencoder,
        "dynamics": dynamics,
        "state_mean": dataset.state_mean,
        "state_std": dataset.state_std,
        "maximum_delay": horizon,
        "ack_lag": int(evaluation_config["ack_lag"]),
        "memory_steps": int(evaluation_config["posterior_memory_steps"]),
        "decay_time": float(evaluation_config["posterior_decay_time"]),
        "prior_strength": float(evaluation_config["posterior_prior_strength"]),
    }
    evaluations: dict[str, Any] = {}
    if progress_path.is_file():
        evaluations = json.loads(progress_path.read_text(encoding="utf-8")).get(
            "evaluations", {}
        )
    for scenario in evaluation_config["scenarios"]:
        scenario_name = str(scenario["name"])
        scenario_evaluations = evaluations.setdefault(scenario_name, {})
        paired_seed = int(evaluation_config["eval_seed"])
        common = {
            "dataset_id": dataset_id,
            "scenario": scenario,
            "episodes": int(evaluation_config["eval_episodes"]),
            "seed": paired_seed,
            "ack_lag": int(evaluation_config["ack_lag"]),
            "cvar_alpha": float(evaluation_config["cvar_alpha"]),
            "environment_config": evaluation_config.get("environment"),
        }
        if "direct" not in scenario_evaluations:
            scenario_evaluations["direct"] = evaluate_controller(
                controller_factory=lambda: DirectObservationController(
                    agent,
                    dataset.state_mean,
                    dataset.state_std,
                ),
                method="direct_observation_iql",
                **common,
            )
            _write_json(progress_path, {"evaluations": evaluations})
        if "dtcorl_ack_map" not in scenario_evaluations:
            scenario_evaluations["dtcorl_ack_map"] = evaluate_controller(
                controller_factory=lambda: OfficialDTCorlAckMapController(
                    **controller_arguments,
                    age_source="ack_map",
                ),
                method="official_dtcorl_transformer_ack_map_adapter",
                **common,
            )
            _write_json(progress_path, {"evaluations": evaluations})
        if "dtcorl_oracle_age" not in scenario_evaluations:
            scenario_evaluations["dtcorl_oracle_age"] = evaluate_controller(
                controller_factory=lambda: OfficialDTCorlAckMapController(
                    **controller_arguments,
                    age_source="oracle",
                ),
                method="official_dtcorl_transformer_oracle_age_upper_bound",
                privileged_selected_age=True,
                **common,
            )
            _write_json(progress_path, {"evaluations": evaluations})

    oracle_clean_equivalence = None
    clean = evaluations.get("clean_delay_0")
    if clean is not None:
        exact_fields = (
            "returns",
            "episode_lengths",
            "successes",
            "successes_at_end",
            "success_occupancies",
        )
        field_equality = {
            field: clean["direct"].get(field)
            == clean["dtcorl_oracle_age"].get(field)
            for field in exact_fields
        }
        oracle_clean_equivalence = {
            "field_exact_equality": field_equality,
            "all_required_fields_exactly_equal": all(field_equality.values()),
        }
        if not oracle_clean_equivalence["all_required_fields_exactly_equal"]:
            raise ValueError(
                "DT-CORL oracle path failed the clean exactness gate: "
                + json.dumps(field_equality, sort_keys=True)
            )

    verify_package_source_unchanged(local_source_provenance)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Official_DT_CORL_adapter_comparison",
        "source_provenance": official_source_provenance(),
        "local_source_provenance": local_source_provenance,
        "base_checkpoint": str(base_checkpoint_path),
        "dtcorl_checkpoint": str(dtcorl_checkpoint_path),
        "main_information_boundary": "delayed_observation_revealed_ACKs_actions_and_rewards",
        "oracle_is_upper_bound_not_main_method": True,
        "oracle_clean_equivalence": oracle_clean_equivalence,
        "evaluations": evaluations,
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dtcorl-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.config.resolve(),
        args.base_checkpoint.resolve(),
        args.dtcorl_checkpoint.resolve(),
        args.output.resolve(),
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
