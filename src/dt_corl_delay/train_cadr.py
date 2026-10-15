"""Train clean dynamics and evaluate the CADR deployment framework."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np
import torch

from .controllers import CadrController, DirectObservationController
from .cql import CQLAgent, CQLHyperparameters
from .delayed_eval import evaluate_controller
from .dynamics import DeltaDynamicsEnsemble
from .iql import IQLAgent, IQLHyperparameters
from .offline_data import OfflineTransitions, load_offline_transitions
from .provenance import (
    file_input_provenance,
    package_source_provenance,
    verify_package_source_unchanged,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _load_agent(payload: dict[str, Any], device: torch.device) -> Any:
    config = payload["config"]
    metadata = payload["dataset_metadata"]
    algorithm = str(payload.get("agent_kind", config.get("algorithm", "iql"))).lower()
    common_agent_arguments = {
        "state_dim": int(metadata["state_dim"]),
        "action_dim": int(metadata["action_dim"]),
        "hidden_dims": tuple(int(value) for value in config["hidden_dims"]),
        "learning_rate": float(config["learning_rate"]),
        "device": device,
    }
    if algorithm == "iql":
        agent = IQLAgent(
            **common_agent_arguments,
            hyperparameters=IQLHyperparameters(
                gamma=float(config["gamma"]),
                target_tau=float(config["target_tau"]),
                expectile=float(config["expectile"]),
                advantage_temperature=float(config["advantage_temperature"]),
                max_advantage_weight=float(config["max_advantage_weight"]),
            ),
        )
    elif algorithm == "cql":
        agent = CQLAgent(
            **common_agent_arguments,
            hyperparameters=CQLHyperparameters(
                gamma=float(config["gamma"]),
                target_tau=float(config["target_tau"]),
                cql_alpha=float(config["cql_alpha"]),
                cql_temperature=float(config.get("cql_temperature", 1.0)),
                cql_n_actions=int(config.get("cql_n_actions", 10)),
                bc_steps=int(config.get("bc_steps", 10_000)),
                initial_entropy_alpha=float(config.get("initial_entropy_alpha", 1.0)),
            ),
        )
    else:
        raise ValueError(f"unsupported checkpoint agent kind: {algorithm}")
    agent.load_checkpoint(payload["agent"])
    agent.policy.eval()
    agent.q.eval()
    agent.target_q.eval()
    if hasattr(agent, "value"):
        agent.value.eval()
    return agent


@torch.inference_mode()
def _validation_metrics(
    dynamics: DeltaDynamicsEnsemble,
    dataset: OfflineTransitions,
    generator: torch.Generator,
    batch_size: int,
    subset_indices: torch.Tensor,
) -> dict[str, float]:
    states, actions, _, next_states, _ = dataset.sample(
        batch_size,
        generator,
        subset_indices=subset_indices,
    )
    predictions = dynamics.predict_next_members(states, actions)
    ensemble_prediction = predictions.mean(dim=0)
    mse = (ensemble_prediction - next_states).pow(2).mean()
    disagreement = predictions.var(dim=0, correction=0).mean()
    return {
        "validation_mse": float(mse),
        "validation_disagreement": float(disagreement),
    }


@torch.inference_mode()
def _full_validation_metrics(
    dynamics: DeltaDynamicsEnsemble,
    dataset: Any,
    validation_indices: torch.Tensor,
    batch_size: int,
) -> dict[str, float | int | bool]:
    """Evaluate every held-out transition exactly once in stable order."""

    squared_error_sum = 0.0
    squared_error_count = 0
    disagreement_sum = 0.0
    disagreement_count = 0
    for start in range(0, int(validation_indices.numel()), batch_size):
        indices = validation_indices[start : start + batch_size]
        states = dataset.observations[indices]
        actions = dataset.actions[indices]
        next_states = dataset.next_observations[indices]
        predictions = dynamics.predict_next_members(states, actions)
        squared_errors = (predictions - next_states.unsqueeze(0)).pow(2)
        disagreements = predictions.var(dim=0, correction=0)
        squared_error_sum += float(squared_errors.sum())
        squared_error_count += squared_errors.numel()
        disagreement_sum += float(disagreements.sum())
        disagreement_count += disagreements.numel()
    return {
        "validation_mse": squared_error_sum / squared_error_count,
        "validation_disagreement": disagreement_sum / disagreement_count,
        "validation_transitions_evaluated": int(validation_indices.numel()),
        "deterministic_full_validation_split": True,
    }


@torch.inference_mode()
def _full_multistep_validation_metrics(
    dynamics: DeltaDynamicsEnsemble,
    dataset: Any,
    validation_episode_ids: torch.Tensor,
    horizon: int,
    batch_size: int,
) -> dict[str, float | int | bool | list[float]]:
    """Score clean open-loop rollouts over every valid held-out sequence."""

    if horizon <= 0:
        raise ValueError("validation rollout horizon must be positive")
    starts = torch.arange(
        dataset.size - horizon + 1,
        device=dataset.observations.device,
    )
    starts = starts[
        (dataset.episode_ids[starts] == dataset.episode_ids[starts + horizon - 1])
        & torch.isin(dataset.episode_ids[starts], validation_episode_ids)
    ]
    if starts.numel() == 0:
        raise ValueError("no complete held-out rollout sequences exist")
    mse_sums = [0.0] * horizon
    mse_counts = [0] * horizon
    disagreement_sums = [0.0] * horizon
    disagreement_counts = [0] * horizon
    for start in range(0, int(starts.numel()), batch_size):
        batch_starts = starts[start : start + batch_size]
        member_states = dataset.observations[batch_starts].unsqueeze(0).expand(
            dynamics.ensemble_size,
            -1,
            -1,
        ).clone()
        for offset in range(horizon):
            actions = dataset.actions[batch_starts + offset]
            next_member_states = []
            for member_index, member in enumerate(dynamics.members):
                inputs = torch.cat((member_states[member_index], actions), dim=-1)
                next_member_states.append(
                    (member_states[member_index] + member(inputs)).clamp(-20.0, 20.0)
                )
            member_states = torch.stack(next_member_states, dim=0)
            targets = dataset.next_observations[batch_starts + offset]
            squared_errors = (member_states - targets.unsqueeze(0)).pow(2)
            disagreements = member_states.var(dim=0, correction=0)
            mse_sums[offset] += float(squared_errors.sum())
            mse_counts[offset] += squared_errors.numel()
            disagreement_sums[offset] += float(disagreements.sum())
            disagreement_counts[offset] += disagreements.numel()
    horizon_mse = [
        total / count for total, count in zip(mse_sums, mse_counts, strict=True)
    ]
    horizon_disagreement = [
        total / count
        for total, count in zip(
            disagreement_sums,
            disagreement_counts,
            strict=True,
        )
    ]
    return {
        "deterministic_full_multistep_validation_split": True,
        "validation_sequences_evaluated": int(starts.numel()),
        "validation_rollout_horizon": horizon,
        "validation_multistep_mse": statistics.fmean(horizon_mse),
        "validation_final_horizon_mse": horizon_mse[-1],
        "validation_horizon_mse": horizon_mse,
        "validation_multistep_disagreement": statistics.fmean(
            horizon_disagreement
        ),
        "validation_horizon_disagreement": horizon_disagreement,
    }


def _save_dynamics_checkpoint(
    path: Path,
    step: int,
    config: dict[str, Any],
    dynamics: DeltaDynamicsEnsemble,
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "config": config,
        "dynamics": dynamics.checkpoint(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_generator": train_generator.get_state(),
            "validation_generator": validation_generator.get_state(),
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _last_logged_elapsed(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return float(json.loads(rows[-1])["elapsed_seconds"]) if rows else 0.0


def _restore_dynamics_rng(
    payload: dict[str, Any],
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> None:
    state = payload.get("rng_state")
    if state is None:
        raise ValueError("checkpoint predates exact-resume RNG state support")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])
    validation_generator.set_state(state["validation_generator"])


def run(
    config_path: Path,
    base_checkpoint_path: Path | None,
    output_dir: Path,
    resume: bool = False,
    evaluate: bool = True,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs_path = output_dir / "inputs.json"
    local_source_provenance = package_source_provenance()
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        saved_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("resume config differs from the saved run config")
        if (output_dir / "summary.json").exists():
            return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        requested_inputs = {
            "base_checkpoint": (
                file_input_provenance(base_checkpoint_path)
                if base_checkpoint_path is not None
                else None
            ),
            "evaluate": evaluate,
            "local_source_provenance": local_source_provenance,
        }
        if not inputs_path.is_file() or json.loads(
            inputs_path.read_text(encoding="utf-8")
        ) != requested_inputs:
            raise ValueError("resume checkpoint inputs do not match the output snapshot")
    else:
        requested_inputs = {
            "base_checkpoint": (
                file_input_provenance(base_checkpoint_path)
                if base_checkpoint_path is not None
                else None
            ),
            "evaluate": evaluate,
            "local_source_provenance": local_source_provenance,
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "config.json", config)
        _write_json(inputs_path, requested_inputs)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")

    if evaluate and base_checkpoint_path is None:
        raise ValueError("policy evaluation requires a base checkpoint")
    base_payload: dict[str, Any] | None = None
    agent: Any | None = None
    base_algorithm: str | None = None
    if base_checkpoint_path is not None:
        base_payload = torch.load(
            base_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if config["dataset_id"] != base_payload["dataset_metadata"]["dataset_id"]:
            raise ValueError("CADR and base policy checkpoints use different datasets")
        agent = _load_agent(base_payload, device)
        base_algorithm = str(
            base_payload.get(
                "agent_kind",
                base_payload["config"].get("algorithm", "iql"),
            )
        ).lower()
    load_started = time.perf_counter()
    dataset = load_offline_transitions(config, device)
    dataset_load_seconds = time.perf_counter() - load_started
    dynamics = DeltaDynamicsEnsemble(
        state_dim=int(dataset.metadata["state_dim"]),
        action_dim=int(dataset.metadata["action_dim"]),
        hidden_dims=tuple(int(value) for value in config["dynamics_hidden_dims"]),
        ensemble_size=int(config["ensemble_size"]),
        learning_rate=float(config["dynamics_learning_rate"]),
        device=device,
    )
    train_generator = torch.Generator(device=device).manual_seed(seed + 101)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 202)
    split_generator = torch.Generator(device=device).manual_seed(seed + 303)
    episode_ids = torch.unique(dataset.episode_ids, sorted=True)
    episode_permutation = episode_ids[
        torch.randperm(episode_ids.numel(), generator=split_generator, device=device)
    ]
    validation_episode_count = max(1, int(round(episode_ids.numel() * 0.1)))
    validation_episode_ids = episode_permutation[:validation_episode_count]
    training_episode_ids = episode_permutation[validation_episode_count:]
    if training_episode_ids.numel() == 0:
        raise ValueError("episode split left no training episodes")
    training_indices = torch.nonzero(
        torch.isin(dataset.episode_ids, training_episode_ids),
        as_tuple=False,
    ).squeeze(1)
    validation_indices = torch.nonzero(
        torch.isin(dataset.episode_ids, validation_episode_ids),
        as_tuple=False,
    ).squeeze(1)

    training_steps = int(config["dynamics_training_steps"])
    checkpoint_interval = int(config.get("checkpoint_interval", training_steps))
    log_interval = int(config["log_interval"])
    log_path = output_dir / "dynamics_training.jsonl"
    starting_step = 1
    previous_elapsed = 0.0
    latest_checkpoint = output_dir / "dynamics_checkpoint_latest.pt"
    final_checkpoint = output_dir / "dynamics_checkpoint.pt"
    if resume and (latest_checkpoint.exists() or final_checkpoint.exists()):
        resume_path = final_checkpoint if final_checkpoint.exists() else latest_checkpoint
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_payload["config"] != config:
            raise ValueError("dynamics checkpoint config differs from the requested config")
        dynamics.load_checkpoint(resume_payload["dynamics"], load_optimizer=True)
        _restore_dynamics_rng(resume_payload, train_generator, validation_generator)
        starting_step = int(resume_payload["step"]) + 1
        previous_elapsed = _last_logged_elapsed(log_path)
    training_started = time.perf_counter()
    log_mode = "a" if starting_step > 1 else "w"
    with log_path.open(log_mode, encoding="utf-8", newline="\n") as handle:
        for step in range(starting_step, training_steps + 1):
            states, actions, _, next_states, _ = dataset.sample(
                int(config["dynamics_batch_size"]),
                train_generator,
                subset_indices=training_indices,
            )
            metrics = dynamics.update(states, actions, next_states, train_generator)
            if step == 1 or step % log_interval == 0:
                validation = _validation_metrics(
                    dynamics,
                    dataset,
                    validation_generator,
                    int(config["validation_batch_size"]),
                    validation_indices,
                )
                row = {
                    "step": step,
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - training_started,
                    **{name: float(value) for name, value in metrics.items()},
                    **validation,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            if step % checkpoint_interval == 0 and step < training_steps:
                _save_dynamics_checkpoint(
                    output_dir / "dynamics_checkpoint_latest.pt",
                    step,
                    config,
                    dynamics,
                    train_generator,
                    validation_generator,
                )
    training_seconds = previous_elapsed + time.perf_counter() - training_started
    dynamics.eval()
    _save_dynamics_checkpoint(
        output_dir / "dynamics_checkpoint.pt",
        training_steps,
        config,
        dynamics,
        train_generator,
        validation_generator,
    )
    final_validation = _full_validation_metrics(
        dynamics,
        dataset,
        validation_indices,
        int(config["validation_batch_size"]),
    )
    final_validation.update(
        _full_multistep_validation_metrics(
            dynamics,
            dataset,
            validation_episode_ids,
            int(config["max_delay"]),
            int(config["validation_batch_size"]),
        )
    )

    evaluation_progress_path = output_dir / "evaluation_progress.json"
    evaluations: dict[str, dict[str, Any]] = {}
    if evaluate and evaluation_progress_path.is_file():
        evaluations = json.loads(
            evaluation_progress_path.read_text(encoding="utf-8")
        ).get("evaluations", {})
    if evaluate:
        assert agent is not None and base_algorithm is not None

    for scenario in config.get("scenarios", []) if evaluate else []:
        scenario_name = str(scenario["name"])
        if scenario_name in evaluations:
            continue
        paired_seed = int(config["eval_seed"])
        direct = evaluate_controller(
            controller_factory=lambda: DirectObservationController(
                agent,
                dataset.state_mean,
                dataset.state_std,
            ),
            dataset_id=config["dataset_id"],
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=paired_seed,
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method=f"direct_observation_{base_algorithm}",
            environment_config=config.get("environment"),
        )
        cadr = evaluate_controller(
            controller_factory=lambda: CadrController(
                agent=agent,
                dynamics=dynamics,
                state_mean=dataset.state_mean,
                state_std=dataset.state_std,
                max_delay=int(config["max_delay"]),
                ack_lag=int(config["ack_lag"]),
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
            dataset_id=config["dataset_id"],
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=paired_seed,
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method=f"cadr_{base_algorithm}",
            environment_config=config.get("environment"),
        )
        evaluations[scenario_name] = {"direct": direct, "cadr": cadr}
        _write_json(evaluation_progress_path, {"evaluations": evaluations})

    clean_equivalence = None
    clean = evaluations.get("clean_delay_0")
    if evaluate and clean is not None:
        exact_fields = (
            "returns",
            "episode_lengths",
            "successes",
            "successes_at_end",
            "success_occupancies",
        )
        field_equality = {
            field: clean["direct"].get(field) == clean["cadr"].get(field)
            for field in exact_fields
        }
        clean_equivalence = {
            "field_exact_equality": field_equality,
            "all_required_fields_exactly_equal": all(field_equality.values()),
        }
        if not clean_equivalence["all_required_fields_exactly_equal"]:
            raise ValueError(
                "CADR failed the preregistered clean no-regression gate: "
                + json.dumps(field_equality, sort_keys=True)
            )

    verify_package_source_unchanged(local_source_provenance)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": (
            f"CADR_with_frozen_{base_algorithm.upper()}"
            if evaluate and base_algorithm is not None
            else "clean_delta_dynamics_ensemble_for_CADR"
        ),
        "training_data_delay": "none_clean_only",
        "deployment_parameter_updates": False,
        "local_source_provenance": local_source_provenance,
        "information_boundary": "delayed_observation_and_revealed_acks_only",
        "paired_clean_equivalence": clean_equivalence,
        "base_checkpoint": str(base_checkpoint_path) if base_checkpoint_path else None,
        "dataset_load_seconds": dataset_load_seconds,
        "dynamics_training_seconds": training_seconds,
        "dynamics_updates_per_second": training_steps / training_seconds,
        "dynamics_validation": final_validation,
        "episode_split": {
            "method": "whole_episode_random_90_10",
            "training_episodes": int(training_episode_ids.numel()),
            "validation_episodes": int(validation_episode_ids.numel()),
            "training_transitions": int(training_indices.numel()),
            "validation_transitions": int(validation_indices.numel()),
        },
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
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dynamics-only", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.config.resolve(),
        args.base_checkpoint.resolve() if args.base_checkpoint else None,
        args.output.resolve(),
        resume=args.resume,
        evaluate=not args.dynamics_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
