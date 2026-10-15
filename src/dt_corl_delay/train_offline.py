"""Train a clean-data offline baseline and evaluate a frozen delayed policy."""

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

from .cql import CQLAgent, CQLHyperparameters
from .delayed_eval import evaluate_frozen_policy
from .iql import IQLAgent, IQLHyperparameters
from .offline_data import load_offline_transitions
from .provenance import (
    package_source_provenance,
    verify_package_source_unchanged,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _save_checkpoint(
    path: Path,
    step: int,
    config: dict[str, Any],
    dataset: Any,
    agent: Any,
    batch_generator: torch.Generator,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "config": config,
        "dataset_metadata": dataset.metadata,
        "state_mean": dataset.state_mean.detach().cpu(),
        "state_std": dataset.state_std.detach().cpu(),
        "agent_kind": str(config.get("algorithm", "iql")).lower(),
        "agent": agent.checkpoint(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "batch_generator": batch_generator.get_state(),
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


def _restore_rng_state(payload: dict[str, Any], batch_generator: torch.Generator) -> None:
    state = payload.get("rng_state")
    if state is None:
        raise ValueError("checkpoint predates exact-resume RNG state support")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    batch_generator.set_state(state["batch_generator"])


def run(config_path: Path, output_dir: Path, resume: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    local_source_provenance = package_source_provenance()
    inputs_path = output_dir / "inputs.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        saved_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("resume config differs from the saved run config")
        if (output_dir / "summary.json").exists():
            return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        requested_inputs = {"local_source_provenance": local_source_provenance}
        if not inputs_path.is_file() or json.loads(
            inputs_path.read_text(encoding="utf-8")
        ) != requested_inputs:
            raise ValueError("resume source inputs do not match the output snapshot")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "config.json", config)
        _write_json(
            inputs_path,
            {"local_source_provenance": local_source_provenance},
        )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")

    load_started = time.perf_counter()
    dataset = load_offline_transitions(config, device)
    dataset_load_seconds = time.perf_counter() - load_started
    _write_json(output_dir / "dataset_metadata.json", dataset.metadata)
    batch_generator = torch.Generator(device=device).manual_seed(seed + 1)
    algorithm = str(config.get("algorithm", "iql")).lower()
    common_agent_arguments = {
        "state_dim": int(dataset.metadata["state_dim"]),
        "action_dim": int(dataset.metadata["action_dim"]),
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
        raise ValueError(f"unsupported offline algorithm: {algorithm}")

    log_path = output_dir / "training.jsonl"
    starting_step = 1
    previous_elapsed = 0.0
    latest_checkpoint = output_dir / "checkpoint_latest.pt"
    final_checkpoint = output_dir / "checkpoint.pt"
    if resume and (latest_checkpoint.exists() or final_checkpoint.exists()):
        resume_path = final_checkpoint if final_checkpoint.exists() else latest_checkpoint
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_payload["config"] != config:
            raise ValueError("checkpoint config differs from the requested resume config")
        agent.load_checkpoint(resume_payload["agent"], load_optimizers=True)
        _restore_rng_state(resume_payload, batch_generator)
        starting_step = int(resume_payload["step"]) + 1
        previous_elapsed = _last_logged_elapsed(log_path)
    train_started = time.perf_counter()
    training_steps = int(config["training_steps"])
    checkpoint_interval = int(config.get("checkpoint_interval", training_steps))
    log_mode = "a" if starting_step > 1 else "w"
    with log_path.open(log_mode, encoding="utf-8", newline="\n") as log_handle:
        for step in range(starting_step, training_steps + 1):
            batch = dataset.sample(int(config["batch_size"]), batch_generator)
            metrics = agent.update(batch)
            if step == 1 or step % int(config["log_interval"]) == 0:
                row = {
                    "step": step,
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - train_started,
                    **{name: float(value) for name, value in metrics.items()},
                }
                log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                log_handle.flush()
            if step % checkpoint_interval == 0 and step < training_steps:
                _save_checkpoint(
                    output_dir / "checkpoint_latest.pt",
                    step,
                    config,
                    dataset,
                    agent,
                    batch_generator,
                )
    training_seconds = previous_elapsed + time.perf_counter() - train_started

    _save_checkpoint(
        output_dir / "checkpoint.pt",
        training_steps,
        config,
        dataset,
        agent,
        batch_generator,
    )

    evaluation_progress_path = output_dir / "evaluation_progress.json"
    evaluations: dict[str, Any] = {}
    if evaluation_progress_path.is_file():
        evaluations = json.loads(
            evaluation_progress_path.read_text(encoding="utf-8")
        ).get("evaluations", {})
    for scenario in config["scenarios"]:
        scenario_name = str(scenario["name"])
        if scenario_name in evaluations:
            continue
        result = evaluate_frozen_policy(
            agent=agent,
            state_mean=dataset.state_mean,
            state_std=dataset.state_std,
            dataset_id=config["dataset_id"],
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=int(config["eval_seed"]),
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method=f"direct_observation_{algorithm}",
            environment_config=config.get("environment"),
        )
        evaluations[scenario_name] = result
        _write_json(evaluation_progress_path, {"evaluations": evaluations})

    verify_package_source_unchanged(local_source_provenance)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm.upper(),
        "training_data_delay": "none_clean_only",
        "deployment_parameter_updates": False,
        "local_source_provenance": local_source_provenance,
        "dataset_load_seconds": dataset_load_seconds,
        "training_seconds": training_seconds,
        "updates_per_second": training_steps / training_seconds,
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.output.resolve(), resume=args.resume)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
