"""Train and evaluate a pinned official DT-CORL reconstruction baseline."""

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
import torch.nn.functional as F

from .controllers import DirectObservationController
from .delayed_eval import evaluate_controller
from .offline_data import OfflineTransitions, load_offline_transitions
from .provenance import (
    file_input_provenance,
    package_source_provenance,
    verify_package_source_unchanged,
)
from .official_dtcorl import (
    OfficialDTCorlAckMapController,
    build_official_reconstructor,
    official_dynamic_parameter_count,
    official_source_provenance,
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


def _episode_split(
    dataset: OfflineTransitions,
    validation_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    episode_ids = torch.unique(dataset.episode_ids, sorted=True)
    permutation = episode_ids[
        torch.randperm(episode_ids.numel(), generator=generator, device=episode_ids.device)
    ]
    validation_count = max(1, int(round(episode_ids.numel() * validation_fraction)))
    training_episode_ids = permutation[validation_count:]
    validation_episode_ids = permutation[:validation_count]
    if training_episode_ids.numel() == 0:
        raise ValueError("episode split left no training episodes")
    return training_episode_ids, validation_episode_ids


def _valid_sequence_starts(
    dataset: OfflineTransitions,
    episode_ids: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    if horizon <= 0 or horizon > dataset.size:
        raise ValueError("invalid reconstruction horizon")
    starts = torch.arange(
        dataset.size - horizon + 1,
        device=dataset.observations.device,
    )
    same_episode = dataset.episode_ids[starts] == dataset.episode_ids[starts + horizon - 1]
    included_episode = torch.isin(dataset.episode_ids[starts], episode_ids)
    valid = starts[same_episode & included_episode]
    if valid.numel() == 0:
        raise ValueError("no complete reconstruction sequences exist for the split")
    return valid


def _sample_sequences(
    dataset: OfflineTransitions,
    valid_starts: torch.Tensor,
    horizon: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.randint(
        valid_starts.numel(),
        (batch_size,),
        generator=generator,
        device=valid_starts.device,
    )
    starts = valid_starts[positions]
    offsets = torch.arange(horizon, device=starts.device)
    indices = starts[:, None] + offsets[None, :]
    start_states = dataset.observations[starts].unsqueeze(1)
    actions = dataset.actions[indices]
    rewards = dataset.rewards[indices]
    targets = dataset.next_observations[indices]
    padding_mask = torch.zeros(
        (batch_size, horizon),
        dtype=torch.bool,
        device=starts.device,
    )
    return start_states, actions, rewards, targets, padding_mask


def _predict_sequences(
    autoencoder: torch.nn.Module,
    dynamics: torch.nn.Module,
    start_states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    horizon = actions.shape[1]
    timesteps = torch.arange(horizon, dtype=torch.long, device=actions.device)
    start_latents = autoencoder.encode(start_states)
    predicted_latents = dynamics(
        latents=start_latents,
        actions=actions,
        rewards=rewards,
        timesteps=timesteps,
        masks=padding_mask,
    )
    return autoencoder.decode(predicted_latents)


@torch.inference_mode()
def _validation_metrics(
    autoencoder: torch.nn.Module,
    dynamics: torch.nn.Module,
    dataset: OfflineTransitions,
    valid_starts: torch.Tensor,
    horizon: int,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    autoencoder_was_training = autoencoder.training
    dynamics_was_training = dynamics.training
    autoencoder.eval()
    dynamics.eval()
    start_states, actions, rewards, targets, padding_mask = _sample_sequences(
        dataset,
        valid_starts,
        horizon,
        batch_size,
        generator,
    )
    predictions = _predict_sequences(
        autoencoder,
        dynamics,
        start_states,
        actions,
        rewards,
        padding_mask,
    )
    horizon_mse = (predictions - targets).pow(2).mean(dim=(0, 2))
    if autoencoder_was_training:
        autoencoder.train()
    if dynamics_was_training:
        dynamics.train()
    return {
        "validation_mse": float(horizon_mse.mean()),
        "validation_rmse": float(torch.sqrt(horizon_mse.mean())),
        "validation_final_horizon_mse": float(horizon_mse[-1]),
        "validation_horizon_mse": horizon_mse.cpu().tolist(),
    }


def _save_checkpoint(
    path: Path,
    step: int,
    config: dict[str, Any],
    autoencoder: torch.nn.Module,
    dynamics: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    validation_generator: torch.Generator,
    local_source_provenance: dict[str, Any],
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "config": config,
        "autoencoder": autoencoder.state_dict(),
        "dynamics": dynamics.state_dict(),
        "optimizer": optimizer.state_dict(),
        "source_provenance": official_source_provenance(),
        "local_source_provenance": local_source_provenance,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "training_generator": training_generator.get_state(),
            "validation_generator": validation_generator.get_state(),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_checkpoint(
    payload: dict[str, Any],
    autoencoder: torch.nn.Module,
    dynamics: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> None:
    autoencoder.load_state_dict(payload["autoencoder"])
    dynamics.load_state_dict(payload["dynamics"])
    optimizer.load_state_dict(payload["optimizer"])
    state = payload["rng_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    training_generator.set_state(state["training_generator"])
    validation_generator.set_state(state["validation_generator"])


def _last_elapsed(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    return float(json.loads(rows[-1])["elapsed_seconds"]) if rows else 0.0


def run(
    config_path: Path,
    base_checkpoint_path: Path,
    output_dir: Path,
    resume: bool = False,
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
            "base_checkpoint": file_input_provenance(base_checkpoint_path),
            "official_dtcorl_source": official_source_provenance(),
            "local_source_provenance": local_source_provenance,
        }
        if not inputs_path.is_file() or json.loads(
            inputs_path.read_text(encoding="utf-8")
        ) != requested_inputs:
            raise ValueError("resume checkpoint/source inputs do not match the output snapshot")
    else:
        requested_inputs = {
            "base_checkpoint": file_input_provenance(base_checkpoint_path),
            "official_dtcorl_source": official_source_provenance(),
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

    base_payload = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    if str(config["dataset_id"]) != str(base_payload["dataset_metadata"]["dataset_id"]):
        raise ValueError("DT-CORL adapter and frozen policy use different datasets")
    agent = _load_agent(base_payload, device)
    dataset_load_started = time.perf_counter()
    dataset = load_offline_transitions(config, device)
    dataset_load_seconds = time.perf_counter() - dataset_load_started

    horizon = int(config["maximum_delay"])
    autoencoder, dynamics = build_official_reconstructor(
        state_dim=int(dataset.metadata["state_dim"]),
        action_dim=int(dataset.metadata["action_dim"]),
        maximum_delay=horizon,
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["transformer_hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        residual_dropout=float(config.get("residual_dropout", 0.1)),
        hidden_dropout=float(config.get("hidden_dropout", 0.1)),
        device=device,
    )
    optimizer = torch.optim.Adam(
        list(autoencoder.parameters()) + list(dynamics.parameters()),
        lr=float(config["dynamics_learning_rate"]),
    )

    split_generator = torch.Generator(device=device).manual_seed(seed + 401)
    training_generator = torch.Generator(device=device).manual_seed(seed + 402)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 403)
    training_episode_ids, validation_episode_ids = _episode_split(
        dataset,
        float(config.get("validation_fraction", 0.1)),
        split_generator,
    )
    training_starts = _valid_sequence_starts(dataset, training_episode_ids, horizon)
    validation_starts = _valid_sequence_starts(dataset, validation_episode_ids, horizon)

    training_steps = int(config["dynamics_training_steps"])
    log_interval = int(config["log_interval"])
    checkpoint_interval = int(config.get("checkpoint_interval", training_steps))
    log_path = output_dir / "dynamics_training.jsonl"
    latest_checkpoint = output_dir / "dtcorl_checkpoint_latest.pt"
    final_checkpoint = output_dir / "dtcorl_checkpoint.pt"
    starting_step = 1
    previous_elapsed = 0.0
    if resume and (latest_checkpoint.exists() or final_checkpoint.exists()):
        resume_path = final_checkpoint if final_checkpoint.exists() else latest_checkpoint
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload["config"] != config:
            raise ValueError("DT-CORL checkpoint config differs from the requested config")
        _restore_checkpoint(
            payload,
            autoencoder,
            dynamics,
            optimizer,
            training_generator,
            validation_generator,
        )
        starting_step = int(payload["step"]) + 1
        previous_elapsed = _last_elapsed(log_path)

    training_started = time.perf_counter()
    log_mode = "a" if starting_step > 1 else "w"
    autoencoder.train()
    dynamics.train()
    with log_path.open(log_mode, encoding="utf-8", newline="\n") as handle:
        for step in range(starting_step, training_steps + 1):
            batch = _sample_sequences(
                dataset,
                training_starts,
                horizon,
                int(config["dynamics_batch_size"]),
                training_generator,
            )
            predictions = _predict_sequences(
                autoencoder,
                dynamics,
                batch[0],
                batch[1],
                batch[2],
                batch[4],
            )
            loss = F.mse_loss(predictions, batch[3])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.get("gradient_clip_norm") is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(autoencoder.parameters()) + list(dynamics.parameters()),
                    float(config["gradient_clip_norm"]),
                )
            optimizer.step()
            if step == 1 or step % log_interval == 0:
                validation = _validation_metrics(
                    autoencoder,
                    dynamics,
                    dataset,
                    validation_starts,
                    horizon,
                    int(config["validation_batch_size"]),
                    validation_generator,
                )
                row = {
                    "step": step,
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - training_started,
                    "training_mse": float(loss),
                    **validation,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            if step % checkpoint_interval == 0 and step < training_steps:
                _save_checkpoint(
                    latest_checkpoint,
                    step,
                    config,
                    autoencoder,
                    dynamics,
                    optimizer,
                    training_generator,
                    validation_generator,
                    local_source_provenance,
                )
    training_seconds = previous_elapsed + time.perf_counter() - training_started
    autoencoder.eval()
    dynamics.eval()
    _save_checkpoint(
        final_checkpoint,
        training_steps,
        config,
        autoencoder,
        dynamics,
        optimizer,
        training_generator,
        validation_generator,
        local_source_provenance,
    )
    final_validation = _validation_metrics(
        autoencoder,
        dynamics,
        dataset,
        validation_starts,
        horizon,
        int(config["validation_batch_size"]),
        validation_generator,
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
        paired_seed = int(config["eval_seed"])
        direct = evaluate_controller(
            controller_factory=lambda: DirectObservationController(
                agent,
                dataset.state_mean,
                dataset.state_std,
            ),
            dataset_id=str(config["dataset_id"]),
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=paired_seed,
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method="direct_observation_iql",
            environment_config=config.get("environment"),
        )
        dtcorl = evaluate_controller(
            controller_factory=lambda: OfficialDTCorlAckMapController(
                agent=agent,
                autoencoder=autoencoder,
                dynamics=dynamics,
                state_mean=dataset.state_mean,
                state_std=dataset.state_std,
                maximum_delay=horizon,
                ack_lag=int(config["ack_lag"]),
                memory_steps=int(config["posterior_memory_steps"]),
                decay_time=float(config["posterior_decay_time"]),
                prior_strength=float(config["posterior_prior_strength"]),
            ),
            dataset_id=str(config["dataset_id"]),
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=paired_seed,
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method="official_dtcorl_transformer_ack_map_adapter",
            environment_config=config.get("environment"),
        )
        oracle = evaluate_controller(
            controller_factory=lambda: OfficialDTCorlAckMapController(
                agent=agent,
                autoencoder=autoencoder,
                dynamics=dynamics,
                state_mean=dataset.state_mean,
                state_std=dataset.state_std,
                maximum_delay=horizon,
                ack_lag=int(config["ack_lag"]),
                memory_steps=int(config["posterior_memory_steps"]),
                decay_time=float(config["posterior_decay_time"]),
                prior_strength=float(config["posterior_prior_strength"]),
                age_source="oracle",
            ),
            dataset_id=str(config["dataset_id"]),
            scenario=scenario,
            episodes=int(config["eval_episodes"]),
            seed=paired_seed,
            ack_lag=int(config["ack_lag"]),
            cvar_alpha=float(config["cvar_alpha"]),
            method="official_dtcorl_transformer_oracle_age_upper_bound",
            privileged_selected_age=True,
            environment_config=config.get("environment"),
        )
        evaluations[scenario_name] = {
            "direct": direct,
            "dtcorl_ack_map": dtcorl,
            "dtcorl_oracle_age": oracle,
        }
        _write_json(evaluation_progress_path, {"evaluations": evaluations})

    verify_package_source_unchanged(local_source_provenance)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Official_DT_CORL_reconstructor_with_ACK_MAP_adapter",
        "source_provenance": official_source_provenance(),
        "local_source_provenance": local_source_provenance,
        "adaptation": {
            "upstream_neural_architecture_modified": False,
            "dataset_api": "maintained_Minari_or_ManiSkill_adapter",
            "policy_backbone": "shared_frozen_offline_agent",
            "upstream_privileged_delay_replaced_by": "censored_ACK_MAP_age",
            "information_boundary": "delayed_observation_revealed_ACKs_actions_and_rewards",
        },
        "training_data_delay": "none_clean_only",
        "deployment_parameter_updates": False,
        "oracle_is_upper_bound_not_main_method": True,
        "base_checkpoint": str(base_checkpoint_path),
        "dataset_load_seconds": dataset_load_seconds,
        "dynamics_training_seconds": training_seconds,
        "dynamics_updates_per_second": training_steps / training_seconds,
        "dynamic_parameter_count": official_dynamic_parameter_count(autoencoder, dynamics),
        "dynamics_validation": final_validation,
        "episode_split": {
            "method": "whole_episode_random",
            "training_episodes": int(training_episode_ids.numel()),
            "validation_episodes": int(validation_episode_ids.numel()),
            "training_complete_sequences": int(training_starts.numel()),
            "validation_complete_sequences": int(validation_starts.numel()),
            "sequence_horizon": horizon,
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
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.config.resolve(),
        args.base_checkpoint.resolve(),
        args.output.resolve(),
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
