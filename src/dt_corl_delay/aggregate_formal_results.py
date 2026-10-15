"""Aggregate formal run summaries with seed-aware confidence intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SECONDARY_SUCCESS_ENDPOINTS = (
    "success_at_end",
    "success_occupancy",
    "post_change_success_occupancy",
)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _method_results(value: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    if "returns" in value:
        yield str(value.get("method", "method")), value
        return
    for key, candidate in value.items():
        if isinstance(candidate, dict) and "returns" in candidate:
            yield str(candidate.get("method", key)), candidate


def _load_rows(run_roots: list[Path], series: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: list[Path] = []
    for root in run_roots:
        if root.is_file() and root.name == "summary.json":
            summaries.append(root)
        elif root.is_dir():
            summaries.extend(root.rglob("summary.json"))
    for summary_path in sorted(set(path.resolve() for path in summaries)):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluations = summary.get("evaluations")
        if not isinstance(evaluations, dict):
            continue
        config_path = summary_path.parent / "config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        if str(config.get("experiment_series", "")) != series:
            continue
        expected_scenarios = {
            str(scenario["name"])
            for scenario in config.get("scenarios", [])
        }
        if expected_scenarios and set(evaluations) != expected_scenarios:
            raise ValueError(
                f"summary/config scenario mismatch in {summary_path}: "
                f"missing={sorted(expected_scenarios - set(evaluations))}, "
                f"unexpected={sorted(set(evaluations) - expected_scenarios)}"
            )
        training_seed = int(config.get("seed", -1))
        environment = config.get("environment", {})
        task = str(
            environment.get("env_id")
            if isinstance(environment, dict) and environment.get("env_id")
            else config.get("dataset_id", "unknown_task")
        )
        required_methods = tuple(str(value) for value in config.get("required_methods", []))
        run_id = summary_path.parent.name
        for scenario, scenario_value in evaluations.items():
            if not isinstance(scenario_value, dict):
                continue
            for method_key, result in _method_results(scenario_value):
                returns = [float(value) for value in result["returns"]]
                episode_seeds = [int(value) for value in result["episode_seeds"]]
                successes = result.get("successes")
                secondary_success_arrays = {
                    "success_at_end": result.get("successes_at_end"),
                    "success_occupancy": result.get("success_occupancies"),
                    "post_change_success_occupancy": result.get(
                        "post_change_success_occupancies"
                    ),
                }
                if len(returns) != len(episode_seeds):
                    raise ValueError(f"return/seed length mismatch in {summary_path}")
                if successes is not None and len(successes) != len(returns):
                    raise ValueError(f"success/return length mismatch in {summary_path}")
                for endpoint, values in secondary_success_arrays.items():
                    if values is not None and len(values) != len(returns):
                        raise ValueError(
                            f"{endpoint}/return length mismatch in {summary_path}"
                        )
                for episode_index, (episode_seed, episode_return) in enumerate(
                    zip(episode_seeds, returns)
                ):
                    rows.append(
                        {
                            "series": series,
                            "run_id": run_id,
                            "summary_path": str(summary_path),
                            "training_seed": training_seed,
                            "task": task,
                            "required_methods": required_methods,
                            "scenario": str(scenario),
                            "method_key": method_key,
                            "method": str(result.get("method", method_key)),
                            "episode_index": episode_index,
                            "episode_seed": episode_seed,
                            "return": episode_return,
                            "inference_ms_p50": float(
                                result.get("inference_ms_p50", float("nan"))
                            ),
                            "inference_ms_p95": float(
                                result.get("inference_ms_p95", float("nan"))
                            ),
                            "mean_state_disagreement": result.get(
                                "mean_state_disagreement", ""
                            ),
                            "mean_effective_age": result.get(
                                "mean_effective_age_mean", ""
                            ),
                            "mean_posterior_delay": result.get(
                                "mean_posterior_delay_mean", ""
                            ),
                            "mean_posterior_entropy": result.get(
                                "mean_posterior_entropy", ""
                            ),
                            "diffusion_replans": result.get(
                                "diffusion_replans", ""
                            ),
                            "mean_executed_chunk_length": result.get(
                                "mean_executed_chunk_length", ""
                            ),
                            "success": (
                                "" if successes is None else float(successes[episode_index])
                            ),
                            **{
                                endpoint: (
                                    ""
                                    if values is None
                                    else float(values[episode_index])
                                )
                                for endpoint, values in secondary_success_arrays.items()
                            },
                        }
                    )
    deduplicated: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["task"]),
            int(row["training_seed"]),
            str(row["scenario"]),
            str(row["method_key"]),
            int(row["episode_seed"]),
        )
        previous = deduplicated.get(key)
        if previous is None:
            deduplicated[key] = row
            continue
        if (
            float(previous["return"]) != float(row["return"])
            or previous["success"] != row["success"]
            or any(
                previous.get(endpoint, "") != row.get(endpoint, "")
                for endpoint in SECONDARY_SUCCESS_ENDPOINTS
            )
        ):
            raise ValueError(f"conflicting duplicate evaluation row: {key}")
    rows = list(deduplicated.values())
    if not rows:
        raise ValueError("no evaluation episode rows found")
    return rows


def _hierarchical_bootstrap_ci(
    values_by_seed: dict[int, list[float]],
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float]:
    seeds = np.asarray(sorted(values_by_seed), dtype=np.int64)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_values: list[float] = []
        for seed in sampled_seeds:
            values = np.asarray(values_by_seed[int(seed)], dtype=np.float64)
            sampled_values.extend(
                rng.choice(values, size=len(values), replace=True).tolist()
            )
        estimates[index] = np.mean(sampled_values)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def _exact_sign_flip_pvalues(seed_deltas: list[float]) -> tuple[float, float]:
    """Exact cluster-level randomization p-values for a right-minus-left delta."""

    values = np.asarray(seed_deltas, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one seed delta is required")
    if values.size > 20:
        raise ValueError("exact sign-flip testing supports at most 20 seeds")
    observed = float(values.mean())
    permuted = np.asarray(
        [
            np.mean(values * np.asarray(signs, dtype=np.float64))
            for signs in itertools.product((-1.0, 1.0), repeat=values.size)
        ],
        dtype=np.float64,
    )
    tolerance = 1e-12
    two_sided = float(np.mean(np.abs(permuted) >= abs(observed) - tolerance))
    greater = float(np.mean(permuted >= observed - tolerance))
    return two_sided, greater


def _holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(np.asarray(pvalues, dtype=np.float64))
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running_maximum = 0.0
    count = len(pvalues)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[int(original_index)]))
        running_maximum = max(running_maximum, candidate)
        adjusted[int(original_index)] = running_maximum
    return adjusted.tolist()


def _method_sort_key(method: str) -> tuple[int, str]:
    lowered = method.lower()
    if "direct" in lowered:
        return (0, lowered)
    if "cadr" in lowered:
        return (1, lowered)
    if "oracle" in lowered:
        return (3, lowered)
    if "dtcorl" in lowered:
        return (2, lowered)
    return (4, lowered)


def _validate_training_seed_completeness(
    rows: list[dict[str, Any]],
    required_seeds: set[int],
) -> None:
    if not required_seeds:
        raise ValueError("at least one required training seed must be declared")
    groups: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        groups[(str(row["task"]), str(row["scenario"]), str(row["method_key"]))].add(
            int(row["training_seed"])
        )
    failures = {
        f"{task}::{scenario}::{method}": {
            "missing": sorted(required_seeds - observed),
            "unexpected": sorted(observed - required_seeds),
        }
        for (task, scenario, method), observed in groups.items()
        if observed != required_seeds
    }
    if failures:
        raise ValueError(
            "formal series has incomplete or unexpected training seeds: "
            + json.dumps(failures, sort_keys=True)
        )


def _validate_task_completeness(
    rows: list[dict[str, Any]],
    required_tasks: set[str],
) -> None:
    if not required_tasks:
        raise ValueError("at least one required task must be declared")
    observed = {str(row["task"]) for row in rows}
    if observed != required_tasks:
        raise ValueError(
            "formal series has incomplete or unexpected tasks: "
            + json.dumps(
                {
                    "missing": sorted(required_tasks - observed),
                    "unexpected": sorted(observed - required_tasks),
                },
                sort_keys=True,
            )
        )


def _validate_paired_episode_completeness(rows: list[dict[str, Any]]) -> None:
    scenarios: dict[tuple[str, str], dict[str, set[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        scenarios[(str(row["task"]), str(row["scenario"]))][
            str(row["method_key"])
        ].add(
            (int(row["training_seed"]), int(row["episode_seed"]))
        )
    failures: dict[str, dict[str, Any]] = {}
    methods_by_task: dict[str, set[str]] = defaultdict(set)
    declared_by_task: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        declared_by_task[str(row["task"])].update(row.get("required_methods", ()))
    for (task, _), methods in scenarios.items():
        methods_by_task[task].update(methods)
    for task, declared in declared_by_task.items():
        if declared:
            missing = sorted(declared - methods_by_task[task])
            unexpected = sorted(methods_by_task[task] - declared)
            if missing or unexpected:
                failures[f"{task}::__declared_method_set__"] = {
                    "missing_methods": missing,
                    "unexpected_methods": unexpected,
                }
    for (task, scenario), methods in scenarios.items():
        expected_methods = declared_by_task[task] or methods_by_task[task]
        missing_methods = sorted(expected_methods - set(methods))
        unexpected_methods = sorted(set(methods) - expected_methods)
        if missing_methods or unexpected_methods:
            failures[f"{task}::{scenario}::__method_set__"] = {
                "missing_methods": missing_methods,
                "unexpected_methods": unexpected_methods,
            }
        union = set().union(*methods.values())
        for method, observed in methods.items():
            if observed != union:
                failures[f"{task}::{scenario}::{method}"] = {
                    "missing_paired_episode_keys": [
                        list(key) for key in sorted(union - observed)
                    ]
                }
    if failures:
        raise ValueError(
            "formal series has unpaired or missing episodes: "
            + json.dumps(failures, sort_keys=True)
        )


def _aggregate(
    rows: list[dict[str, Any]],
    bootstrap_seed: int,
    resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(bootstrap_seed)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["scenario"], row["method_key"], row["method"])].append(row)
    aggregates: list[dict[str, Any]] = []
    for (task, scenario, method_key, method), group in sorted(groups.items()):
        returns_by_seed: dict[int, list[float]] = defaultdict(list)
        success_by_seed: dict[int, list[float]] = defaultdict(list)
        secondary_success_by_seed: dict[
            str, dict[int, list[float]]
        ] = {
            endpoint: defaultdict(list)
            for endpoint in SECONDARY_SUCCESS_ENDPOINTS
        }
        for row in group:
            returns_by_seed[int(row["training_seed"])].append(float(row["return"]))
            if row["success"] != "":
                success_by_seed[int(row["training_seed"])].append(float(row["success"]))
            for endpoint in SECONDARY_SUCCESS_ENDPOINTS:
                if row.get(endpoint, "") != "":
                    secondary_success_by_seed[endpoint][
                        int(row["training_seed"])
                    ].append(float(row[endpoint]))
        returns = np.asarray([row["return"] for row in group], dtype=np.float64)
        return_low, return_high = _hierarchical_bootstrap_ci(
            returns_by_seed,
            rng,
            resamples,
        )
        result: dict[str, Any] = {
            "task": task,
            "scenario": scenario,
            "method_key": method_key,
            "method": method,
            "training_seeds": len(returns_by_seed),
            "episodes": len(group),
            "mean_return": float(returns.mean()),
            "stdev_return": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
            "return_ci95_low": return_low,
            "return_ci95_high": return_high,
        }
        if success_by_seed:
            successes = np.asarray(
                [value for values in success_by_seed.values() for value in values],
                dtype=np.float64,
            )
            success_low, success_high = _hierarchical_bootstrap_ci(
                success_by_seed,
                rng,
                resamples,
            )
            result.update(
                {
                    "mean_success": float(successes.mean()),
                    "success_ci95_low": success_low,
                    "success_ci95_high": success_high,
                }
            )
        else:
            result.update(
                {
                    "mean_success": "",
                    "success_ci95_low": "",
                    "success_ci95_high": "",
                }
            )
        for endpoint in SECONDARY_SUCCESS_ENDPOINTS:
            values_by_seed = secondary_success_by_seed[endpoint]
            if values_by_seed:
                values = np.asarray(
                    [value for seed_values in values_by_seed.values() for value in seed_values],
                    dtype=np.float64,
                )
                endpoint_low, endpoint_high = _hierarchical_bootstrap_ci(
                    values_by_seed,
                    rng,
                    resamples,
                )
                result.update(
                    {
                        f"mean_{endpoint}": float(values.mean()),
                        f"{endpoint}_ci95_low": endpoint_low,
                        f"{endpoint}_ci95_high": endpoint_high,
                    }
                )
            else:
                result.update(
                    {
                        f"mean_{endpoint}": "",
                        f"{endpoint}_ci95_low": "",
                        f"{endpoint}_ci95_high": "",
                    }
                )
        aggregates.append(result)

    paired_rows: list[dict[str, Any]] = []
    by_scenario: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[(str(row["task"]), str(row["scenario"]))].append(row)
    for (task, scenario), scenario_rows in sorted(by_scenario.items()):
        methods = sorted(
            {str(row["method_key"]) for row in scenario_rows},
            key=_method_sort_key,
        )
        lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in scenario_rows:
            key = (
                str(row["method_key"]),
                int(row["training_seed"]),
                int(row["episode_seed"]),
            )
            if key in lookup:
                raise ValueError(
                    f"duplicate paired-result key: {task}, {scenario}, {key}"
                )
            lookup[key] = row
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1 :]:
                paired_return_by_seed: dict[int, list[float]] = defaultdict(list)
                paired_success_by_seed: dict[int, list[float]] = defaultdict(list)
                secondary_paired_by_seed: dict[
                    str, dict[int, list[float]]
                ] = {
                    endpoint: defaultdict(list)
                    for endpoint in SECONDARY_SUCCESS_ENDPOINTS
                }
                pair_keys = sorted(
                    (training_seed, episode_seed)
                    for method, training_seed, episode_seed in lookup
                    if method == left
                    and (right, training_seed, episode_seed) in lookup
                )
                for training_seed, episode_seed in pair_keys:
                    left_row = lookup[(left, training_seed, episode_seed)]
                    right_row = lookup[(right, training_seed, episode_seed)]
                    paired_return_by_seed[training_seed].append(
                        float(right_row["return"]) - float(left_row["return"])
                    )
                    if left_row["success"] != "" and right_row["success"] != "":
                        paired_success_by_seed[training_seed].append(
                            float(right_row["success"]) - float(left_row["success"])
                        )
                    for endpoint in SECONDARY_SUCCESS_ENDPOINTS:
                        if (
                            left_row.get(endpoint, "") != ""
                            and right_row.get(endpoint, "") != ""
                        ):
                            secondary_paired_by_seed[endpoint][training_seed].append(
                                float(right_row[endpoint]) - float(left_row[endpoint])
                            )
                if not paired_return_by_seed:
                    continue
                return_deltas = np.asarray(
                    [value for values in paired_return_by_seed.values() for value in values],
                    dtype=np.float64,
                )
                low, high = _hierarchical_bootstrap_ci(
                    paired_return_by_seed,
                    rng,
                    resamples,
                )
                seed_return_deltas = [
                    float(np.mean(paired_return_by_seed[seed]))
                    for seed in sorted(paired_return_by_seed)
                ]
                return_p_two_sided, return_p_greater = _exact_sign_flip_pvalues(
                    seed_return_deltas
                )
                return_seed_stdev = (
                    float(np.std(seed_return_deltas, ddof=1))
                    if len(seed_return_deltas) > 1
                    else 0.0
                )
                paired: dict[str, Any] = {
                    "task": task,
                    "scenario": scenario,
                    "left_method": left,
                    "right_method": right,
                    "delta_definition": "right_minus_left",
                    "training_seeds": len(paired_return_by_seed),
                    "paired_episodes": int(return_deltas.size),
                    "mean_return_delta": float(return_deltas.mean()),
                    "return_delta_ci95_low": low,
                    "return_delta_ci95_high": high,
                    "mean_seed_return_delta": float(np.mean(seed_return_deltas)),
                    "stdev_seed_return_delta": return_seed_stdev,
                    "paired_seed_return_effect_dz": (
                        float(np.mean(seed_return_deltas)) / return_seed_stdev
                        if return_seed_stdev > 0.0
                        else ""
                    ),
                    "return_sign_flip_p_two_sided": return_p_two_sided,
                    "return_sign_flip_p_greater": return_p_greater,
                }
                if paired_success_by_seed:
                    success_deltas = np.asarray(
                        [
                            value
                            for values in paired_success_by_seed.values()
                            for value in values
                        ],
                        dtype=np.float64,
                    )
                    success_low, success_high = _hierarchical_bootstrap_ci(
                        paired_success_by_seed,
                        rng,
                        resamples,
                    )
                    seed_success_deltas = [
                        float(np.mean(paired_success_by_seed[seed]))
                        for seed in sorted(paired_success_by_seed)
                    ]
                    success_p_two_sided, success_p_greater = _exact_sign_flip_pvalues(
                        seed_success_deltas
                    )
                    discordant_right_better = 0
                    discordant_left_better = 0
                    for training_seed, episode_seed in pair_keys:
                        left_success = lookup[(left, training_seed, episode_seed)]["success"]
                        right_success = lookup[(right, training_seed, episode_seed)]["success"]
                        if left_success == "" or right_success == "":
                            continue
                        if float(right_success) > float(left_success):
                            discordant_right_better += 1
                        elif float(right_success) < float(left_success):
                            discordant_left_better += 1
                    paired.update(
                        {
                            "mean_success_delta": float(success_deltas.mean()),
                            "success_delta_ci95_low": success_low,
                            "success_delta_ci95_high": success_high,
                            "success_sign_flip_p_two_sided": success_p_two_sided,
                            "success_sign_flip_p_greater": success_p_greater,
                            "success_discordant_right_better": discordant_right_better,
                            "success_discordant_left_better": discordant_left_better,
                            "success_matched_odds_ratio_haldane": (
                                (discordant_right_better + 0.5)
                                / (discordant_left_better + 0.5)
                            ),
                        }
                    )
                else:
                    paired.update(
                        {
                            "mean_success_delta": "",
                            "success_delta_ci95_low": "",
                            "success_delta_ci95_high": "",
                            "success_sign_flip_p_two_sided": "",
                            "success_sign_flip_p_greater": "",
                            "success_discordant_right_better": "",
                            "success_discordant_left_better": "",
                            "success_matched_odds_ratio_haldane": "",
                        }
                    )
                for endpoint in SECONDARY_SUCCESS_ENDPOINTS:
                    values_by_seed = secondary_paired_by_seed[endpoint]
                    if values_by_seed:
                        values = np.asarray(
                            [
                                value
                                for seed_values in values_by_seed.values()
                                for value in seed_values
                            ],
                            dtype=np.float64,
                        )
                        endpoint_low, endpoint_high = _hierarchical_bootstrap_ci(
                            values_by_seed,
                            rng,
                            resamples,
                        )
                        seed_deltas = [
                            float(np.mean(values_by_seed[seed]))
                            for seed in sorted(values_by_seed)
                        ]
                        endpoint_p_two_sided, endpoint_p_greater = (
                            _exact_sign_flip_pvalues(seed_deltas)
                        )
                        paired.update(
                            {
                                f"mean_{endpoint}_delta": float(values.mean()),
                                f"{endpoint}_delta_ci95_low": endpoint_low,
                                f"{endpoint}_delta_ci95_high": endpoint_high,
                                f"{endpoint}_sign_flip_p_two_sided": endpoint_p_two_sided,
                                f"{endpoint}_sign_flip_p_greater": endpoint_p_greater,
                            }
                        )
                    else:
                        paired.update(
                            {
                                f"mean_{endpoint}_delta": "",
                                f"{endpoint}_delta_ci95_low": "",
                                f"{endpoint}_delta_ci95_high": "",
                                f"{endpoint}_sign_flip_p_two_sided": "",
                                f"{endpoint}_sign_flip_p_greater": "",
                            }
                        )
                paired_rows.append(paired)

    # Multiplicity families are defined by method pair and outcome across all
    # scenarios in the requested experiment series.
    for outcome in ("return", "success", *SECONDARY_SUCCESS_ENDPOINTS):
        pvalue_key = f"{outcome}_sign_flip_p_two_sided"
        adjusted_key = f"{outcome}_sign_flip_p_two_sided_holm"
        families: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(paired_rows):
            if row.get(pvalue_key, "") != "":
                families[(row["left_method"], row["right_method"])].append(index)
        for indices in families.values():
            adjusted = _holm_adjust(
                [float(paired_rows[index][pvalue_key]) for index in indices]
            )
            for index, value in zip(indices, adjusted):
                paired_rows[index][adjusted_key] = value
        for row in paired_rows:
            row.setdefault(adjusted_key, "")
    return aggregates, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--required-training-seed",
        type=int,
        action="append",
        dest="required_training_seeds",
        help="repeat for every predeclared formal training seed; defaults to 101--105",
    )
    parser.add_argument(
        "--required-task",
        action="append",
        dest="required_tasks",
        required=True,
        help="repeat for every predeclared task in the aggregation series",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = _load_rows([path.resolve() for path in args.run_root], args.series)
    required_training_seeds = set(
        args.required_training_seeds or [101, 102, 103, 104, 105]
    )
    _validate_task_completeness(rows, set(args.required_tasks))
    _validate_training_seed_completeness(rows, required_training_seeds)
    _validate_paired_episode_completeness(rows)
    aggregates, paired = _aggregate(rows, args.bootstrap_seed, args.resamples)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "aggregate.csv", aggregates)
    _write_csv(output / "paired_deltas.csv", paired)
    _write_json(
        output / "summary.json",
        {
            "series": args.series,
            "bootstrap": {
                "method": "hierarchical_training_seed_then_episode_percentile",
                "seed": args.bootstrap_seed,
                "resamples": args.resamples,
            },
            "required_training_seeds": sorted(required_training_seeds),
            "aggregate": aggregates,
            "paired_deltas": paired,
        },
    )
    print(json.dumps({"rows": len(rows), "groups": len(aggregates)}, indent=2))


if __name__ == "__main__":
    main()
