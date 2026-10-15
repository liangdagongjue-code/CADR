"""Build publication tables and figures from complete frozen formal reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate_formal_results import (
    _exact_sign_flip_pvalues,
    _hierarchical_bootstrap_ci,
    _holm_adjust,
)


MAIN_REPORTS = (
    "formal_v1_maniskill_synthetic",
    "formal_v1_maniskill_real_network",
    "formal_v1_minari_synthetic",
    "formal_v1_minari_real_network",
)
ABLATION_VARIANTS = (
    "ack_map",
    "no_content",
    "fixed_prior",
    "single_model",
    "replan_every_step",
)
DIRECT_DP = "official_maniskill_dp_direct_delayed_observation"
CADR_DP = "cadr_posterior_state_reconstruction_plus_official_maniskill_dp"
ORACLE_DP = (
    "privileged_oracle_age_state_reconstruction_plus_"
    "official_maniskill_dp_upper_bound"
)
RUN_DIAGNOSTIC_FIELDS = (
    "inference_ms_p50",
    "inference_ms_p95",
    "mean_state_disagreement",
    "mean_effective_age",
    "mean_posterior_delay",
    "mean_posterior_entropy",
    "diffusion_replans",
    "mean_executed_chunk_length",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _is_maniskill(task: str) -> bool:
    return task in {"PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"}


def _is_adroit(task: str) -> bool:
    return task.startswith("D4RL/")


def _primary_spec(task: str) -> tuple[str, str, str]:
    if _is_maniskill(task):
        return DIRECT_DP, CADR_DP, "success"
    return "direct_observation_iql", "cadr_iql", "success" if _is_adroit(task) else "return"


def _oriented_primary_row(row: dict[str, Any]) -> dict[str, Any] | None:
    task = str(row["task"])
    direct, cadr, outcome = _primary_spec(task)
    left = str(row["left_method"])
    right = str(row["right_method"])
    if {left, right} != {direct, cadr}:
        return None
    sign = 1.0 if left == direct and right == cadr else -1.0
    prefix = "success" if outcome == "success" else "return"
    mean = row[f"mean_{prefix}_delta"]
    low = row[f"{prefix}_delta_ci95_low"]
    high = row[f"{prefix}_delta_ci95_high"]
    if mean == "" or low == "" or high == "":
        raise ValueError(f"primary endpoint is missing for {task}: {row['scenario']}")
    low_value, high_value = float(low), float(high)
    if sign < 0:
        low_value, high_value = -high_value, -low_value
    return {
        "task": task,
        "scenario": str(row["scenario"]),
        "endpoint": outcome,
        "contrast": "CADR_minus_direct",
        "mean_delta": sign * float(mean),
        "ci95_low": low_value,
        "ci95_high": high_value,
        "training_seeds": int(row["training_seeds"]),
        "paired_episodes": int(row["paired_episodes"]),
        "p_two_sided": float(row[f"{prefix}_sign_flip_p_two_sided"]),
        "p_two_sided_holm": float(row[f"{prefix}_sign_flip_p_two_sided_holm"]),
    }


def _load_episode_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _runtime_diagnostics(reports_root: Path) -> list[dict[str, Any]]:
    """Aggregate one run-level diagnostic per training seed without episode inflation."""

    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    seen: set[tuple[str, str, str, str, int]] = set()
    for report_name in MAIN_REPORTS:
        for row in _load_episode_csv(reports_root / report_name / "episodes.csv"):
            run_key = (
                report_name,
                row["task"],
                row["scenario"],
                row["method"],
                int(row["training_seed"]),
            )
            if run_key in seen:
                continue
            seen.add(run_key)
            for field in RUN_DIAGNOSTIC_FIELDS:
                value = row.get(field, "")
                if value in (None, ""):
                    continue
                numeric = float(value)
                if not np.isfinite(numeric):
                    continue
                grouped.setdefault(
                    (report_name, row["task"], row["scenario"], row["method"], field),
                    [],
                ).append(numeric)
    return [
        {
            "series": series,
            "task": task,
            "scenario": scenario,
            "method": method,
            "diagnostic": field,
            "training_seeds": len(values),
            "mean": float(np.mean(values)),
            "minimum": min(values),
            "maximum": max(values),
        }
        for (series, task, scenario, method, field), values in sorted(grouped.items())
    ]


def _trace_scale_summary(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in primary_rows:
        if not str(row["series"]).endswith("real_network"):
            continue
        scenario = str(row["scenario"])
        scale = next(
            (suffix.removeprefix("_") for suffix in ("_native", "_x2", "_x4") if scenario.endswith(suffix)),
            None,
        )
        if scale is None:
            raise ValueError(f"unrecognized trace scale: {scenario}")
        grouped.setdefault((str(row["task"]), str(row["endpoint"]), scale), []).append(
            float(row["mean_delta"])
        )
    result = []
    for (task, endpoint, scale), values in sorted(grouped.items()):
        if len(values) != 5:
            raise ValueError(
                f"trace scale must contain five source traces: {task}, {scale}, {len(values)}"
            )
        result.append(
            {
                "task": task,
                "endpoint": endpoint,
                "scale": scale,
                "source_traces": len(values),
                "mean_scenario_delta": float(np.mean(values)),
                "minimum_scenario_delta": min(values),
                "maximum_scenario_delta": max(values),
            }
        )
    if not result:
        raise ValueError("no trace-driven primary contrasts found")
    return result


def _ablation_contrasts(
    reports_root: Path,
    bootstrap_seed: int,
    resamples: int,
) -> list[dict[str, Any]]:
    main_rows = _load_episode_csv(
        reports_root / "formal_v1_maniskill_synthetic" / "episodes.csv"
    )
    main = {
        (
            row["task"], row["scenario"], int(row["training_seed"]),
            int(row["episode_seed"]),
        ): float(row["success"])
        for row in main_rows
        if row["task"] == "PickCube-v1" and row["method"] == CADR_DP
    }
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for variant in ABLATION_VARIANTS:
        report = reports_root / f"ablation_v1_maniskill_pickcube_{variant}"
        method = f"cadr_dp_ablation_{variant}_plus_official_maniskill_dp"
        variant_rows = _load_episode_csv(report / "episodes.csv")
        by_scenario: dict[str, dict[int, list[float]]] = {}
        observed_keys: set[tuple[str, str, int, int]] = set()
        for row in variant_rows:
            if row["task"] != "PickCube-v1" or row["method"] != method:
                continue
            key = (
                row["task"], row["scenario"], int(row["training_seed"]),
                int(row["episode_seed"]),
            )
            observed_keys.add(key)
            if key not in main:
                raise ValueError(f"ablation episode has no paired main CADR row: {key}")
            by_scenario.setdefault(row["scenario"], {}).setdefault(
                int(row["training_seed"]), []
            ).append(main[key] - float(row["success"]))
        expected_keys = {key for key in main if key[1] in by_scenario}
        if observed_keys != expected_keys:
            raise ValueError(
                f"main/ablation pairing mismatch for {variant}: "
                f"missing={len(expected_keys - observed_keys)}, "
                f"unexpected={len(observed_keys - expected_keys)}"
            )
        variant_results: list[dict[str, Any]] = []
        for scenario, values_by_seed in sorted(by_scenario.items()):
            values = [value for seed_values in values_by_seed.values() for value in seed_values]
            low, high = _hierarchical_bootstrap_ci(values_by_seed, rng, resamples)
            seed_deltas = [
                float(np.mean(values_by_seed[seed])) for seed in sorted(values_by_seed)
            ]
            p_two_sided, p_greater = _exact_sign_flip_pvalues(seed_deltas)
            variant_results.append(
                {
                    "variant": variant,
                    "task": "PickCube-v1",
                    "scenario": scenario,
                    "endpoint": "success",
                    "contrast": "full_CADR_minus_ablation",
                    "mean_delta": float(np.mean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "training_seeds": len(values_by_seed),
                    "paired_episodes": len(values),
                    "p_two_sided": p_two_sided,
                    "p_greater": p_greater,
                }
            )
        adjusted = _holm_adjust([row["p_two_sided"] for row in variant_results])
        for row, value in zip(variant_results, adjusted):
            row["p_two_sided_holm"] = value
        results.extend(variant_results)
    if not results:
        raise ValueError("no paired ablation contrasts found")
    return results


def _plot_maniskill_fixed_delay(
    aggregate_rows: list[dict[str, Any]], output: Path
) -> None:
    tasks = ("PickCube-v1", "StackCube-v1", "PegInsertionSide-v1")
    methods = (
        (DIRECT_DP, "Direct", "#7f7f7f"),
        (CADR_DP, "CADR-DP", "#0072b2"),
        (ORACLE_DP, "Oracle age", "#d55e00"),
    )
    delays = (0, 1, 2, 4, 8)
    lookup = {
        (str(row["task"]), str(row["scenario"]), str(row["method"])): row
        for row in aggregate_rows
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    for axis, task in zip(axes, tasks):
        for method, label, color in methods:
            values = [
                float(lookup[(task, f"fixed_delay_{delay}" if delay else "clean_delay_0", method)]["mean_success"])
                for delay in delays
            ]
            axis.plot(delays, values, marker="o", label=label, color=color)
        axis.set_title(task.replace("-v1", ""))
        axis.set_xlabel("Observation delay (control steps)")
        axis.set_xticks(delays)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Success-once probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def _plot_trace_heatmap(
    rows: list[dict[str, Any]],
    tasks: list[str],
    endpoint: str,
    output: Path,
) -> None:
    scales = ("native", "x2", "x4")
    lookup = {
        (str(row["task"]), str(row["scale"])): float(row["mean_scenario_delta"])
        for row in rows
        if row["endpoint"] == endpoint
    }
    values = np.asarray(
        [[lookup[(task, scale)] for scale in scales] for task in tasks],
        dtype=np.float64,
    )
    limit = max(1e-9, float(np.abs(values).max()))
    figure, axis = plt.subplots(figsize=(5.2, max(2.6, 0.52 * len(tasks))))
    image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(scales)), labels=scales)
    axis.set_yticks(range(len(tasks)), labels=[task.replace("/expert-v2", "") for task in tasks])
    axis.set_xlabel("Trace latency scale")
    axis.set_title(f"CADR - direct ({endpoint}) averaged over five traces")
    for row_index in range(len(tasks)):
        for column_index in range(len(scales)):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2g}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 * limit else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def _write_results_markdown(
    path: Path,
    primary_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    freeze_hash: str,
) -> None:
    selected = {
        "clean_delay_0", "markov_switching", "ood_changepoint",
        "ood_drifting", "drifting_to_long_delay", "ood_support_shift",
    }
    lines = [
        "# Frozen formal results",
        "",
        f"Source/protocol freeze: `{freeze_hash}`.",
        "",
        "Positive deltas favor CADR. Values are mean paired effects with 95% "
        "hierarchical bootstrap intervals.",
        "",
        "| Task | Scenario | Endpoint | CADR - direct | 95% CI | Holm p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in primary_rows:
        if row["scenario"] not in selected:
            continue
        lines.append(
            f"| {row['task']} | {row['scenario']} | {row['endpoint']} | "
            f"{row['mean_delta']:.4g} | [{row['ci95_low']:.4g}, "
            f"{row['ci95_high']:.4g}] | {row['p_two_sided_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Trace-driven WiFi/5G summary",
            "",
            "Each row is a descriptive average of the five predeclared source traces; "
            "scenario-level paired intervals remain in `primary_contrasts.csv`.",
            "",
            "| Task | Endpoint | Scale | Mean CADR - direct | Range across traces |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in trace_rows:
        lines.append(
            f"| {row['task']} | {row['endpoint']} | {row['scale']} | "
            f"{row['mean_scenario_delta']:.4g} | [{row['minimum_scenario_delta']:.4g}, "
            f"{row['maximum_scenario_delta']:.4g}] |"
        )
    lines.extend(
        [
            "",
            "## PickCube component ablations",
            "",
            "Positive deltas favor the complete CADR-DP method over the named ablation.",
            "",
            "| Variant | Scenario | Full CADR - ablation | 95% CI | Holm p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in ablation_rows:
        if row["scenario"] not in selected:
            continue
        lines.append(
            f"| {row['variant']} | {row['scenario']} | {row['mean_delta']:.4g} | "
            f"[{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] | "
            f"{row['p_two_sided_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Clean-channel inference latency",
            "",
            "Values summarize the five training-seed run-level latency statistics; "
            "they are descriptive system measurements, not episode replicates.",
            "",
            "| Task | Method | Mean p50 ms | Mean p95 ms |",
            "|---|---|---:|---:|",
        ]
    )
    runtime_lookup = {
        (row["task"], row["scenario"], row["method"], row["diagnostic"]): row
        for row in runtime_rows
    }
    tasks = sorted({row["task"] for row in primary_rows})
    for task in tasks:
        direct, cadr, _ = _primary_spec(task)
        for method, label in ((direct, "Direct"), (cadr, "CADR")):
            p50 = runtime_lookup.get(
                (task, "clean_delay_0", method, "inference_ms_p50")
            )
            p95 = runtime_lookup.get(
                (task, "clean_delay_0", method, "inference_ms_p95")
            )
            if p50 is None or p95 is None:
                continue
            lines.append(
                f"| {task} | {label} | {p50['mean']:.4g} | {p95['mean']:.4g} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build(
    reports_root: Path,
    output: Path,
    freeze_manifest: Path,
    bootstrap_seed: int = 20260823,
    resamples: int = 10_000,
) -> dict[str, Any]:
    reports_root = reports_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = _read_json(freeze_manifest.resolve())
    aggregate_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for report_name in MAIN_REPORTS:
        summary = _read_json(reports_root / report_name / "summary.json")
        if summary["series"] != report_name:
            raise ValueError(f"formal report series mismatch: {report_name}")
        for row in summary["aggregate"]:
            aggregate_rows.append({"series": report_name, **row})
        for row in summary["paired_deltas"]:
            primary = _oriented_primary_row(row)
            if primary is not None:
                primary_rows.append({"series": report_name, **primary})
    expected_tasks = {
        "PickCube-v1", "StackCube-v1", "PegInsertionSide-v1",
        "mujoco/halfcheetah/medium-v0", "mujoco/hopper/medium-v0",
        "mujoco/walker2d/medium-v0", "D4RL/door/expert-v2",
        "D4RL/pen/expert-v2", "D4RL/hammer/expert-v2",
        "D4RL/relocate/expert-v2",
    }
    if {row["task"] for row in primary_rows} != expected_tasks:
        raise ValueError("primary contrast report does not cover all ten frozen tasks")
    ablation_rows = _ablation_contrasts(reports_root, bootstrap_seed, resamples)
    trace_rows = _trace_scale_summary(primary_rows)
    runtime_rows = _runtime_diagnostics(reports_root)
    if not runtime_rows:
        raise ValueError("formal reports contain no runtime or reconstruction diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "main_aggregate.csv", aggregate_rows)
    _write_csv(output / "primary_contrasts.csv", primary_rows)
    _write_csv(output / "ablation_contrasts.csv", ablation_rows)
    _write_csv(output / "trace_scale_summary.csv", trace_rows)
    _write_csv(output / "runtime_diagnostics.csv", runtime_rows)
    _plot_maniskill_fixed_delay(
        [row for row in aggregate_rows if row["series"] == "formal_v1_maniskill_synthetic"],
        output / "maniskill_fixed_delay_success",
    )
    _plot_trace_heatmap(
        trace_rows,
        ["PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"],
        "success",
        output / "maniskill_trace_success_delta",
    )
    _plot_trace_heatmap(
        trace_rows,
        [
            "mujoco/halfcheetah/medium-v0", "mujoco/hopper/medium-v0",
            "mujoco/walker2d/medium-v0",
        ],
        "return",
        output / "mujoco_trace_return_delta",
    )
    _plot_trace_heatmap(
        trace_rows,
        [
            "D4RL/door/expert-v2", "D4RL/pen/expert-v2",
            "D4RL/hammer/expert-v2", "D4RL/relocate/expert-v2",
        ],
        "success",
        output / "adroit_trace_success_delta",
    )
    _write_results_markdown(
        output / "RESULTS.md",
        primary_rows,
        ablation_rows,
        trace_rows,
        runtime_rows,
        str(freeze["tree_sha256"]),
    )
    result = {
        "source_freeze_tree_sha256": freeze["tree_sha256"],
        "main_reports": list(MAIN_REPORTS),
        "tasks": sorted(expected_tasks),
        "primary_contrasts": len(primary_rows),
        "ablation_contrasts": len(ablation_rows),
        "trace_scale_summaries": len(trace_rows),
        "runtime_diagnostics": len(runtime_rows),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": resamples,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    _write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.reports_root,
                args.output,
                args.freeze_manifest,
                args.bootstrap_seed,
                args.resamples,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
