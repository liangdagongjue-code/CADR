from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from dt_corl_delay.build_paper_artifacts import (
    ABLATION_VARIANTS,
    CADR_DP,
    DIRECT_DP,
    MAIN_REPORTS,
    ORACLE_DP,
    _oriented_primary_row,
    _primary_spec,
    build,
)


class PaperArtifactTest(unittest.TestCase):
    def test_primary_spec_is_domain_aware(self) -> None:
        self.assertEqual(_primary_spec("PickCube-v1")[2], "success")
        self.assertEqual(_primary_spec("D4RL/door/expert-v2")[2], "success")
        self.assertEqual(_primary_spec("mujoco/hopper/medium-v0")[2], "return")

    def test_primary_contrast_is_oriented_as_cadr_minus_direct(self) -> None:
        row = {
            "task": "mujoco/hopper/medium-v0",
            "scenario": "ood_changepoint",
            "left_method": "cadr_iql",
            "right_method": "direct_observation_iql",
            "mean_return_delta": -2.0,
            "return_delta_ci95_low": -3.0,
            "return_delta_ci95_high": -1.0,
            "training_seeds": 5,
            "paired_episodes": 100,
            "return_sign_flip_p_two_sided": 0.0625,
            "return_sign_flip_p_two_sided_holm": 0.125,
        }
        result = _oriented_primary_row(row)
        assert result is not None
        self.assertEqual(result["mean_delta"], 2.0)
        self.assertEqual(result["ci95_low"], 1.0)
        self.assertEqual(result["ci95_high"], 3.0)

    @staticmethod
    def _paired(
        task: str, outcome: str, scenario: str = "clean_delay_0"
    ) -> dict[str, object]:
        direct, cadr, _ = _primary_spec(task)
        prefix = "success" if outcome == "success" else "return"
        row: dict[str, object] = {
            "task": task,
            "scenario": scenario,
            "left_method": direct,
            "right_method": cadr,
            "training_seeds": 5,
            "paired_episodes": 5,
            f"mean_{prefix}_delta": 0.1,
            f"{prefix}_delta_ci95_low": 0.0,
            f"{prefix}_delta_ci95_high": 0.2,
            f"{prefix}_sign_flip_p_two_sided": 0.0625,
            f"{prefix}_sign_flip_p_two_sided_holm": 0.0625,
        }
        return row

    def test_end_to_end_builder_writes_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = root / "reports"
            reports.mkdir()
            maniskill_tasks = ("PickCube-v1", "StackCube-v1", "PegInsertionSide-v1")
            minari_tasks = (
                "mujoco/halfcheetah/medium-v0", "mujoco/hopper/medium-v0",
                "mujoco/walker2d/medium-v0", "D4RL/door/expert-v2",
                "D4RL/pen/expert-v2", "D4RL/hammer/expert-v2",
                "D4RL/relocate/expert-v2",
            )
            fixed = ("clean_delay_0", "fixed_delay_1", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8")
            aggregate = []
            for task in maniskill_tasks:
                for scenario in fixed:
                    for method, success in (
                        (DIRECT_DP, 0.5), (CADR_DP, 0.7), (ORACLE_DP, 0.8)
                    ):
                        aggregate.append(
                            {
                                "task": task,
                                "scenario": scenario,
                                "method_key": method,
                                "method": method,
                                "training_seeds": 5,
                                "episodes": 5,
                                "mean_return": success,
                                "stdev_return": 0.0,
                                "return_ci95_low": success,
                                "return_ci95_high": success,
                                "mean_success": success,
                                "success_ci95_low": success,
                                "success_ci95_high": success,
                            }
                        )
            summaries = {
                "formal_v1_maniskill_synthetic": {
                    "aggregate": aggregate,
                    "paired_deltas": [self._paired(task, "success") for task in maniskill_tasks],
                },
                "formal_v1_maniskill_real_network": {
                    "aggregate": [],
                    "paired_deltas": [
                        self._paired(task, "success", f"source{source}_{scale}")
                        for task in maniskill_tasks
                        for scale in ("native", "x2", "x4")
                        for source in range(5)
                    ],
                },
                "formal_v1_minari_synthetic": {
                    "aggregate": [],
                    "paired_deltas": [
                        self._paired(task, "success" if task.startswith("D4RL/") else "return")
                        for task in minari_tasks
                    ],
                },
                "formal_v1_minari_real_network": {
                    "aggregate": [],
                    "paired_deltas": [
                        self._paired(
                            task,
                            "success" if task.startswith("D4RL/") else "return",
                            f"source{source}_{scale}",
                        )
                        for task in minari_tasks
                        for scale in ("native", "x2", "x4")
                        for source in range(5)
                    ],
                },
            }
            for name in MAIN_REPORTS:
                report = reports / name
                report.mkdir()
                (report / "summary.json").write_text(
                    json.dumps({"series": name, **summaries[name]})
                )
            episode_fields = (
                "task", "scenario", "training_seed", "episode_seed", "method",
                "success", "inference_ms_p50", "inference_ms_p95",
            )
            for name in MAIN_REPORTS:
                main_episode_path = reports / name / "episodes.csv"
                with main_episode_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=episode_fields)
                    writer.writeheader()
                    if name == "formal_v1_maniskill_synthetic":
                        for seed in range(101, 106):
                            writer.writerow(
                                {
                                    "task": "PickCube-v1", "scenario": "clean_delay_0",
                                    "training_seed": seed, "episode_seed": seed,
                                    "method": CADR_DP, "success": 1.0,
                                    "inference_ms_p50": 2.0,
                                    "inference_ms_p95": 3.0,
                                }
                            )
            for variant in ABLATION_VARIANTS:
                report = reports / f"ablation_v1_maniskill_pickcube_{variant}"
                report.mkdir()
                with (report / "episodes.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=episode_fields)
                    writer.writeheader()
                    for seed in range(101, 106):
                        writer.writerow(
                            {
                                "task": "PickCube-v1", "scenario": "clean_delay_0",
                                "training_seed": seed, "episode_seed": seed,
                                "method": f"cadr_dp_ablation_{variant}_plus_official_maniskill_dp",
                                "success": 0.0,
                            }
                        )
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({"tree_sha256": "abc123"}))
            output = reports / "paper"
            summary = build(reports, output, freeze, resamples=100)
            self.assertEqual(len(summary["tasks"]), 10)
            for name in (
                "main_aggregate.csv", "primary_contrasts.csv",
                "ablation_contrasts.csv", "RESULTS.md",
                "trace_scale_summary.csv", "runtime_diagnostics.csv",
                "maniskill_fixed_delay_success.svg",
                "maniskill_fixed_delay_success.png", "summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
