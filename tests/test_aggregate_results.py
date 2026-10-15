import json
from pathlib import Path
import tempfile
import unittest

from dt_corl_delay.aggregate_formal_results import (
    _aggregate,
    _validate_task_completeness,
    _exact_sign_flip_pvalues,
    _holm_adjust,
    _load_rows,
    _method_sort_key,
    _validate_paired_episode_completeness,
    _validate_training_seed_completeness,
)


class FormalResultAggregationTest(unittest.TestCase):
    def test_formal_task_completeness_rejects_an_entire_missing_task(self) -> None:
        rows = [{"task": "task_a"}, {"task": "task_b"}]
        _validate_task_completeness(rows, {"task_a", "task_b"})
        with self.assertRaisesRegex(ValueError, "incomplete or unexpected tasks"):
            _validate_task_completeness(rows, {"task_a", "task_b", "task_c"})

    def test_privileged_oracle_sorts_after_deployable_dtcorl(self) -> None:
        methods = [
            "official_dtcorl_transformer_oracle_age_upper_bound",
            "official_dtcorl_transformer_ack_map_adapter",
            "cadr_iql",
            "direct_observation_iql",
        ]
        self.assertEqual(
            sorted(methods, key=_method_sort_key),
            [methods[3], methods[2], methods[1], methods[0]],
        )

    def test_formal_seed_completeness_is_checked_per_method_and_scenario(self) -> None:
        rows = [
            {
                "task": "task_a",
                "scenario": "clean",
                "method_key": method,
                "training_seed": seed,
            }
            for method in ("direct", "cadr")
            for seed in (101, 102)
        ]
        _validate_training_seed_completeness(rows, {101, 102})
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _validate_training_seed_completeness(rows, {101, 102})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            _validate_training_seed_completeness(rows, {101})

    def test_missing_paired_episode_is_rejected(self) -> None:
        rows = [
            {
                "task": "task_a",
                "scenario": "clean",
                "method_key": method,
                "training_seed": 101,
                "episode_seed": episode_seed,
            }
            for method in ("direct", "cadr")
            for episode_seed in (7001, 7002)
        ]
        _validate_paired_episode_completeness(rows)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "unpaired"):
            _validate_paired_episode_completeness(rows)

    def test_paired_delta_is_hierarchical_and_direction_is_explicit(self) -> None:
        rows = []
        for training_seed in (101, 102):
            for episode_index, episode_seed in enumerate((7001, 7002)):
                for method_key, value, success in (
                    ("direct", 1.0 + episode_index, 0.0),
                    ("cadr", 2.0 + episode_index, 1.0),
                ):
                    rows.append(
                        {
                            "task": "task_a",
                            "scenario": "fixed_delay_2",
                            "method_key": method_key,
                            "method": method_key,
                            "training_seed": training_seed,
                            "episode_seed": episode_seed,
                            "return": value,
                            "success": success,
                            "success_at_end": success,
                            "success_occupancy": success,
                            "post_change_success_occupancy": success,
                        }
                    )
        aggregate, paired = _aggregate(rows, bootstrap_seed=3, resamples=100)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["delta_definition"], "right_minus_left")
        self.assertEqual(paired[0]["left_method"], "direct")
        self.assertEqual(paired[0]["right_method"], "cadr")
        self.assertEqual(paired[0]["mean_return_delta"], 1.0)
        self.assertEqual(paired[0]["return_delta_ci95_low"], 1.0)
        self.assertEqual(paired[0]["return_delta_ci95_high"], 1.0)
        self.assertEqual(paired[0]["mean_success_delta"], 1.0)
        self.assertEqual(paired[0]["mean_success_at_end_delta"], 1.0)
        self.assertEqual(paired[0]["mean_success_occupancy_delta"], 1.0)
        self.assertEqual(
            paired[0]["mean_post_change_success_occupancy_delta"], 1.0
        )
        self.assertEqual(paired[0]["return_sign_flip_p_two_sided"], 0.5)
        self.assertEqual(paired[0]["return_sign_flip_p_greater"], 0.25)
        self.assertEqual(paired[0]["return_sign_flip_p_two_sided_holm"], 0.5)
        self.assertEqual(paired[0]["success_discordant_right_better"], 4)
        self.assertEqual(paired[0]["success_discordant_left_better"], 0)

    def test_method_sets_are_checked_within_task_not_across_tasks(self) -> None:
        rows = []
        for task, methods in (("task_a", ("direct", "cadr")), ("task_b", ("direct",))):
            for scenario in ("clean", "delay"):
                for method in methods:
                    rows.append(
                        {
                            "task": task,
                            "scenario": scenario,
                            "method_key": method,
                            "training_seed": 101,
                            "episode_seed": 7001,
                        }
                    )
        _validate_paired_episode_completeness(rows)
        rows = [
            row
            for row in rows
            if not (
                row["task"] == "task_a"
                and row["scenario"] == "delay"
                and row["method_key"] == "cadr"
            )
        ]
        with self.assertRaisesRegex(ValueError, "method_set"):
            _validate_paired_episode_completeness(rows)

    def test_declared_method_missing_from_entire_series_is_rejected(self) -> None:
        rows = [
            {
                "task": "task_a",
                "scenario": "clean",
                "method_key": "direct",
                "required_methods": ("direct", "cadr"),
                "training_seed": 101,
                "episode_seed": 7001,
            }
        ]
        with self.assertRaisesRegex(ValueError, "declared_method_set"):
            _validate_paired_episode_completeness(rows)

    def test_load_rows_filters_series_and_preserves_task_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for run_name, series, task in (
                ("keep", "formal_target", "PickCube-v1"),
                ("drop", "development", "StackCube-v1"),
            ):
                run = root / run_name
                run.mkdir()
                (run / "config.json").write_text(
                    json.dumps(
                        {
                            "experiment_series": series,
                            "seed": 101,
                            "dataset_id": f"dataset/{task}",
                            "environment": {"env_id": task},
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "summary.json").write_text(
                    json.dumps(
                        {
                            "evaluations": {
                                "clean": {
                                    "direct": {
                                        "method": "direct",
                                        "returns": [1.0],
                                        "episode_seeds": [7001],
                                    }
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            rows = _load_rows([root], "formal_target")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task"], "PickCube-v1")
            self.assertEqual(rows[0]["run_id"], "keep")

    def test_exact_sign_flip_and_holm_adjustment(self) -> None:
        self.assertEqual(_exact_sign_flip_pvalues([1.0, 1.0]), (0.5, 0.25))
        self.assertEqual(_holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
