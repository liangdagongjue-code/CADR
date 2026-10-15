from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dt_corl_delay.evaluate_checkpoints import run as run_checkpoint_evaluation
from dt_corl_delay.evaluate_dtcorl_checkpoint import run as run_dtcorl_evaluation
from dt_corl_delay.provenance import file_input_provenance


class CompletedEvaluationResumeTest(unittest.TestCase):
    def _completed_output(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        config = {"seed": 101}
        config_path = root / "requested.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = root / "run"
        output.mkdir()
        (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
        summary: dict[str, object] = {"complete": True, "evaluations": {}}
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return config_path, output, summary

    def test_checkpoint_comparison_resume_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, output, expected = self._completed_output(root)
            actual = run_checkpoint_evaluation(
                config_path,
                root / "unused-policy.pt",
                root / "unused-dynamics.pt",
                output,
                resume=True,
            )
            self.assertEqual(actual, expected)

    def test_dtcorl_comparison_resume_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, output, expected = self._completed_output(root)
            actual = run_dtcorl_evaluation(
                config_path,
                root / "unused-policy.pt",
                root / "unused-dtcorl.pt",
                output,
                resume=True,
            )
            self.assertEqual(actual, expected)

    def test_partial_resume_rejects_mutated_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = {"seed": 101}
            config_path = root / "requested.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            base_checkpoint = root / "policy.pt"
            dynamics_checkpoint = root / "dynamics.pt"
            base_checkpoint.write_bytes(b"original-policy")
            dynamics_checkpoint.write_bytes(b"original-dynamics")
            output = root / "run"
            output.mkdir()
            (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (output / "inputs.json").write_text(
                json.dumps(
                    {
                        "base_checkpoint": file_input_provenance(base_checkpoint),
                        "dynamics_checkpoint": file_input_provenance(dynamics_checkpoint),
                        "include_cadr": True,
                    }
                ),
                encoding="utf-8",
            )

            base_checkpoint.write_bytes(b"replacement-policy")
            with self.assertRaisesRegex(ValueError, "resume checkpoint inputs"):
                run_checkpoint_evaluation(
                    config_path,
                    base_checkpoint,
                    dynamics_checkpoint,
                    output,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
