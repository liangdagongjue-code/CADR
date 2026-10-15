from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dt_corl_delay.experiment import run


class ExperimentSmokeTest(unittest.TestCase):
    def test_minimal_run_writes_expected_artifacts(self) -> None:
        config = {
            "seed": 7,
            "train_episodes": 2,
            "train_horizon": 8,
            "eval_episodes": 2,
            "eval_horizon": 10,
            "ack_lag": 1,
            "cvar_alpha": 0.5,
            "scenarios": [{"name": "fixed", "type": "fixed", "delay": 1}],
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output_path = root / "output"
            metrics = run(config_path, output_path)
            self.assertIn("fixed", metrics["scenarios"])
            self.assertTrue((output_path / "checkpoint.json").is_file())
            self.assertTrue((output_path / "metrics.json").is_file())
            self.assertTrue((output_path / "trajectory_fixed.svg").is_file())


if __name__ == "__main__":
    unittest.main()

