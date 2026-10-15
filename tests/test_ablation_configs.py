from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dt_corl_delay.prepare_cadr_ablation_config import materialize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CadrAblationConfigurationTest(unittest.TestCase):
    def test_variants_change_only_declared_controls(self) -> None:
        template = (
            PROJECT_ROOT
            / "configs"
            / "maniskill_pickcube_official_dp_formal.json"
        )
        source = json.loads(template.read_text(encoding="utf-8"))
        expectations = {
            "ack_map": {
                "age_decision_rule": "map",
                "use_content_conditioning": False,
            },
            "no_content": {
                "age_decision_rule": "posterior_mean",
                "use_content_conditioning": False,
            },
            "fixed_prior": {"posterior_updates": False},
            "single_model": {"ensemble_size": 1},
            "replan_every_step": {"cadr_act_horizon": 1},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for variant, changed in expectations.items():
                output = root / f"{variant}.json"
                config = materialize(template, output, variant)
                self.assertEqual(config["ablation_variant"], variant)
                self.assertFalse(config["include_oracle_upper_bound"])
                self.assertEqual(
                    config["required_methods"],
                    [
                        "official_maniskill_dp_direct_delayed_observation",
                        f"cadr_dp_ablation_{variant}_plus_official_maniskill_dp",
                    ],
                )
                self.assertEqual(config["scenarios"], source["scenarios"])
                self.assertEqual(config["seed"], source["seed"])
                for key, value in changed.items():
                    self.assertEqual(config[key], value)
                self.assertEqual(materialize(template, output, variant), config)

    def test_replan_ablation_rejects_non_chunked_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            template = root / "template.json"
            template.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                materialize(template, root / "output.json", "replan_every_step")


if __name__ == "__main__":
    unittest.main()
