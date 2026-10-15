from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter

from dt_corl_delay.summarize_maniskill_dp_run import summarize


class ManiSkillSummaryTest(unittest.TestCase):
    def test_clean_selected_alias_is_bound_to_the_selected_step_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            checkpoints = run / "checkpoints"
            checkpoints.mkdir(parents=True)
            (run / "run_manifest.json").write_text("{}", encoding="utf-8")
            writer = SummaryWriter(str(run))
            for step, value in ((5, 0.5), (10, 0.4)):
                writer.add_scalar("eval/success_once", value, step)
                writer.add_scalar("eval/success_at_end", value, step)
                writer.add_scalar("eval/return", 10.0 * value, step)
            writer.close()
            selected = {"ema_agent": {"weight": torch.tensor([1.0])}}
            torch.save(selected, checkpoints / "5.pt")
            torch.save(selected, checkpoints / "best_eval_success_once.pt")

            result = summarize(run, run / "training_summary.json")
            selection = result["clean_checkpoint_selection"]
            self.assertEqual(selection["selected_event"]["step"], 5)
            self.assertIs(
                selection["selected_step_checkpoint"]["ema_state_exact_match"],
                True,
            )

            torch.save(
                {"ema_agent": {"weight": torch.tensor([2.0])}},
                checkpoints / "best_eval_success_once.pt",
            )
            with self.assertRaisesRegex(ValueError, "clean-selected step"):
                summarize(run, run / "training_summary.json")


if __name__ == "__main__":
    unittest.main()
