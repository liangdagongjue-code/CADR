import unittest
from types import SimpleNamespace

from dt_corl_delay.delayed_eval import (
    _environment_horizon,
    _episode_success_summary,
)


class EnvironmentHorizonTest(unittest.TestCase):
    def test_success_summary_separates_once_end_and_post_change(self) -> None:
        summary = _episode_success_summary(
            [(0, False), (1, True), (2, False), (3, True)],
            {"type": "changepoint", "change_step": 2},
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["once"], 1.0)
        self.assertEqual(summary["at_end"], 1.0)
        self.assertEqual(summary["occupancy"], 0.5)
        self.assertEqual(summary["post_change_occupancy"], 0.5)

    def test_explicit_horizon_survives_wrapper_spec_without_limit(self) -> None:
        environment = SimpleNamespace(
            spec=SimpleNamespace(max_episode_steps=None),
            unwrapped=SimpleNamespace(spec=SimpleNamespace(max_episode_steps=None)),
        )
        self.assertEqual(
            _environment_horizon(environment, {"max_episode_steps": 50}),
            50,
        )

    def test_unwrapped_spec_is_a_valid_fallback(self) -> None:
        environment = SimpleNamespace(
            spec=SimpleNamespace(max_episode_steps=None),
            unwrapped=SimpleNamespace(spec=SimpleNamespace(max_episode_steps=1000)),
        )
        self.assertEqual(_environment_horizon(environment, None), 1000)


if __name__ == "__main__":
    unittest.main()
