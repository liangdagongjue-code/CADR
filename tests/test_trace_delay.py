from __future__ import annotations

import random
import unittest

from dt_corl_delay.delay_process import DriftingCategoricalDelay, TraceDelay


class TraceDelayTest(unittest.TestCase):
    def test_drifting_delay_reaches_both_endpoint_laws(self) -> None:
        process = DriftingCategoricalDelay(
            delays=(0, 5),
            start_probabilities=(1.0, 0.0),
            end_probabilities=(0.0, 1.0),
            horizon_steps=10,
        )
        rng = random.Random(3)
        self.assertEqual(process.sample(0, rng), 0)
        self.assertEqual(process.sample(10, rng), 5)
        self.assertEqual(process.sample(100, rng), 5)

    def test_run_length_trace_replays_without_expansion(self) -> None:
        trace = TraceDelay([0, 3, 1], durations=[2, 3, 1], random_start=False)
        rng = random.Random(1)
        self.assertEqual([trace.sample(step, rng) for step in range(1, 8)], [0, 0, 3, 3, 3, 1, 0])

    def test_reset_reproduces_fixed_start(self) -> None:
        trace = TraceDelay([2, 4], random_start=False, cycle=False)
        rng = random.Random(2)
        self.assertEqual([trace.sample(step, rng) for step in range(1, 5)], [2, 4, 4, 4])
        trace.reset()
        self.assertEqual(trace.sample(1, rng), 2)


if __name__ == "__main__":
    unittest.main()
