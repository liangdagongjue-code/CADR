from __future__ import annotations

import unittest

import numpy as np

from dt_corl_delay.channel import Ack
from dt_corl_delay.delay_belief import CensoredAckDelayEstimator


def _ack(generated_step: int, delay: int, ack_lag: int = 1) -> Ack:
    delivered = generated_step + delay
    return Ack(
        packet_id=generated_step,
        generated_step=generated_step,
        delivered_step=delivered,
        revealed_step=delivered + ack_lag,
        delay=delay,
    )


class CensoredAckDelayEstimatorTest(unittest.TestCase):
    def test_fixed_delay_converges_without_same_step_leakage(self) -> None:
        estimator = CensoredAckDelayEstimator(max_delay=6, prior_strength=0.2)
        pending: dict[int, list[Ack]] = {}
        for generated_step in range(1, 81):
            acknowledgement = _ack(generated_step, delay=3)
            pending.setdefault(acknowledgement.revealed_step, []).append(acknowledgement)
        for decision_step in range(81):
            estimator.update(decision_step, pending.get(decision_step, ()))
        self.assertGreater(estimator.probabilities[3], 0.95)
        self.assertGreater(estimator.effective_age_probabilities(80)[3], 0.90)

    def test_missing_ack_is_treated_as_right_censored(self) -> None:
        estimator = CensoredAckDelayEstimator(
            max_delay=5,
            prior_strength=0.1,
            prior_probabilities=np.ones(6),
        )
        estimator.update(0, ())
        estimator.update(1, ())
        estimator.update(2, ())
        # Packet g=1 has no ACK at t=2, so d=0 is excluded for that packet.
        self.assertLess(estimator.probabilities[0], 1.0 / 6.0)

    def test_current_age_distribution_for_deterministic_delay(self) -> None:
        estimator = CensoredAckDelayEstimator(max_delay=4)
        estimator._probabilities = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            estimator.effective_age_probabilities(10),
            np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            estimator.effective_age_probabilities(1),
            np.asarray([0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
