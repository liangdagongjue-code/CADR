from __future__ import annotations

import random
import unittest

from dt_corl_delay.channel import DelayedObservationChannel
from dt_corl_delay.delay_process import FixedDelay


class DelayedObservationChannelTest(unittest.TestCase):
    def test_delay_is_not_revealed_at_delivery_decision(self) -> None:
        channel = DelayedObservationChannel(FixedDelay(2), random.Random(1), ack_lag=1)
        channel.reset(10.0)
        channel.send(20.0, generated_step=1)

        self.assertEqual(channel.advance(0).policy_input.observation, 10.0)
        self.assertEqual(channel.advance(1).policy_input.observation, 10.0)
        delivered = channel.advance(3)
        self.assertEqual(delivered.policy_input.observation, 20.0)
        self.assertEqual(delivered.policy_input.acknowledgements, ())

        next_decision = channel.advance(4)
        self.assertEqual(len(next_decision.policy_input.acknowledgements), 1)
        self.assertEqual(next_decision.policy_input.acknowledgements[0].delay, 2)

    def test_out_of_order_packets_do_not_rewind_observation(self) -> None:
        class ScriptedDelay:
            def __init__(self) -> None:
                self.values = iter((5, 0))

            def reset(self) -> None:
                self.values = iter((5, 0))

            def sample(self, step: int, rng: random.Random) -> int:
                del step, rng
                return next(self.values)

        channel = DelayedObservationChannel(ScriptedDelay(), random.Random(2))
        channel.reset(0.0)
        channel.send(1.0, generated_step=1)
        channel.send(2.0, generated_step=2)
        self.assertEqual(channel.advance(2).policy_input.observation, 2.0)
        self.assertEqual(channel.advance(6).policy_input.observation, 2.0)


if __name__ == "__main__":
    unittest.main()

