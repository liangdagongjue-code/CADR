import json
from pathlib import Path
import unittest
import random
import tempfile

import numpy as np
import torch

from dt_corl_delay.channel import DelayedObservationChannel, PolicyInput
from dt_corl_delay.delay_process import FixedDelay
from dt_corl_delay.evaluate_maniskill_cadr_diffusion import run as run_cadr_dp_evaluation
from dt_corl_delay.maniskill_diffusion import (
    CadrDiffusionPolicyController,
    OfficialDiffusionPolicyController,
)


class _FakeDiffusionAgent:
    def __init__(self) -> None:
        self.observation_sequences: list[np.ndarray] = []

    def get_action(self, observations: torch.Tensor) -> torch.Tensor:
        self.observation_sequences.append(observations.cpu().numpy().copy())
        actions = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        return actions


class _ObservationConditionedFakeAgent:
    def get_action(self, observations: torch.Tensor) -> torch.Tensor:
        current = observations[:, -1:, :1]
        return current.repeat(1, 8, 1).clamp(-1.0, 1.0)


class _DynamicsMustNotRunOnCleanDelay:
    state_dim = 1
    action_dim = 1
    ensemble_size = 2

    def rollout_age_candidates(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("zero-delay reconstruction should use the exact observation")


class DiffusionPolicyControllerTest(unittest.TestCase):
    def test_completed_evaluation_resume_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = {"seed": 0}
            config_path = root / "requested.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "run"
            output.mkdir()
            (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
            expected = {"complete": True, "evaluations": {}}
            (output / "summary.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )

            actual = run_cadr_dp_evaluation(
                config_path=config_path,
                policy_checkpoint_path=root / "not-needed-policy.pt",
                dynamics_checkpoint_path=root / "not-needed-dynamics.pt",
                source_root=root / "not-needed-source",
                output_dir=output,
                resume=True,
            )

            self.assertEqual(actual, expected)

    def test_executes_chunks_while_updating_received_observation_history(self) -> None:
        agent = _FakeDiffusionAgent()
        controller = OfficialDiffusionPolicyController(
            agent=agent,
            device=torch.device("cpu"),
            obs_horizon=2,
            act_horizon=8,
        )
        controller.set_episode_seed(123)
        controller.reset(np.asarray([0.0], dtype=np.float32))

        actions = []
        for step in range(9):
            actions.append(
                float(
                    controller.act(
                        PolicyInput(
                            observation=np.asarray([float(step)], dtype=np.float32),
                            acknowledgements=(),
                        ),
                        decision_step=step,
                    )[0]
                )
            )

        self.assertEqual(actions[:8], [float(value) for value in range(8)])
        self.assertEqual(actions[8], 0.0)
        self.assertEqual(len(agent.observation_sequences), 2)
        np.testing.assert_array_equal(
            agent.observation_sequences[0],
            np.asarray([[[0.0], [0.0]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            agent.observation_sequences[1],
            np.asarray([[[7.0], [8.0]]], dtype=np.float32),
        )

    def test_cadr_dp_is_action_equivalent_to_direct_dp_on_clean_channel(self) -> None:
        agent = _ObservationConditionedFakeAgent()
        direct = OfficialDiffusionPolicyController(
            agent=agent,
            device=torch.device("cpu"),
        )
        cadr = CadrDiffusionPolicyController(
            agent=agent,
            dynamics=_DynamicsMustNotRunOnCleanDelay(),
            state_mean=torch.zeros((1, 1)),
            state_std=torch.ones((1, 1)),
            device=torch.device("cpu"),
            max_delay=8,
            ack_lag=1,
            risk_blend=8.0,
            risk_threshold=0.1,
        )
        direct.set_episode_seed(456)
        cadr.set_episode_seed(456)
        initial = np.asarray([0.0], dtype=np.float32)
        direct.reset(initial)
        cadr.reset(initial)
        channel = DelayedObservationChannel(
            FixedDelay(0),
            random.Random(789),
            ack_lag=1,
        )
        channel.reset(initial)

        for step in range(20):
            policy_input = channel.advance(step).policy_input
            direct_action = direct.act(policy_input, step)
            cadr_action = cadr.act(policy_input, step)
            np.testing.assert_array_equal(direct_action, cadr_action)
            next_observation = np.asarray([(step + 1) / 100.0], dtype=np.float32)
            channel.send(next_observation, generated_step=step + 1)
        self.assertEqual(
            cadr.diagnostics()["mean_reconstruction_reliability"],
            1.0,
        )

    def test_oracle_age_is_explicitly_privileged_and_required_each_step(self) -> None:
        common = {
            "agent": _ObservationConditionedFakeAgent(),
            "dynamics": _DynamicsMustNotRunOnCleanDelay(),
            "state_mean": torch.zeros((1, 1)),
            "state_std": torch.ones((1, 1)),
            "device": torch.device("cpu"),
            "max_delay": 8,
            "ack_lag": 1,
        }
        posterior = CadrDiffusionPolicyController(**common)
        with self.assertRaises(RuntimeError):
            posterior.set_oracle_age(0)

        oracle = CadrDiffusionPolicyController(**common, age_source="oracle")
        oracle.reset(np.asarray([0.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            oracle.set_oracle_age(9)
        oracle.set_oracle_age(0)
        oracle.act(
            PolicyInput(
                observation=np.asarray([0.0], dtype=np.float32),
                acknowledgements=(),
            ),
            decision_step=0,
        )
        with self.assertRaises(RuntimeError):
            oracle.act(
                PolicyInput(
                    observation=np.asarray([0.0], dtype=np.float32),
                    acknowledgements=(),
                ),
                decision_step=1,
            )


if __name__ == "__main__":
    unittest.main()
