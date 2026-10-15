from __future__ import annotations

import unittest
import copy
import math

import numpy as np

try:
    import torch
except ImportError:  # The lightweight Windows control environment omits torch.
    torch = None


@unittest.skipIf(torch is None, "torch is validated in the remote research environment")
class TorchComponentsTest(unittest.TestCase):
    def test_cql_update_and_checkpoint_are_finite(self) -> None:
        from dt_corl_delay.cql import CQLAgent, CQLHyperparameters

        device = torch.device("cpu")
        hyperparameters = CQLHyperparameters(cql_n_actions=3, bc_steps=1)
        agent = CQLAgent(3, 2, (16, 16), 3e-4, hyperparameters, device)
        generator = torch.Generator().manual_seed(11)
        batch = (
            torch.randn(8, 3, generator=generator),
            torch.tanh(torch.randn(8, 2, generator=generator)),
            torch.randn(8, 1, generator=generator),
            torch.randn(8, 3, generator=generator),
            torch.zeros(8, 1),
        )
        first_metrics = agent.update(batch)
        second_metrics = agent.update(batch)
        self.assertTrue(all(torch.isfinite(value) for value in first_metrics.values()))
        self.assertTrue(all(torch.isfinite(value) for value in second_metrics.values()))
        action = agent.act(np.zeros(3, dtype=np.float32), torch.zeros(1, 3), torch.ones(1, 3))
        self.assertEqual(action.shape, (2,))
        clone = CQLAgent(3, 2, (16, 16), 3e-4, hyperparameters, device)
        clone.load_checkpoint(agent.checkpoint())
        np.testing.assert_array_equal(
            action,
            clone.act(np.zeros(3, dtype=np.float32), torch.zeros(1, 3), torch.ones(1, 3)),
        )

    def test_cql_exact_continuation_with_rng_and_optimizer_state(self) -> None:
        from dt_corl_delay.cql import CQLAgent, CQLHyperparameters

        device = torch.device("cpu")
        torch.manual_seed(17)
        hyperparameters = CQLHyperparameters(cql_n_actions=2, bc_steps=0)
        agent = CQLAgent(3, 2, (8, 8), 3e-4, hyperparameters, device)
        batch_generator = torch.Generator().manual_seed(19)
        batch = (
            torch.randn(6, 3, generator=batch_generator),
            torch.tanh(torch.randn(6, 2, generator=batch_generator)),
            torch.randn(6, 1, generator=batch_generator),
            torch.randn(6, 3, generator=batch_generator),
            torch.zeros(6, 1),
        )
        agent.update(batch)
        checkpoint = copy.deepcopy(agent.checkpoint())
        rng_state = torch.get_rng_state()
        agent.update(batch)
        expected = {
            f"q.{name}": value.detach().clone() for name, value in agent.q.state_dict().items()
        }
        expected.update(
            {
                f"policy.{name}": value.detach().clone()
                for name, value in agent.policy.state_dict().items()
            }
        )

        clone = CQLAgent(3, 2, (8, 8), 3e-4, hyperparameters, device)
        clone.load_checkpoint(checkpoint, load_optimizers=True)
        torch.set_rng_state(rng_state)
        clone.update(batch)
        actual = {f"q.{name}": value for name, value in clone.q.state_dict().items()}
        actual.update(
            {f"policy.{name}": value for name, value in clone.policy.state_dict().items()}
        )
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)

    def test_dynamics_and_cadr_shapes_are_finite(self) -> None:
        from dt_corl_delay.channel import PolicyInput
        from dt_corl_delay.controllers import CadrController
        from dt_corl_delay.dynamics import DeltaDynamicsEnsemble
        from dt_corl_delay.iql import IQLAgent, IQLHyperparameters

        device = torch.device("cpu")
        agent = IQLAgent(3, 2, (16, 16), 3e-4, IQLHyperparameters(), device)
        dynamics = DeltaDynamicsEnsemble(3, 2, (16, 16), 3, 3e-4, device)
        generator = torch.Generator().manual_seed(7)
        states = torch.randn(32, 3, generator=generator)
        actions = torch.tanh(torch.randn(32, 2, generator=generator))
        next_states = states + 0.1 * torch.randn(32, 3, generator=generator)
        metrics = dynamics.update(states, actions, next_states, generator)
        self.assertTrue(torch.isfinite(metrics["dynamics_loss"]))
        history = [actions[index : index + 1] for index in range(3)]
        batched = dynamics.rollout_age_candidates(states[:1], history, maximum_age=3)
        self.assertEqual(tuple(batched.shape), (4, 3, 3))
        for age in range(4):
            suffix = history[len(history) - age :] if age else []
            expected = dynamics.rollout_members(states[:1], suffix)
            torch.testing.assert_close(batched[age], expected)

        single_member = DeltaDynamicsEnsemble(3, 2, (8,), 1, 3e-4, device)
        single_metrics = single_member.update(
            states,
            actions,
            next_states,
            torch.Generator().manual_seed(17),
        )
        self.assertTrue(torch.isfinite(single_metrics["dynamics_loss"]))
        single_candidates = single_member.rollout_age_candidates(
            states[:1],
            history,
            maximum_age=3,
        )
        self.assertEqual(tuple(single_candidates.shape), (4, 1, 3))

        controller = CadrController(
            agent,
            dynamics,
            torch.zeros(1, 3),
            torch.ones(1, 3),
            max_delay=4,
            ack_lag=1,
            memory_steps=16,
        )
        controller.reset(np.zeros(3, dtype=np.float32))
        first_action = controller.act(
            PolicyInput(np.zeros(3, dtype=np.float32), ()),
            decision_step=0,
        )
        second_action = controller.act(
            PolicyInput(np.ones(3, dtype=np.float32), ()),
            decision_step=1,
        )
        direct_second = agent.act(
            np.ones(3, dtype=np.float32),
            torch.zeros(1, 3),
            torch.ones(1, 3),
        )
        self.assertEqual(first_action.shape, (2,))
        self.assertEqual(second_action.shape, (2,))
        self.assertTrue(np.isfinite(first_action).all())
        self.assertTrue(np.isfinite(second_action).all())
        np.testing.assert_array_equal(second_action, direct_second)
        diagnostics = controller.diagnostics()
        self.assertEqual(diagnostics["mean_observation_changed"], 0.5)

        frozen_prior = CadrController(
            agent,
            dynamics,
            torch.zeros(1, 3),
            torch.ones(1, 3),
            max_delay=4,
            ack_lag=1,
            memory_steps=16,
            use_content_conditioning=False,
            posterior_updates=False,
        )
        frozen_prior.reset(np.zeros(3, dtype=np.float32))
        initial_probabilities = frozen_prior.estimator.probabilities
        frozen_prior.act(
            PolicyInput(np.zeros(3, dtype=np.float32), ()),
            decision_step=0,
        )
        frozen_prior.act(
            PolicyInput(np.ones(3, dtype=np.float32), ()),
            decision_step=1,
        )
        np.testing.assert_array_equal(
            frozen_prior.estimator.probabilities,
            initial_probabilities,
        )
        frozen_diagnostics = frozen_prior.diagnostics()
        self.assertEqual(frozen_diagnostics["mean_observation_changed"], 0.0)
        self.assertEqual(
            frozen_diagnostics["mean_exact_age_from_unchanged_observation"],
            0.0,
        )

    def test_state_disagreement_decomposition_obeys_total_variance(self) -> None:
        from dt_corl_delay.controllers import state_disagreement_components

        age_states = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0]],
                [[2.0, 2.0], [4.0, 2.0]],
            ]
        )
        age_weights = torch.tensor([0.25, 0.75])
        total, epistemic, age = state_disagreement_components(
            age_states,
            age_weights,
        )
        torch.testing.assert_close(
            total.square(),
            epistemic.square() + age.square(),
        )
        self.assertGreater(float(epistemic), 0.0)
        self.assertGreater(float(age), 0.0)

    def test_reconstruction_reliability_has_a_dead_zone(self) -> None:
        from dt_corl_delay.controllers import reconstruction_reliability

        self.assertEqual(reconstruction_reliability(0.1, 64.0, 0.1), 1.0)
        self.assertEqual(reconstruction_reliability(0.05, 64.0, 0.1), 1.0)
        self.assertAlmostEqual(
            reconstruction_reliability(0.2, 8.0, 0.1),
            math.exp(-0.8),
        )

    def test_full_dynamics_validation_covers_each_transition_once(self) -> None:
        from types import SimpleNamespace

        from dt_corl_delay.dynamics import DeltaDynamicsEnsemble
        from dt_corl_delay.train_cadr import _full_validation_metrics

        device = torch.device("cpu")
        dynamics = DeltaDynamicsEnsemble(2, 1, (4,), 2, 3e-4, device)
        dataset = SimpleNamespace(
            observations=torch.zeros(5, 2),
            actions=torch.zeros(5, 1),
            next_observations=torch.ones(5, 2),
        )
        metrics = _full_validation_metrics(
            dynamics,
            dataset,
            torch.tensor([4, 1, 3]),
            batch_size=2,
        )
        self.assertEqual(metrics["validation_transitions_evaluated"], 3)
        self.assertTrue(metrics["deterministic_full_validation_split"])
        self.assertGreaterEqual(metrics["validation_mse"], 0.0)

    def test_full_multistep_validation_covers_complete_sequences(self) -> None:
        from types import SimpleNamespace

        from dt_corl_delay.dynamics import DeltaDynamicsEnsemble
        from dt_corl_delay.train_cadr import _full_multistep_validation_metrics

        device = torch.device("cpu")
        dynamics = DeltaDynamicsEnsemble(2, 1, (4,), 2, 3e-4, device)
        dataset = SimpleNamespace(
            size=6,
            observations=torch.zeros(6, 2),
            actions=torch.zeros(6, 1),
            next_observations=torch.ones(6, 2),
            episode_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        metrics = _full_multistep_validation_metrics(
            dynamics,
            dataset,
            torch.tensor([0, 1]),
            horizon=2,
            batch_size=2,
        )
        self.assertEqual(metrics["validation_sequences_evaluated"], 4)
        self.assertEqual(metrics["validation_rollout_horizon"], 2)
        self.assertEqual(len(metrics["validation_horizon_mse"]), 2)
        self.assertGreaterEqual(metrics["validation_multistep_mse"], 0.0)

    def test_pinned_official_dtcorl_adapter_forward_is_finite(self) -> None:
        from dt_corl_delay.channel import PolicyInput
        from dt_corl_delay.iql import IQLAgent, IQLHyperparameters
        from dt_corl_delay.official_dtcorl import (
            OFFICIAL_DT_CORL_COMMIT,
            OfficialDTCorlAckMapController,
            build_official_reconstructor,
            official_source_provenance,
        )

        device = torch.device("cpu")
        agent = IQLAgent(3, 2, (8, 8), 3e-4, IQLHyperparameters(), device)
        autoencoder, dynamics = build_official_reconstructor(
            state_dim=3,
            action_dim=2,
            maximum_delay=2,
            latent_dim=8,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            attention_dropout=0.0,
            residual_dropout=0.0,
            hidden_dropout=0.0,
            device=device,
        )
        autoencoder.eval()
        dynamics.eval()
        controller = OfficialDTCorlAckMapController(
            agent,
            autoencoder,
            dynamics,
            torch.zeros(1, 3),
            torch.ones(1, 3),
            maximum_delay=2,
            ack_lag=1,
        )
        controller.reset(np.zeros(3, dtype=np.float32))
        first_action = controller.act(
            PolicyInput(np.zeros(3, dtype=np.float32), ()),
            decision_step=0,
        )
        controller.observe_transition(first_action, 0.25, False, False)
        second_action = controller.act(
            PolicyInput(np.ones(3, dtype=np.float32), ()),
            decision_step=1,
        )
        self.assertEqual(first_action.shape, (2,))
        self.assertTrue(np.isfinite(second_action).all())
        self.assertEqual(official_source_provenance()["commit"], OFFICIAL_DT_CORL_COMMIT)

    def test_knn_policy_retrieves_exact_reference_action(self) -> None:
        from dt_corl_delay.knn import KNNPolicyAgent

        states = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
        actions = torch.tensor([[0.1], [0.7], [-0.4]])
        agent = KNNPolicyAgent(states, actions, neighbors=1)
        retrieved = agent.act_normalized(states[1:2])
        torch.testing.assert_close(retrieved, actions[1:2], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
