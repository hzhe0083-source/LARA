import unittest

import torch

from Lara.model.modules.action_model.lara_moe import (
    ActionChunkExpertBank,
    LatentActionMoE,
    aggregate_episode_responsibilities,
    candidate_route_utility,
    centered_utility_targets,
    masked_topk_softmax,
    posterior_from_expert_losses,
    retained_probability_mass,
    route_switch_rate,
    route_stickiness_loss,
    uniform_balance_loss,
    utility_calibration_objective,
)


class LatentActionMoETest(unittest.TestCase):
    def test_masked_topk_softmax_respects_allowed_pool(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        allowed = torch.tensor([[False, True, True, False]])

        probs = masked_topk_softmax(logits, top_k=1, allowed_mask=allowed)

        self.assertTrue(torch.allclose(probs, torch.tensor([[0.0, 1.0, 0.0, 0.0]])))

    def test_forward_uses_episode_pool_and_chunk_topk(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=16,
            num_experts=6,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=32,
            router_hidden_size=24,
        )
        moe.train()

        conditioning_tokens = torch.randn(4, 5, 16)
        latent_action_tokens = torch.randn(4, 2, 16)
        output = moe(conditioning_tokens, latent_action_tokens=latent_action_tokens)

        self.assertEqual(output.tokens.shape, conditioning_tokens.shape)
        self.assertEqual(output.router_probs.shape, (4, 6))
        self.assertEqual(output.pool_mask.shape, (4, 6))
        self.assertEqual(output.active_mask.shape, (4, 6))
        self.assertTrue(torch.all(output.pool_mask.sum(dim=-1) == 3))
        self.assertTrue(torch.all(output.active_mask.sum(dim=-1) == 2))
        self.assertTrue(torch.all(output.active_mask <= output.pool_mask))
        self.assertTrue(torch.allclose(output.router_probs.sum(dim=-1), torch.ones(4)))
        self.assertEqual(output.active_usage.shape, (6,))
        self.assertEqual(output.pool_usage.shape, (6,))
        self.assertGreaterEqual(float(output.dead_expert_ratio), 0.0)
        self.assertLessEqual(float(output.dead_expert_ratio), 1.0)
        self.assertGreaterEqual(float(output.route_top1_match), 0.0)
        self.assertLessEqual(float(output.route_top1_match), 1.0)
        self.assertGreaterEqual(float(output.route_regret), 0.0)
        self.assertGreaterEqual(float(output.loss.detach()), 0.0)

    def test_predict_accepts_external_pool_mask(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=4,
            top_k=1,
            episode_pool_size=2,
            expert_hidden_size=16,
            router_hidden_size=12,
        )
        moe.eval()

        conditioning_tokens = torch.randn(2, 3, 8)
        pool_mask = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
            ]
        )
        output = moe.forward(conditioning_tokens, pool_mask=pool_mask)

        self.assertEqual(output.tokens.shape, conditioning_tokens.shape)
        self.assertTrue(torch.equal(output.pool_mask, pool_mask))
        self.assertTrue(torch.all(output.active_mask <= pool_mask))
        self.assertTrue(torch.all(output.router_probs[~pool_mask] == 0))

    def test_posterior_responsibility_prefers_low_expert_loss(self):
        losses = torch.tensor(
            [
                [0.1, 2.0, 3.0],
                [4.0, 0.2, 2.0],
            ]
        )

        posterior = posterior_from_expert_losses(losses, temperature=0.5)

        self.assertEqual(posterior.argmax(dim=-1).tolist(), [0, 1])
        self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones(2)))

    def test_forward_uses_expert_action_losses_as_posterior_teacher(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
            posterior_temperature=0.5,
        )

        conditioning_tokens = torch.randn(2, 4, 8)
        expert_losses = torch.tensor(
            [
                [0.1, 3.0, 2.0],
                [2.0, 0.2, 4.0],
            ]
        )
        output = moe(conditioning_tokens, expert_action_losses=expert_losses)

        self.assertEqual(output.posterior_probs.argmax(dim=-1).tolist(), [0, 1])
        self.assertTrue(torch.allclose(output.posterior_probs.sum(dim=-1), torch.ones(2)))
        self.assertGreaterEqual(float(output.loss.detach()), 0.0)

    def test_aggregates_episode_responsibilities_for_pool_targets(self):
        posterior = torch.tensor(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.1, 0.9],
            ]
        )
        episode_ids = torch.tensor([3, 3, 7])

        targets = aggregate_episode_responsibilities(posterior, episode_ids)

        expected_ep3 = torch.tensor([0.6, 0.4])
        self.assertTrue(torch.allclose(targets[0], expected_ep3))
        self.assertTrue(torch.allclose(targets[1], expected_ep3))
        self.assertTrue(torch.allclose(targets[2], posterior[2]))

    def test_forward_accepts_trajectory_pool_target(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
        )
        conditioning_tokens = torch.randn(3, 4, 8)
        expert_losses = torch.tensor(
            [
                [0.1, 3.0, 2.0],
                [2.0, 0.2, 4.0],
                [0.5, 2.0, 1.0],
            ]
        )
        posterior = posterior_from_expert_losses(expert_losses)
        pool_target = aggregate_episode_responsibilities(posterior, torch.tensor([1, 1, 2]))

        output = moe(
            conditioning_tokens,
            expert_action_losses=expert_losses,
            pool_target_probs=pool_target,
        )

        self.assertGreaterEqual(float(output.pool_loss), 0.0)
        self.assertEqual(output.pool_probs.shape, pool_target.shape)

    def test_centered_utility_targets_respect_candidate_mask(self):
        utilities = torch.tensor(
            [
                [1.0, 3.0, 100.0],
                [2.0, 4.0, 6.0],
            ]
        )
        candidate_mask = torch.tensor(
            [
                [True, True, False],
                [False, True, True],
            ]
        )

        targets, mask = centered_utility_targets(utilities, candidate_mask)

        self.assertTrue(torch.equal(mask, candidate_mask))
        self.assertTrue(torch.allclose(targets[0], torch.tensor([-1.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(targets[1], torch.tensor([0.0, -1.0, 1.0])))

    def test_utility_calibration_objective_trains_router_logits(self):
        router_logits = torch.tensor([[0.0, 0.5, -0.5]], requires_grad=True)
        utilities = torch.tensor([[0.0, 2.0, -1.0]])

        loss, rank_loss, calibration_error = utility_calibration_objective(
            router_logits,
            utilities,
            rank_loss_weight=0.5,
        )

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreaterEqual(float(rank_loss.detach()), 0.0)
        self.assertGreaterEqual(float(calibration_error.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(router_logits.grad)

    def test_forward_accepts_utility_scores(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
            utility_loss_weight=1.0,
            utility_rank_loss_weight=0.25,
        )
        conditioning_tokens = torch.randn(2, 4, 8)
        utility_scores = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [-0.5, 0.5, 1.5],
            ]
        )

        output = moe(conditioning_tokens, utility_scores=utility_scores)

        self.assertGreater(float(output.utility_loss), 0.0)
        self.assertGreaterEqual(float(output.utility_rank_loss), 0.0)
        self.assertGreaterEqual(float(output.utility_calibration_error), 0.0)
        self.assertTrue(torch.allclose(output.loss, output.utility_loss))

    def test_action_chunk_expert_bank_produces_per_expert_losses(self):
        torch.manual_seed(0)
        experts = ActionChunkExpertBank(
            hidden_size=8,
            num_experts=3,
            expert_hidden_size=16,
            action_horizon=4,
            action_dim=2,
        )
        tokens = torch.randn(2, 5, 8)
        target_actions = torch.randn(2, 4, 2)

        pred_actions = experts(tokens)
        losses = experts.reconstruction_losses(
            pred_actions,
            target_actions,
            execution_horizon=2,
            execution_loss_weight=1.0,
            prediction_loss_weight=0.5,
        )
        posterior = posterior_from_expert_losses(losses.detach())
        weighted_loss = (losses * posterior).sum(dim=-1).mean()

        self.assertEqual(pred_actions.shape, (2, 3, 4, 2))
        self.assertEqual(losses.shape, (2, 3))
        self.assertGreater(float(weighted_loss.detach()), 0.0)

    def test_balance_and_stickiness_losses(self):
        balanced = torch.tensor(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
        collapsed = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        previous = torch.tensor(
            [
                [0.25, 0.75],
                [0.75, 0.25],
            ]
        )

        self.assertLess(float(uniform_balance_loss(balanced)), float(uniform_balance_loss(collapsed)))
        self.assertGreater(float(route_stickiness_loss(balanced, previous)), 0.0)

    def test_forward_accepts_balance_and_stickiness_weights(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
            balance_loss_weight=0.1,
            stickiness_loss_weight=0.2,
        )
        conditioning_tokens = torch.randn(2, 4, 8)
        previous_router_probs = torch.full((2, 3), 1.0 / 3.0)

        output = moe(conditioning_tokens, previous_router_probs=previous_router_probs)

        expected = 0.1 * output.balance_loss + 0.2 * output.stickiness_loss
        self.assertTrue(torch.allclose(output.loss, expected))
        self.assertGreaterEqual(float(output.balance_loss), 0.0)
        self.assertGreaterEqual(float(output.stickiness_loss), 0.0)

    def test_route_switch_rate_uses_valid_pairs(self):
        route_ids = torch.tensor(
            [
                [1, 1, 2, 2],
                [3, 4, 4, 5],
            ]
        )
        valid_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        )

        switch_rate = route_switch_rate(route_ids, valid_mask)

        self.assertTrue(torch.allclose(switch_rate, torch.tensor(0.6)))

    def test_retained_probability_mass_monotonic_with_retention_fraction(self):
        probs = torch.tensor(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )

        curve = retained_probability_mass(probs, [0.25, 0.5, 1.0])

        self.assertLessEqual(float(curve[0.25]), float(curve[0.5]))
        self.assertLessEqual(float(curve[0.5]), float(curve[1.0]))
        self.assertTrue(torch.allclose(curve[1.0], torch.tensor(1.0)))

    def test_candidate_route_utility_combines_value_progress_uncertainty_and_cost(self):
        value = torch.tensor([[1.0, 2.0]])
        progress = torch.tensor([[0.5, 1.0]])
        uncertainty = torch.tensor([[0.25, 0.5]])
        cost = torch.tensor([[0.1, 0.2]])

        utility = candidate_route_utility(
            value_scores=value,
            progress_scores=progress,
            uncertainty_scores=uncertainty,
            cost_scores=cost,
            progress_weight=2.0,
            uncertainty_weight=0.5,
            cost_weight=3.0,
        )

        expected = value + 2.0 * progress - 0.5 * uncertainty - 3.0 * cost
        self.assertTrue(torch.allclose(utility, expected))

    def test_candidate_route_utility_validates_component_shapes(self):
        with self.assertRaisesRegex(ValueError, "share shape"):
            candidate_route_utility(
                value_scores=torch.zeros(2, 3),
                progress_scores=torch.zeros(2, 4),
            )


if __name__ == "__main__":
    unittest.main()
