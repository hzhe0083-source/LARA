import unittest

import torch

from Lara.model.modules.action_model.lara_moe import (
    ActionChunkExpertBank,
    LatentActionMoE,
    RouteUtilityHead,
    aggregate_episode_responsibilities,
    candidate_route_utility,
    centered_utility_targets,
    expert_diversity_loss,
    masked_topk_softmax,
    posterior_from_expert_losses,
    retained_probability_mass,
    route_entropy_regularization_loss,
    route_switch_rate,
    route_stickiness_loss,
    sparse_route_budget,
    uniform_balance_loss,
    utility_calibration_objective,
    utility_component_supervision_loss,
    utility_from_expert_losses,
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

    def test_select_resident_pool_can_be_reused_for_chunk_routing(self):
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

        first_chunk_tokens = torch.randn(2, 3, 8)
        next_chunk_tokens = torch.randn(2, 3, 8)

        resident_pool = moe.select_resident_pool(first_chunk_tokens)
        output = moe(next_chunk_tokens, pool_mask=resident_pool.mask)

        self.assertEqual(resident_pool.logits.shape, (2, 4))
        self.assertEqual(resident_pool.probs.shape, (2, 4))
        self.assertTrue(torch.all(resident_pool.mask.sum(dim=-1) == 2))
        self.assertTrue(torch.equal(output.pool_mask, resident_pool.mask))
        self.assertTrue(torch.all(output.active_mask <= resident_pool.mask))

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

    def test_posterior_responsibility_supports_floor_and_top_r(self):
        losses = torch.tensor([[0.1, 0.2, 3.0, 4.0]])

        posterior = posterior_from_expert_losses(
            losses,
            temperature=0.5,
            uniform_floor=0.2,
            top_r=2,
        )

        self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones(1)))
        self.assertGreater(float(posterior[0, 0]), 0.0)
        self.assertGreater(float(posterior[0, 1]), 0.0)
        self.assertEqual(float(posterior[0, 2]), 0.0)
        self.assertEqual(float(posterior[0, 3]), 0.0)
        self.assertGreaterEqual(float(posterior.min()), 0.0)

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

    def test_route_utility_head_produces_components_and_scores(self):
        torch.manual_seed(0)
        head = RouteUtilityHead(
            hidden_size=8,
            num_experts=3,
            utility_hidden_size=16,
            progress_weight=2.0,
            uncertainty_weight=0.5,
            cost_weight=3.0,
        )
        tokens = torch.randn(2, 4, 8)
        cost = torch.ones(2, 3) * 0.1

        output = head(tokens, cost_scores=cost)

        self.assertEqual(output["utility_scores"].shape, (2, 3))
        self.assertEqual(output["value_scores"].shape, (2, 3))
        self.assertEqual(output["progress_scores"].shape, (2, 3))
        self.assertEqual(output["uncertainty_scores"].shape, (2, 3))
        self.assertTrue(torch.all(output["uncertainty_scores"] >= 0))

    def test_forward_can_generate_utility_scores_from_head(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
            utility_loss_weight=1.0,
            use_utility_head=True,
            utility_hidden_size=12,
        )
        conditioning_tokens = torch.randn(2, 4, 8)

        output = moe(conditioning_tokens)

        self.assertIsNotNone(output.utility_scores)
        self.assertEqual(output.utility_scores.shape, (2, 3))
        self.assertGreater(float(output.utility_loss), 0.0)
        self.assertTrue(torch.allclose(output.loss, output.utility_loss))

    def test_utility_component_supervision_loss_uses_available_targets(self):
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        progress = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        uncertainty = torch.ones(2, 2)
        target_mask = torch.tensor([[True, False], [True, True]])

        loss = utility_component_supervision_loss(
            value_scores=value,
            progress_scores=progress,
            uncertainty_scores=uncertainty,
            value_targets=torch.zeros(2, 2),
            progress_targets=torch.ones(2, 2),
            target_mask=target_mask,
        )

        self.assertGreater(float(loss), 0.0)

    def test_forward_can_supervise_utility_head_components(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
            use_utility_head=True,
            utility_hidden_size=12,
            utility_head_loss_weight=0.5,
        )
        conditioning_tokens = torch.randn(2, 4, 8)
        value_targets = torch.zeros(2, 3)
        progress_targets = torch.ones(2, 3)

        output = moe(
            conditioning_tokens,
            utility_value_targets=value_targets,
            utility_progress_targets=progress_targets,
        )

        self.assertGreater(float(output.utility_head_loss), 0.0)
        self.assertTrue(torch.allclose(output.loss, 0.5 * output.utility_head_loss))

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

    def test_action_chunk_expert_bank_routes_weighted_actions(self):
        pred_actions = torch.tensor(
            [
                [
                    [[1.0, 0.0], [3.0, 0.0]],
                    [[5.0, 0.0], [7.0, 0.0]],
                ]
            ]
        )
        route_weights = torch.tensor([[0.25, 0.75]])
        target_actions = torch.zeros(1, 2, 2)

        routed_actions = ActionChunkExpertBank.routed_actions(pred_actions, route_weights)
        loss = ActionChunkExpertBank.action_chunk_loss(
            routed_actions,
            target_actions,
            execution_horizon=1,
            execution_loss_weight=1.0,
            prediction_loss_weight=0.5,
        )

        self.assertTrue(torch.allclose(routed_actions, torch.tensor([[[4.0, 0.0], [6.0, 0.0]]])))
        self.assertGreater(float(loss), 0.0)

    def test_action_chunk_expert_bank_can_condition_on_state(self):
        torch.manual_seed(0)
        experts = ActionChunkExpertBank(
            hidden_size=8,
            num_experts=2,
            expert_hidden_size=16,
            action_horizon=3,
            action_dim=2,
            state_dim=4,
        )
        tokens = torch.randn(1, 5, 8)
        state_a = torch.zeros(1, 1, 4)
        state_b = torch.ones(1, 1, 4)

        actions_a = experts(tokens, state=state_a)
        actions_b = experts(tokens, state=state_b)

        self.assertEqual(actions_a.shape, (1, 2, 3, 2))
        self.assertFalse(torch.allclose(actions_a, actions_b))

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

    def test_expert_diversity_loss_penalizes_identical_outputs(self):
        identical = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
        diverse = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])

        self.assertGreater(float(expert_diversity_loss(identical)), float(expert_diversity_loss(diverse)))
        self.assertEqual(float(expert_diversity_loss(torch.ones(1, 1, 1, 2))), 0.0)

    def test_route_entropy_regularizer_prefers_higher_entropy(self):
        collapsed = torch.tensor([[1.0, 0.0]])
        uniform = torch.tensor([[0.5, 0.5]])

        self.assertLess(
            float(route_entropy_regularization_loss(uniform, uniform)),
            float(route_entropy_regularization_loss(collapsed, collapsed)),
        )

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

    def test_forward_accepts_diversity_and_entropy_weights(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=3,
            top_k=2,
            episode_pool_size=3,
            expert_hidden_size=16,
            router_hidden_size=12,
            diversity_loss_weight=0.1,
            entropy_loss_weight=0.2,
        )
        conditioning_tokens = torch.randn(2, 4, 8)

        output = moe(conditioning_tokens)

        expected = 0.1 * output.diversity_loss + 0.2 * output.entropy_loss
        self.assertTrue(torch.allclose(output.loss, expected))
        self.assertGreaterEqual(float(output.diversity_loss), 0.0)
        self.assertLessEqual(float(output.entropy_loss), 0.0)

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

    def test_utility_from_expert_losses_prefers_lower_losses(self):
        losses = torch.tensor([[0.1, 2.0, 1.0]])

        utility = utility_from_expert_losses(losses)

        self.assertEqual(utility.argmax(dim=-1).tolist(), [0])
        self.assertTrue(torch.allclose(utility.mean(dim=-1), torch.zeros(1), atol=1e-6))

    def test_utility_calibration_defaults_to_pool_candidates(self):
        torch.manual_seed(0)
        moe = LatentActionMoE(
            hidden_size=8,
            num_experts=4,
            top_k=1,
            episode_pool_size=2,
            expert_hidden_size=16,
            router_hidden_size=12,
            utility_loss_weight=1.0,
        )
        conditioning_tokens = torch.randn(2, 4, 8)
        pool_mask = torch.tensor(
            [
                [True, True, False, False],
                [False, True, True, False],
            ]
        )
        utility_scores = torch.tensor(
            [
                [1.0, 2.0, 100.0, 100.0],
                [100.0, -1.0, 1.0, 100.0],
            ]
        )

        output = moe(conditioning_tokens, pool_mask=pool_mask, utility_scores=utility_scores)

        self.assertGreater(float(output.utility_loss), 0.0)
        self.assertTrue(torch.equal(output.pool_mask, pool_mask))
        self.assertTrue(torch.isfinite(output.loss).item())

    def test_sparse_route_budget_reports_active_and_resident_fractions(self):
        metrics = sparse_route_budget(
            total_experts=8,
            active_experts=2,
            resident_experts=4,
            shared_params=100,
            params_per_expert=10,
        )

        self.assertEqual(metrics["active_expert_fraction"], 0.25)
        self.assertEqual(metrics["resident_expert_fraction"], 0.5)
        self.assertEqual(metrics["active_params"], 120.0)
        self.assertEqual(metrics["resident_params"], 140.0)
        self.assertEqual(metrics["total_params"], 180.0)

    def test_sparse_route_budget_rejects_active_experts_outside_resident_pool(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            sparse_route_budget(total_experts=8, active_experts=5, resident_experts=4)


if __name__ == "__main__":
    unittest.main()
