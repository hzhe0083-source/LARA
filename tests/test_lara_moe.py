import unittest

import torch

from Lara.model.modules.action_model.lara_moe import (
    LatentActionMoE,
    aggregate_episode_responsibilities,
    masked_topk_softmax,
    posterior_from_expert_losses,
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


if __name__ == "__main__":
    unittest.main()
