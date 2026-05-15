import unittest

import torch

from Lara.model.modules.action_model.lara_moe import (
    LatentActionMoE,
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


if __name__ == "__main__":
    unittest.main()
