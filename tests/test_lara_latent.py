import unittest

import torch

from Lara.model.modules.action_model.lara_latent import (
    LatentActionHead,
    LatentActionTransitionHead,
    PosteriorLatentActionEncoder,
    VectorQuantizer,
)


class LatentActionHeadTest(unittest.TestCase):
    def test_posterior_encoder_returns_latent_tokens(self):
        torch.manual_seed(0)
        encoder = PosteriorLatentActionEncoder(
            context_dim=8,
            action_dim=3,
            action_horizon=4,
            num_latent_tokens=2,
            hidden_dim=16,
        )
        context_tokens = torch.randn(2, 5, 8)
        future_actions = torch.randn(2, 4, 3)

        latent_tokens = encoder(context_tokens, future_actions)

        self.assertEqual(latent_tokens.shape, (2, 2, 8))

    def test_posterior_encoder_rejects_wrong_horizon(self):
        encoder = PosteriorLatentActionEncoder(
            context_dim=8,
            action_dim=3,
            action_horizon=4,
            num_latent_tokens=2,
            hidden_dim=16,
        )

        with self.assertRaisesRegex(ValueError, "Expected future_actions horizon 4"):
            encoder(torch.randn(2, 5, 8), torch.randn(2, 3, 3))

    def test_vector_quantizer_returns_codes_and_perplexity(self):
        torch.manual_seed(0)
        quantizer = VectorQuantizer(codebook_size=8, code_dim=6, commitment_weight=0.25)
        inputs = torch.randn(2, 3, 6)

        quantized, indices, vq_loss, code_usage_loss, perplexity = quantizer(inputs)

        self.assertEqual(quantized.shape, inputs.shape)
        self.assertEqual(indices.shape, (2, 3))
        self.assertGreaterEqual(float(vq_loss.detach()), 0.0)
        self.assertGreaterEqual(float(code_usage_loss.detach()), 0.0)
        self.assertGreaterEqual(float(perplexity), 1.0)

    def test_latent_action_head_forward_and_predict(self):
        torch.manual_seed(0)
        head = LatentActionHead(
            context_dim=8,
            action_dim=3,
            action_horizon=4,
            num_latent_tokens=2,
            codebook_size=8,
            hidden_dim=16,
            code_usage_loss_weight=0.1,
        )
        context_tokens = torch.randn(2, 5, 8)
        future_actions = torch.randn(2, 4, 3)

        output = head(context_tokens, future_actions)
        predicted_tokens = head.predict(context_tokens)

        self.assertEqual(output.tokens.shape, (2, 2, 8))
        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertGreaterEqual(float(output.vq_loss), 0.0)
        self.assertGreaterEqual(float(output.prior_loss), 0.0)
        self.assertGreaterEqual(float(output.code_usage_loss), 0.0)
        self.assertEqual(predicted_tokens.shape, (2, 2, 8))

    def test_transition_head_predicts_execution_and_prediction_boundary_states(self):
        torch.manual_seed(0)
        head = LatentActionTransitionHead(
            context_dim=8,
            state_dim=5,
            hidden_dim=16,
            num_boundaries=2,
        )
        context_tokens = torch.randn(2, 5, 8)
        latent_tokens = torch.randn(2, 2, 8)

        predicted_states = head(context_tokens, latent_tokens)

        self.assertEqual(predicted_states.shape, (2, 2, 5))
        self.assertTrue(torch.isfinite(predicted_states).all().item())


if __name__ == "__main__":
    unittest.main()
