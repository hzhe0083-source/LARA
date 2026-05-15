import unittest

import torch
from omegaconf import OmegaConf

from Lara.model.framework.act import ActionHeadAdapter


def tiny_action_config():
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 32,
                    "add_pos_embed": True,
                    "max_seq_len": 16,
                    "action_dim": 2,
                    "state_dim": 3,
                    "future_action_window_size": 2,
                    "action_horizon": 3,
                    "execution_horizon": 1,
                    "execution_loss_weight": 1.0,
                    "prediction_loss_weight": 0.5,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 1,
                    "use_latent_action_tokens": True,
                    "use_latent_action_head": False,
                    "use_lara_moe": False,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 16,
                    "num_inference_timesteps": 1,
                    "num_target_vision_tokens": 1,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 16,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "num_layers": 1,
                        "output_dim": 32,
                        "positional_embeddings": None,
                        "max_num_positional_embeddings": 16,
                    },
                }
            },
            "trainer": {"repeated_diffusion_steps": 1},
        }
    )


class ActionHeadAdapterSmokeTest(unittest.TestCase):
    def test_forward_and_predict_with_dummy_so101_shapes(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(config=tiny_action_config(), context_hidden_size=16)
        adapter.eval()

        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        actions = torch.randn(1, 3, 2)
        state = torch.randn(1, 1, 3)

        output = adapter(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            actions=actions,
            state=state,
            return_aux=True,
        )

        self.assertIn("total_action_loss", output)
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

        pred_actions = adapter.predict_action(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            state=state,
        )

        self.assertEqual(pred_actions.shape, (1, 3, 2))
        self.assertTrue(torch.isfinite(pred_actions).all().item())


if __name__ == "__main__":
    unittest.main()
