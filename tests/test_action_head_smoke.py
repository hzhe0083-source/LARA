import unittest

import torch
from omegaconf import OmegaConf

from Lara.model.framework.act import ActionHeadAdapter


def tiny_action_config(**action_overrides):
    action_model = {
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
    action_model.update(action_overrides)
    return OmegaConf.create(
        {
            "framework": {"action_model": action_model},
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

    def test_forward_with_direct_moe_action_experts_adds_aux_loss(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=2,
                lara_top_k=1,
                lara_use_direct_action_experts=True,
                lara_direct_expert_loss_weight=0.5,
            ),
            context_hidden_size=16,
        )
        adapter.train()

        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        actions = torch.randn(1, 3, 2)
        state = torch.randn(1, 1, 3)

        output = adapter(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            actions=actions,
            state=state,
            trajectory_ids=[7],
            return_aux=True,
        )

        self.assertIn("moe_direct_expert_loss", output)
        self.assertTrue(torch.isfinite(output["moe_direct_expert_loss"]).item())
        expected_total = output["action_loss"] + output["moe_router_loss"] + output["moe_direct_expert_loss"]
        self.assertTrue(torch.allclose(output["total_action_loss"], expected_total))

    def test_moe_can_use_expert_action_losses_as_utility_scores(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=3,
                lara_top_k=2,
                lara_use_direct_action_experts=True,
                lara_use_action_loss_utility=True,
                lara_utility_loss_weight=0.25,
            ),
            context_hidden_size=16,
        )
        adapter.train()

        embodied_tokens = torch.randn(2, 2, 16)
        latent_tokens = torch.randn(2, 1, 16)
        actions = torch.randn(2, 3, 2)
        state = torch.randn(2, 1, 3)

        output = adapter(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            actions=actions,
            state=state,
            trajectory_ids=[5, 5],
            return_aux=True,
        )

        self.assertIn("moe_utility_loss", output)
        self.assertIn("moe_utility_scores", output)
        self.assertIn("moe_route_quality_utility_spearman", output)
        self.assertIn("moe_route_quality_utility_topk_consistency", output)
        self.assertIn("moe_route_quality_retained_probability_mass_0_5", output)
        self.assertGreater(float(output["moe_utility_loss"]), 0.0)
        self.assertEqual(output["moe_utility_scores"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

    def test_predict_action_can_reuse_resident_pool_mask(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=4,
                lara_episode_pool_size=2,
                lara_top_k=1,
                lara_use_expert_loss_posterior=False,
            ),
            context_hidden_size=16,
        )
        adapter.eval()

        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        state = torch.randn(1, 1, 3)

        resident_pool = adapter.select_resident_pool(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
        )
        pred_actions = adapter.predict_action(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            state=state,
            pool_mask=resident_pool.mask,
        )

        self.assertEqual(resident_pool.mask.shape, (1, 4))
        self.assertTrue(torch.all(resident_pool.mask.sum(dim=-1) == 2))
        self.assertEqual(pred_actions.shape, (1, 3, 2))
        self.assertTrue(torch.isfinite(pred_actions).all().item())

    def test_direct_moe_action_output_trains_and_predicts_action_chunks(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=4,
                lara_episode_pool_size=2,
                lara_top_k=1,
                lara_use_direct_action_experts=True,
                lara_use_direct_action_output=True,
                lara_use_expert_loss_posterior=False,
            ),
            context_hidden_size=16,
        )
        adapter.train()

        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        actions = torch.randn(1, 3, 2)
        state = torch.randn(1, 1, 3)

        output = adapter(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            actions=actions,
            state=state,
            trajectory_ids=[3],
            return_aux=True,
        )

        self.assertIn("moe_direct_routed_action_loss", output)
        self.assertTrue(torch.allclose(output["action_loss"], output["moe_direct_routed_action_loss"]))
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

        adapter.eval()
        pred_actions = adapter.predict_action(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            state=state,
        )

        self.assertEqual(pred_actions.shape, (1, 3, 2))
        self.assertTrue(torch.isfinite(pred_actions).all().item())

    def test_transition_head_adds_boundary_state_loss_when_targets_exist(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                lara_use_transition_head=True,
                lara_transition_hidden_dim=16,
                lara_transition_loss_weight=0.25,
            ),
            context_hidden_size=16,
        )
        adapter.train()

        embodied_tokens = torch.randn(2, 2, 16)
        latent_tokens = torch.randn(2, 1, 16)
        actions = torch.randn(2, 3, 2)
        state = torch.randn(2, 1, 3)
        execution_target = torch.randn(2, 3)
        prediction_target = torch.randn(2, 3)

        output = adapter(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            actions=actions,
            state=state,
            execution_state_target=execution_target,
            prediction_state_target=prediction_target,
            execution_state_target_mask=[True, True],
            prediction_state_target_mask=[True, False],
            return_aux=True,
        )

        self.assertIn("transition_state_loss", output)
        self.assertGreater(float(output["transition_state_loss"].detach()), 0.0)
        expected_total = output["action_loss"] + output["transition_state_loss"]
        self.assertTrue(torch.allclose(output["total_action_loss"], expected_total))


if __name__ == "__main__":
    unittest.main()
