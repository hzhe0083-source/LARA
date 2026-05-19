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

    def test_latent_action_head_uses_short_latent_horizon(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_latent_action_head=True,
                latent_action_horizon=1,
                lara_num_latent_tokens=2,
                lara_codebook_size=8,
                lara_latent_hidden_dim=16,
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
            actions_are_future=True,
            state=state,
            return_aux=True,
        )

        self.assertEqual(adapter.latent_action_horizon, 1)
        self.assertEqual(adapter.latent_action_head.posterior.action_horizon, 1)
        self.assertIn("latent_action_loss", output)
        self.assertIn("latent_action_reconstruction_loss", output)
        self.assertIn("latent_action_reconstruction_loss_weighted", output)
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

    def test_predict_action_can_return_moe_route_aux_for_closed_loop_cache(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_top_k=1,
                lara_episode_pool_size=2,
                lara_expert_hidden_dim=16,
                lara_router_hidden_dim=16,
            ),
            context_hidden_size=16,
        )
        adapter.eval()
        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        state = torch.randn(1, 1, 3)
        pool_mask = torch.tensor([[True, True, False]])
        previous_router_probs = torch.tensor([[0.5, 0.5, 0.0]])

        output = adapter.predict_action(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            state=state,
            pool_mask=pool_mask,
            previous_router_probs=previous_router_probs,
            return_aux=True,
        )

        self.assertEqual(output["actions"].shape, (1, 3, 2))
        self.assertEqual(output["router_probs"].shape, (1, 3))
        self.assertTrue(torch.equal(output["pool_mask"], pool_mask))
        self.assertTrue(torch.all(output["active_mask"] <= pool_mask))

    def test_future_actions_must_match_configured_horizon(self):
        adapter = ActionHeadAdapter(config=tiny_action_config(), context_hidden_size=16)
        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        short_future_actions = torch.randn(1, 2, 2)

        with self.assertRaisesRegex(ValueError, "future_actions must have exactly 3 steps"):
            adapter(
                embodied_action_tokens=embodied_tokens,
                latent_action_tokens=latent_tokens,
                actions=short_future_actions,
                actions_are_future=True,
                return_aux=True,
            )

    def test_action_mask_excludes_padded_future_steps_from_flow_loss(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(config=tiny_action_config(), context_hidden_size=16)
        adapter.train()
        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        actions = torch.randn(1, 3, 2)
        state = torch.randn(1, 1, 3)

        output = adapter(
            embodied_action_tokens=embodied_tokens,
            latent_action_tokens=latent_tokens,
            actions=actions,
            actions_are_future=True,
            action_mask=torch.tensor([[False, False, False]]),
            state=state,
            return_aux=True,
        )

        self.assertTrue(torch.allclose(output["action_loss"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(output["total_action_loss"], torch.tensor(0.0)))

    def test_future_action_mask_must_match_configured_horizon(self):
        adapter = ActionHeadAdapter(config=tiny_action_config(), context_hidden_size=16)
        embodied_tokens = torch.randn(1, 2, 16)
        latent_tokens = torch.randn(1, 1, 16)
        actions = torch.randn(1, 3, 2)

        with self.assertRaisesRegex(ValueError, "future_action_mask must have exactly 3 steps"):
            adapter(
                embodied_action_tokens=embodied_tokens,
                latent_action_tokens=latent_tokens,
                actions=actions,
                actions_are_future=True,
                action_mask=torch.tensor([[True, True]]),
                return_aux=True,
            )

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
                lara_pool_coverage_loss_weight=0.25,
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
        self.assertIn("moe_loss", output)
        self.assertIn("moe_route_distill_loss_raw", output)
        self.assertIn("moe_route_distill_loss_weighted", output)
        self.assertIn("moe_pool_distill_loss_weighted", output)
        self.assertIn("moe_pool_coverage_loss_weighted", output)
        self.assertIn("moe_pool_teacher_mass", output)
        self.assertIn("moe_pool_critical_miss_rate", output)
        self.assertTrue(torch.isfinite(output["moe_direct_expert_loss"]).item())
        expected_total = output["action_loss"] + output["moe_loss"] + output["moe_direct_expert_loss"]
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
        self.assertIn("moe_utility_loss_weighted", output)
        self.assertIn("moe_utility_scores", output)
        self.assertIn("moe_route_quality_utility_spearman", output)
        self.assertIn("moe_route_quality_utility_topk_consistency", output)
        self.assertIn("moe_route_quality_retained_probability_mass_0_5", output)
        self.assertGreater(float(output["moe_utility_loss"]), 0.0)
        self.assertEqual(output["moe_utility_scores"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

    def test_moe_can_supervise_utility_components_from_direct_expert_losses(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=3,
                lara_top_k=2,
                lara_use_direct_action_experts=True,
                lara_use_utility_head=True,
                lara_use_action_loss_utility_components=True,
                lara_utility_head_loss_weight=0.5,
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

        self.assertIn("moe_utility_head_loss", output)
        self.assertIn("moe_utility_value_scores", output)
        self.assertIn("moe_utility_progress_scores", output)
        self.assertIn("moe_utility_uncertainty_scores", output)
        self.assertGreater(float(output["moe_utility_head_loss"]), 0.0)
        self.assertEqual(output["moe_utility_value_scores"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

    def test_moe_can_use_transition_state_consistency_as_utility(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=3,
                lara_top_k=2,
                lara_use_transition_head=True,
                lara_transition_hidden_dim=16,
                lara_transition_loss_weight=0.25,
                lara_use_state_utility=True,
                lara_use_state_utility_components=True,
                lara_use_utility_head=True,
                lara_utility_loss_weight=0.25,
                lara_utility_head_loss_weight=0.5,
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
            trajectory_ids=[9, 9],
            execution_state_target=execution_target,
            prediction_state_target=prediction_target,
            execution_state_target_mask=[True, True],
            prediction_state_target_mask=[True, True],
            return_aux=True,
        )

        self.assertIn("moe_state_utility_error", output)
        self.assertIn("moe_utility_scores", output)
        self.assertIn("moe_utility_head_loss", output)
        self.assertGreater(float(output["moe_state_utility_error"]), 0.0)
        self.assertGreater(float(output["moe_utility_loss"]), 0.0)
        self.assertGreater(float(output["moe_utility_head_loss"]), 0.0)
        self.assertEqual(output["moe_utility_scores"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

    def test_direct_moe_uses_short_router_and_utility_horizons(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=3,
                lara_top_k=2,
                lara_use_direct_action_experts=True,
                lara_use_action_loss_utility=True,
                router_horizon=1,
                utility_horizon=2,
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
            actions_are_future=True,
            state=state,
            trajectory_ids=[4, 4],
            return_aux=True,
        )

        self.assertEqual(adapter.router_horizon, 1)
        self.assertEqual(adapter.utility_horizon, 2)
        self.assertIn("moe_utility_scores", output)
        self.assertEqual(output["moe_utility_scores"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["total_action_loss"]).item())

    def test_residual_moe_expert_loss_posterior_accepts_short_router_horizon(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=3,
                lara_top_k=2,
                router_horizon=1,
                utility_horizon=2,
                lara_use_action_loss_utility=True,
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
            actions_are_future=True,
            state=state,
            trajectory_ids=[8],
            return_aux=True,
        )

        self.assertIn("moe_route_distill_loss", output)
        self.assertIn("moe_utility_scores", output)
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

    def test_residual_direct_experts_report_improvement_and_residual_metrics(self):
        torch.manual_seed(0)
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_num_experts=3,
                lara_episode_pool_size=3,
                lara_top_k=2,
                lara_use_direct_action_experts=True,
                lara_direct_expert_action_mode="residual",
                lara_direct_expert_improvement_posterior=True,
                lara_direct_expert_hard_assignment=True,
                lara_direct_expert_posterior_top_r=2,
                lara_direct_expert_improvement_margin=0.0,
                lara_direct_expert_shared_only_gate=True,
                lara_direct_expert_residual_scale=0.5,
                lara_direct_expert_residual_max_norm=0.02,
                lara_direct_expert_residual_warmup_steps=10,
                lara_direct_expert_residual_cost_weight=0.01,
                lara_direct_residual_norm_loss_weight=0.001,
                lara_direct_residual_diversity_loss_weight=0.002,
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
            trajectory_ids=[1, 1],
            return_aux=True,
        )

        self.assertEqual(adapter.direct_expert_action_mode, "residual")
        for key in [
            "moe_shared_action_loss",
            "moe_direct_expert_improvement_mean",
            "moe_direct_expert_improvement_top1",
            "moe_direct_expert_improvement_positive_rate",
            "moe_direct_expert_improvement_candidate_rate",
            "moe_direct_assignment_active_rate",
            "moe_direct_assignment_shared_only_rate",
            "moe_direct_assignment_selected_experts",
            "moe_direct_posterior_entropy",
            "moe_direct_posterior_entropy_p50",
            "moe_direct_posterior_effective_experts",
            "moe_direct_posterior_top1_prob",
            "moe_direct_posterior_top2_mass",
            "moe_direct_posterior_support_size",
            "moe_direct_posterior_usage_0",
            "moe_direct_posterior_usage_1",
            "moe_direct_posterior_usage_2",
            "moe_direct_residual_norm",
            "moe_direct_residual_norm_0",
            "moe_direct_residual_norm_1",
            "moe_direct_residual_norm_2",
            "moe_direct_residual_raw_norm",
            "moe_direct_residual_raw_norm_0",
            "moe_direct_residual_raw_norm_1",
            "moe_direct_residual_raw_norm_2",
            "moe_direct_residual_clamp_rate",
            "moe_direct_residual_clamp_rate_0",
            "moe_direct_residual_clamp_rate_1",
            "moe_direct_residual_clamp_rate_2",
            "moe_direct_residual_scale_base",
            "moe_direct_residual_scale_effective",
            "moe_direct_residual_warmup_fraction",
            "moe_direct_residual_max_norm",
            "moe_direct_residual_norm_loss_weighted",
            "moe_direct_residual_diversity_loss_weighted",
            "moe_direct_residual_regularization_loss",
            "moe_direct_expert_improvement_mean_0",
            "moe_direct_expert_improvement_mean_1",
            "moe_direct_expert_improvement_mean_2",
            "moe_direct_expert_improvement_positive_rate_0",
            "moe_direct_expert_improvement_positive_rate_1",
            "moe_direct_expert_improvement_positive_rate_2",
            "moe_direct_assignment_usage_0",
            "moe_direct_assignment_usage_1",
            "moe_direct_assignment_usage_2",
        ]:
            self.assertIn(key, output)
            self.assertTrue(torch.isfinite(output[key]).all().item())
        expected_total = (
            output["action_loss"]
            + output["moe_loss"]
            + output["moe_direct_expert_loss"]
            + output["moe_direct_residual_regularization_loss"]
        )
        self.assertTrue(torch.allclose(output["total_action_loss"], expected_total))
        self.assertLessEqual(float(output["moe_direct_residual_norm"]), 0.0201)
        self.assertTrue(torch.allclose(output["moe_direct_residual_scale_base"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(output["moe_direct_residual_warmup_fraction"], torch.tensor(0.1)))
        self.assertTrue(torch.allclose(output["moe_direct_residual_scale_effective"], torch.tensor(0.05)))
        self.assertTrue(torch.allclose(output["moe_direct_residual_max_norm"], torch.tensor(0.02)))

    def test_direct_expert_residual_clamp_bounds_per_expert_correction(self):
        adapter = ActionHeadAdapter(
            config=tiny_action_config(
                use_lara_moe=True,
                lara_use_direct_action_experts=True,
                lara_direct_expert_action_mode="residual",
                lara_direct_expert_residual_max_norm=0.5,
            ),
            context_hidden_size=16,
        )

        residuals = torch.ones(2, 3, 4, 2)
        clamped = adapter._clamp_direct_expert_residuals(residuals)
        norms = adapter._expert_residual_norm(clamped)

        self.assertTrue(torch.all(norms <= 0.5001).item())
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, 0.5), atol=1e-4))

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
        self.assertIn("transition_state_loss_raw", output)
        self.assertIn("transition_state_loss_weighted", output)
        self.assertGreater(float(output["transition_state_loss"].detach()), 0.0)
        expected_total = output["action_loss"] + output["transition_state_loss"]
        self.assertTrue(torch.allclose(output["total_action_loss"], expected_total))


if __name__ == "__main__":
    unittest.main()
