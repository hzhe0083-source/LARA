from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_qwen_checks_special_token_counts_before_view():
    src = read("Lara/model/framework/qwen.py")
    assert "def _validate_token_mask" in src
    assert "Unexpected {stream_name} token count per sample" in src
    assert "expected_action_token_count" in src
    assert "expected_embodied_action_token_count" in src
    assert "last_hidden[action_mask].view(batch_size, expected_action_count, hidden_size)" in src


def test_flow_matching_timestep_buckets_are_clamped():
    for path in [
        "Lara/model/modules/action_model/GR00T_ActionHeader.py",
        "Lara/model/modules/action_model/LayerwiseFM_ActionHeader.py",
    ]:
        src = read(path)
        assert "def discretize_time" in src
        assert "clamp(0, self.num_timestep_buckets - 1)" in src
        assert "self.discretize_time(t[:, 0, 0])" in src


def test_train_lara_gradient_accumulation_pattern():
    src = read("Lara/training/train_lara.py")
    assert "with self.accelerator.accumulate(self.model):\n            self.optimizer.zero_grad()" not in src
    assert "if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:" in src
    assert "self.optimizer.zero_grad(set_to_none=True)" in src


def test_so101_batches_expose_future_actions():
    dataset_src = read("Lara/dataloader/gr00t_lerobot/datasets.py")
    core_src = read("Lara/model/framework/Lara_core.py")
    assert "future_actions=action" in dataset_src
    assert 'return_dict["current_state"] = state[0:1]' in dataset_src
    assert "trajectory_id=trajectory_name" in dataset_src
    assert "base_index=step" in dataset_src
    assert '"future_actions" if "future_actions" in examples[0] else "action"' in core_src
    assert '"current_state" if "current_state" in examples[0] else "state"' in core_src
    assert 'trajectory_ids = [example["trajectory_id"] for example in examples] if "trajectory_id" in examples[0] else None' in core_src
    assert "trajectory_ids=trajectory_ids" in core_src


def test_aux_action_losses_are_not_double_counted_by_trainer():
    core_src = read("Lara/model/framework/Lara_core.py")
    assert '"action_loss": action_output["total_action_loss"]' in core_src
    assert "for key, value in action_output.items()" not in core_src


def test_optional_latent_action_head_stage_one_exists():
    latent_src = read("Lara/model/modules/action_model/lara_latent.py")
    adapter_src = read("Lara/model/framework/act.py")
    config_src = read("scripts/config/lara_so101_ft.yaml")
    assert "class PosteriorLatentActionEncoder" in latent_src
    assert "class VectorQuantizer" in latent_src
    assert "class LatentActionPrior" in latent_src
    assert "class LatentActionHead" in latent_src
    assert "use_latent_action_head" in adapter_src
    assert "self.latent_action_head.predict" in adapter_src
    assert "use_latent_action_head: false" in config_src


def test_optional_moe_router_stage_two_exists():
    moe_src = read("Lara/model/modules/action_model/lara_moe.py")
    adapter_src = read("Lara/model/framework/act.py")
    config_src = read("scripts/config/lara_so101_ft.yaml")
    flow_src = read("Lara/model/modules/action_model/GR00T_ActionHeader.py")
    gap = read("document/IMPLEMENTATION_GAP.md")
    assert "class LatentActionMoE" in moe_src
    assert "class ActionChunkExpertBank" in moe_src
    assert "class PosteriorResponsibilityHead" in moe_src
    assert "class ChunkRouter" in moe_src
    assert "class EpisodePoolRouter" in moe_src
    assert "def masked_topk_softmax" in moe_src
    assert "def topk_mask" in moe_src
    assert "def masked_kl_div" in moe_src
    assert "def posterior_from_expert_losses" in moe_src
    assert "def route_diagnostics" in moe_src
    assert "def uniform_balance_loss" in moe_src
    assert "def route_stickiness_loss" in moe_src
    assert "def route_switch_rate" in moe_src
    assert "def retained_probability_mass" in moe_src
    assert "def candidate_route_utility" in moe_src
    assert "def aggregate_episode_responsibilities" in moe_src
    assert "def utility_calibration_objective" in moe_src
    assert "expert_action_losses" in moe_src
    assert "pool_target_probs" in moe_src
    assert "utility_scores" in moe_src
    assert "def expert_conditioning_tokens" in moe_src
    assert "self.direct_action_experts" in adapter_src
    assert "moe_direct_expert_loss" in adapter_src
    assert "def _trajectory_ids_to_tensor" in adapter_src
    assert "aggregate_episode_responsibilities(posterior_probs, trajectory_tensor)" in adapter_src
    assert "has_training_teacher = expert_action_losses is not None or latent_action_tokens is not None" in moe_src
    assert "episode_pool_size" in moe_src
    assert "active_mask = topk_mask(router_logits, top_k=self.top_k, allowed_mask=pool_mask)" in moe_src
    assert "pool_loss_weight" in moe_src
    assert "posterior_temperature" in moe_src
    assert "reduction: str = \"mean\"" in flow_src
    assert "reduction=\"none\"" in adapter_src
    assert "def _expert_action_losses" in adapter_src
    assert "with torch.no_grad()" in adapter_src
    assert "use_lara_moe" in adapter_src
    assert "moe_router_loss" in adapter_src
    assert "moe_pool_distill_loss" in adapter_src
    assert "moe_balance_loss" in adapter_src
    assert "moe_stickiness_loss" in adapter_src
    assert "moe_dead_expert_ratio" in adapter_src
    assert "moe_route_regret" in adapter_src
    assert "use_lara_moe: false" in config_src
    assert "lara_episode_pool_size: 4" in config_src
    assert "lara_utility_loss_weight: 0.0" in config_src
    assert "lara_utility_rank_loss_weight: 0.0" in config_src
    assert "lara_balance_loss_weight: 0.0" in config_src
    assert "lara_stickiness_loss_weight: 0.0" in config_src
    assert "lara_use_direct_action_experts: false" in config_src
    assert "lara_direct_expert_loss_weight: 1.0" in config_src
    assert "lara_posterior_temperature: 1.0" in config_src
    assert "lara_use_expert_loss_posterior: true" in config_src
    assert "Stage-2 MoE/router scaffold" in gap


def test_paper_gap_is_explicit():
    readme = read("README.md")
    gap = read("document/IMPLEMENTATION_GAP.md")
    assert "only implemented the VLA/action-baseline part" in readme
    assert "not the final latent-action MoE/router implementation" in readme
    assert "Experimental scaffolding exists but is not complete or validated" in readme
    assert "Missing Paper Components" in gap
    assert "MoE action experts" in gap
    assert "Trajectory ids are now passed" in gap
    assert "chunk-level top-k routing constrained to the resident pool" in gap
    assert "pool target aggregation" in gap
    assert "per-expert action-loss posterior path" in gap
