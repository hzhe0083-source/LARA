from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_qwen_checks_special_token_counts_before_view():
    src = read("Lara/model/framework/qwen.py")
    qwen3_src = read("Lara/model/modules/vlm/QWen3.py")
    qwen25_src = read("Lara/model/modules/vlm/QWen2_5.py")
    vj2_src = read("Lara/model/framework/vj2.py")
    assert "def _validate_token_mask" in src
    assert "Unexpected {stream_name} token count per sample" in src
    assert "expected_action_token_count" in src
    assert "expected_embodied_action_token_count" in src
    assert "last_hidden[action_mask].view(batch_size, expected_action_count, hidden_size)" in src
    assert 'self.config.framework.vj2_model.get("num_world_model_views", 2)' in vj2_src
    assert "VJ2WorldModel expected {self.num_world_model_views} video views" in vj2_src
    assert 'attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")' in qwen3_src
    assert "attn_implementation=attn_implementation" in qwen3_src
    assert 'attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")' in qwen25_src
    assert "attn_implementation=attn_implementation" in qwen25_src


def test_flow_matching_timestep_buckets_are_clamped():
    for path in [
        "Lara/model/modules/action_model/GR00T_ActionHeader.py",
        "Lara/model/modules/action_model/LayerwiseFM_ActionHeader.py",
    ]:
        src = read(path)
        assert "def discretize_time" in src
        assert "clamp(0, self.num_timestep_buckets - 1)" in src
        assert "self.discretize_time(t[:, 0, 0])" in src


def test_training_entrypoints_share_safe_gradient_accumulation_pattern():
    for path in [
        "Lara/training/train_lara.py",
        "Lara/training/train_lara_video.py",
        "Lara/training/train_lara_cotrain.py",
    ]:
        src = read(path)
        assert "with self.accelerator.accumulate(self.model):\n            self.optimizer.zero_grad()" not in src
        assert "if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:" in src
        assert "self.model, self.optimizer, self.lr_scheduler" in src
        assert "self.lr_scheduler," in src
        assert "self.optimizer.zero_grad(set_to_none=True)" in src
        assert "\n        if accelerator.is_main_process:" not in src
        assert "\n        accelerator.wait_for_everyone()" not in src


def test_so101_batches_expose_future_actions():
    dataset_src = read("Lara/dataloader/gr00t_lerobot/datasets.py")
    v3_dataset_src = read("Lara/dataloader/lerobot_v3_datasets.py")
    core_src = read("Lara/model/framework/Lara_core.py")
    dataloader_src = read("Lara/dataloader/__init__.py")
    trainer_src = read("Lara/training/train_lara.py")
    assert "future_actions=action" in dataset_src
    assert "future_action_mask=self._future_action_mask" in dataset_src
    assert "include_episode_start: bool = False" in dataset_src
    assert 'return_dict["episode_start_image"] = self._episode_start_images' in dataset_src
    assert 'return_dict["current_state"] = state[0:1]' in dataset_src
    assert 'example["future_actions"] = example["action"]' in v3_dataset_src
    assert 'example["future_action_mask"]' in v3_dataset_src
    assert 'example["current_state"] = example["state"]' in v3_dataset_src
    assert 'example["video"] = np.stack(video_views, axis=0)' in v3_dataset_src
    assert "video_horizon: int | None = None" in v3_dataset_src
    assert "video_horizon=cfg.framework.vj2_model.num_frames" in dataloader_src
    assert "class TaskTextDataset" in v3_dataset_src
    assert "def _load_task_map" in v3_dataset_src
    assert 'tasks_parquet = meta_dir / "tasks.parquet"' in v3_dataset_src
    assert 'sample["task"] = self.task_map[task_index]' in v3_dataset_src
    assert 'return_dict["execution_state_target"] = execution_state' in dataset_src
    assert 'return_dict["prediction_state_target"] = prediction_state' in dataset_src
    assert 'example["execution_state_target"] = execution_state' in v3_dataset_src
    assert 'example["prediction_state_target"] = prediction_state' in v3_dataset_src
    assert 'execution_horizon=cfg.framework.action_model.get("execution_horizon", None)' in dataloader_src
    assert "action_horizon=cfg.framework.action_model.action_horizon" in dataloader_src
    assert "trajectory_id=trajectory_name" in dataset_src
    assert "base_index=step" in dataset_src
    assert "def load_counterfactual_utility_label_index" in dataset_src
    assert "counterfactual_utility_labels_path" in dataset_src
    assert "counterfactual_utility_sample_labeled_only" in dataset_src
    assert "self._counterfactual_utility_labeled_step_indices" in dataset_src
    assert 'return_dict["utility_scores"] = utility_label["utility_scores"].copy()' in dataset_src
    assert 'return_dict["utility_candidate_mask"] = utility_label["utility_candidate_mask"].copy()' in dataset_src
    assert '"future_actions" if "future_actions" in examples[0] else "action"' in core_src
    assert 'actions_are_future = action_key == "future_actions"' in core_src
    assert 'action_mask_key = "future_action_mask" if "future_action_mask" in examples[0] else "action_mask"' in core_src
    assert "action_mask=action_mask" in core_src
    assert 'past_actions = optional_batch_field("past_actions")' in core_src
    assert "actions_are_future=actions_are_future" in core_src
    assert "past_actions=past_actions" in core_src
    assert '"current_state" if "current_state" in examples[0] else "state"' in core_src
    assert '"future_actions" if "future_actions" in examples[0] else "action"' in trainer_src
    assert '"current_state" if "current_state" in examples[0] else "state"' in trainer_src
    assert "def optional_batch_field" in core_src
    assert 'trajectory_ids = optional_batch_field("trajectory_id")' in core_src
    assert "trajectory_ids=trajectory_ids" in core_src
    assert 'utility_scores = optional_batch_field("utility_scores")' in core_src
    assert 'utility_value_targets = optional_batch_field("utility_value_targets")' in core_src
    assert 'previous_router_probs = optional_batch_field("previous_router_probs")' in core_src
    assert 'episode_start_images = optional_batch_field("episode_start_image")' in core_src
    assert "self.action_head.lara_moe is not None and episode_start_images is not None" in core_src
    assert "initial_context_tokens = self.action_head.conditioning_tokens" in core_src
    assert "utility_scores=utility_scores" in core_src
    assert "utility_value_targets=utility_value_targets" in core_src
    assert "previous_router_probs=previous_router_probs" in core_src
    assert "initial_context_tokens=initial_context_tokens" in core_src
    assert "execution_state_target=execution_state_target" in core_src
    assert "prediction_state_target=prediction_state_target" in core_src


def test_benchmark_dataset_entrypoints_are_explicit():
    mixture_src = read("Lara/dataloader/gr00t_lerobot/mixtures.py")
    v3_dataset_src = read("Lara/dataloader/lerobot_v3_datasets.py")
    downloader_src = read("scripts/download_benchmark_data.py")
    libero_cfg = read("scripts/config/lara_libero100_baseline.yaml")
    metaworld_cfg = read("scripts/config/lara_metaworld_mt50_baseline.yaml")
    readme = read("README.md")
    assert '"libero100"' in mixture_src
    assert '("kevin_libero100_lerobot", 1.0, "libero_franka")' in mixture_src
    assert '"metaworld_mt50"' in mixture_src
    assert '("lerobot_metaworld_mt50", 1.0, "metaworld")' in mixture_src
    assert 'feature.get("dtype") == "image"' in v3_dataset_src
    assert "Configured image keys are missing" in v3_dataset_src
    assert "Configured action key" in v3_dataset_src
    assert "Configured state key" in v3_dataset_src
    assert "action_horizon: int | None = None" in v3_dataset_src
    assert "video_horizon: int | None = None" in v3_dataset_src
    assert 'data_cfg.get("action_horizon", 60)' in v3_dataset_src
    assert 'data_cfg.get("video_horizon", 8)' in v3_dataset_src
    assert "range(video_horizon)" in v3_dataset_src
    assert "range(action_horizon+1)" not in v3_dataset_src
    assert "kevin-ys-zhang/libero100_lerobot" in downloader_src
    assert "lerobot/metaworld_mt50" in downloader_src
    assert 'DEFAULT_INCLUDE_PATTERNS = ["meta/**", "data/chunk-*/*.parquet"]' in downloader_src
    assert "expected_data_parquet_files=279" in downloader_src
    assert "expected_data_parquet_files=492" in downloader_src
    assert "expected {dataset.expected_data_parquet_files} data chunk parquet files" in downloader_src
    assert "HF_HUB_DISABLE_XET" in downloader_src
    assert "chunk_parquet_files" in downloader_src
    assert "missing data/chunk-*/*.parquet" in downloader_src
    assert "missing meta/tasks.parquet or meta/tasks.jsonl" in downloader_src
    assert "--allow-incomplete" in downloader_src
    assert '(not report["ready"]) and (not args.allow_incomplete)' in downloader_src
    assert "dataset_py: lerobot_v3_datasets" in libero_cfg
    assert "data_mix: libero100" in libero_cfg
    assert "state_dim: 9" in libero_cfg
    assert "action_dim: 7" in libero_cfg
    assert "use_lara_moe: false" in libero_cfg
    assert "data_mix: metaworld_mt50" in metaworld_cfg
    assert "state_dim: 4" in metaworld_cfg
    assert "action_dim: 4" in metaworld_cfg
    assert "use_lara_moe: false" in metaworld_cfg
    assert "scripts/download_benchmark_data.py --dataset all --preflight-only" in readme
    assert "exits nonzero while any selected dataset is incomplete" in readme
    assert "--allow-incomplete" in readme


def test_action_head_future_windows_are_strict():
    adapter_src = read("Lara/model/framework/act.py")
    assert "actions_are_future: bool = False" in adapter_src
    assert "action_mask=None" in adapter_src
    assert "past_actions=None" in adapter_src
    assert "def _action_mask_to_tensor" in adapter_src
    assert "action_mask_target = self._action_mask_to_tensor(action_mask, actions, actions_are_future)" in adapter_src
    assert "actions_are_future and actions.shape[1] != self.action_horizon" in adapter_src
    assert "future_actions must have exactly {self.action_horizon} steps" in adapter_src
    assert "not actions_are_future and actions.shape[1] < self.action_horizon" in adapter_src
    assert "actions if actions_are_future else actions[:, -self.action_horizon :, :]" in adapter_src


def test_aux_action_losses_are_not_double_counted_by_trainer():
    core_src = read("Lara/model/framework/Lara_core.py")
    trainer_src = read("Lara/training/train_lara.py")
    trainer_tools_src = read("Lara/training/trainer_utils/trainer_tools.py")
    assert '"action_loss": action_output["total_action_loss"]' in core_src
    assert 'output[f"metric/{key}"] = value.detach()' in core_src
    assert '"metric/wm_loss_raw"' in core_src
    assert '"metric/wm_loss_weight"' in core_src
    assert "self._loss_scale(\"wm\", fallback_key=\"vlm\", default=1.0)" in core_src
    assert "split_loss_and_metric_outputs(output_dict)" in trainer_src
    assert "total_loss = sum(loss_dict.values())" in trainer_src
    assert "def split_loss_and_metric_outputs" in trainer_tools_src
    assert "def action_eval_metrics" in trainer_tools_src
    assert "self._get_next_eval_batch()" in trainer_src
    assert "eval/skipped_no_eval_dataloader" in trainer_src
    assert "eval/full_horizon_mae" in trainer_src
    assert "eval/execution_horizon_mae" in trainer_src


def test_optional_latent_action_head_stage_one_exists():
    latent_src = read("Lara/model/modules/action_model/lara_latent.py")
    adapter_src = read("Lara/model/framework/act.py")
    config_src = read("scripts/config/lara_so101_ft.yaml")
    latent_cfg = read("scripts/config/lara_so101_latent_vq.yaml")
    assert "class PosteriorLatentActionEncoder" in latent_src
    assert "class VectorQuantizer" in latent_src
    assert "class LatentActionPrior" in latent_src
    assert "class LatentActionDecoder" in latent_src
    assert "class LatentActionHead" in latent_src
    assert "class LatentActionTransitionHead" in latent_src
    assert "self.latent_action_horizon" in adapter_src
    assert "self.router_horizon" in adapter_src
    assert "self.utility_horizon" in adapter_src
    assert '"latent_action_horizon"' in adapter_src
    assert '"router_horizon"' in adapter_src
    assert '"utility_horizon"' in adapter_src
    assert "latent_actions_target = actions_target[:, : self.latent_action_horizon, :]" in adapter_src
    assert "future_action_mask=latent_action_mask" in adapter_src
    assert "router_actions_target = actions_target[:, : self.router_horizon, :]" in adapter_src
    assert "utility_actions_target = actions_target[:, : self.utility_horizon, :]" in adapter_src
    assert "use_latent_action_head" in adapter_src
    assert "self.latent_action_head.predict" in adapter_src
    assert "latent_action_reconstruction_loss" in adapter_src
    assert "latent_action_reconstruction_loss_weighted" in adapter_src
    assert "latent_action_code_usage_loss" in adapter_src
    assert "latent_action_horizon: 10" in config_src
    assert "router_horizon: 10" in config_src
    assert "utility_horizon: 10" in config_src
    assert "self.transition_head" in adapter_src
    assert "transition_state_loss" in adapter_src
    assert "use_latent_action_head: false" in config_src
    assert "use_latent_action_head: true" in latent_cfg
    assert "lara_latent_reconstruction_loss_weight: 1.0" in latent_cfg
    assert "lara_use_transition_head: false" in config_src
    assert "lara_transition_loss_weight: 0.0" in config_src
    assert "lara_code_usage_loss_weight: 0.0" in config_src
    assert "lara_code_usage_temperature: 1.0" in config_src


def test_optional_moe_router_stage_two_exists():
    moe_src = read("Lara/model/modules/action_model/lara_moe.py")
    adapter_src = read("Lara/model/framework/act.py")
    core_src = read("Lara/model/framework/Lara_core.py")
    server_src = read("deployment/model_server/tools/websocket_policy_server.py")
    server_policy_src = read("deployment/model_server/server_policy.py")
    protocol_src = read("Lara/evaluation/lara_protocol.py")
    protocol_cli_src = read("scripts/summarize_lara_protocol.py")
    utility_cli_src = read("scripts/build_counterfactual_utility_labels.py")
    readiness_cli_src = read("scripts/audit_lara_paper_readiness.py")
    smoke_src = read("scripts/smoke_lara_real_components.py")
    config_src = read("scripts/config/lara_so101_ft.yaml")
    moe_cfg = read("scripts/config/lara_so101_moe_direct.yaml")
    utility_cfg = read("scripts/config/lara_so101_utility_pool.yaml")
    flow_src = read("Lara/model/modules/action_model/GR00T_ActionHeader.py")
    gap = read("document/IMPLEMENTATION_GAP.md")
    assert "class LatentActionMoE" in moe_src
    assert "class ActionChunkExpertBank" in moe_src
    assert "class RouteUtilityHead" in moe_src
    assert "class ResidentPoolOutput" in moe_src
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
    assert "def expert_diversity_loss" in moe_src
    assert "def route_entropy_regularization_loss" in moe_src
    assert "def route_switch_rate" in moe_src
    assert "def retained_probability_mass" in moe_src
    assert "def spearman_rank_correlation" in moe_src
    assert "def kendall_rank_correlation" in moe_src
    assert "def topk_route_consistency" in moe_src
    assert "def route_regret_from_scores" in moe_src
    assert "def posterior_router_kl" in moe_src
    assert "def pool_coverage_objective" in moe_src
    assert "def pool_coverage_diagnostics" in moe_src
    assert "def route_quality_metrics" in moe_src
    assert "def sparse_route_budget" in moe_src
    assert "def forced_router_probs_from_scores" in moe_src
    assert "def subset_retention_success_curve" in protocol_src
    assert "def subset_retention_rows" in protocol_src
    assert "def matched_compute_row" in protocol_src
    assert "def matched_budget_flags" in protocol_src
    assert "def matched_expert_budget_flags" in protocol_src
    assert "def pareto_frontier_flags" in protocol_src
    assert "def protocol_summary_from_records" in protocol_src
    assert "def protocol_evidence_audit" in protocol_src
    assert "def counterfactual_utility_records_from_rollouts" in protocol_src
    assert "def counterfactual_utility_matrix_from_records" in protocol_src
    assert "def step_context_id" in protocol_src
    assert "trajectory_key: str = \"trajectory_id\"" in protocol_src
    assert "base_index_key: str = \"base_index\"" in protocol_src
    assert '"utility_candidate_mask"' in protocol_src
    assert "min_candidates_per_context" in protocol_src
    assert "PAPER_REQUIRED_METRIC_KEYS" in protocol_src
    assert "def route_sequence_diagnostics" in protocol_src
    assert "def rollout_record_with_route_diagnostics" in protocol_src
    assert "def normalize_protocol_records" in protocol_src
    assert "router_probs_sequence" in protocol_src
    assert "route_diagnostics_by_fraction" in protocol_src
    assert "protocol_summary_from_records" in protocol_cli_src
    assert "protocol_evidence_audit" in protocol_cli_src
    assert "--resident-fraction-key" in protocol_cli_src
    assert "--no-route-sequence-diagnostics" in protocol_cli_src
    assert "--require-paper-metrics" in protocol_cli_src
    assert "--required-resident-fractions" in protocol_cli_src
    assert "JSON or JSONL rollout records" in protocol_cli_src
    assert "counterfactual_utility_records_from_rollouts" in utility_cli_src
    assert "counterfactual_utility_matrix_from_records" in utility_cli_src
    assert "--min-candidates-per-context" in utility_cli_src
    assert "def audit_lara_paper_readiness" in readiness_cli_src
    assert "baseline_defaults_safe" in readiness_cli_src
    assert "so101_horizon_contract" in readiness_cli_src
    assert "counterfactual_utility_sidecar" in readiness_cli_src
    assert "closed_loop_protocol_records" in readiness_cli_src
    assert "full_so101_training_artifact" in readiness_cli_src
    assert "closed_loop_robot_eval_artifact" in readiness_cli_src
    assert "counterfactual_utility_matrix_from_records" in readiness_cli_src
    assert "protocol_evidence_audit" in readiness_cli_src
    assert "--allow-incomplete" in readiness_cli_src
    assert "--min-training-steps" in readiness_cli_src
    assert "--min-robot-eval-episodes" in readiness_cli_src
    assert "uses_real_so101_data" in readiness_cli_src
    assert "has_counterfactual_utility_eval" in readiness_cli_src
    assert "checkpoint_path must point to an existing checkpoint" in readiness_cli_src
    assert "def smoke_lara_real_components" in smoke_src
    assert "def _exception_status" in smoke_src
    assert "def apply_smoke_overrides" in smoke_src
    assert "def build_real_examples" in smoke_src
    assert "def summarize_examples" in smoke_src
    assert "def place_smoke_trainable_components" in smoke_src
    assert "def smoke_optimizer_parameters" in smoke_src
    assert 'for attr in ("vj2", "action_head")' in smoke_src
    assert "--instantiate" in smoke_src
    assert "--run-step" in smoke_src
    assert "--optimizer-step" in smoke_src
    assert "--optimizer-lr" in smoke_src
    assert "--use-real-batch" in smoke_src
    assert "--real-batch-size" in smoke_src
    assert "--attn-implementation" in smoke_src
    assert "--use-latent-action-head" in smoke_src
    assert "--use-transition-head" in smoke_src
    assert "--transition-loss-weight" in smoke_src
    assert "--use-lara-moe" in smoke_src
    assert "--use-direct-action-experts" in smoke_src
    assert "--use-direct-action-output" in smoke_src
    assert "--use-action-loss-utility-components" in smoke_src
    assert "--use-state-utility" in smoke_src
    assert "--use-state-utility-components" in smoke_src
    assert "--counterfactual-utility-labels-path" in smoke_src
    assert "counterfactual_utility_sample_labeled_only" in smoke_src
    assert "num_utility_experts=cfg.framework.action_model.get(\"lara_num_experts\", None)" in smoke_src
    assert "lara_use_direct_action_experts" in smoke_src
    assert "lara_use_direct_action_output" in smoke_src
    assert "lara_use_action_loss_utility_components" in smoke_src
    assert "lara_use_state_utility" in smoke_src
    assert "lara_use_state_utility_components" in smoke_src
    assert "use_direct_action_output" in smoke_src
    assert "def candidate_route_utility" in moe_src
    assert "def utility_from_expert_losses" in moe_src
    assert "def uncertainty_from_expert_losses" in moe_src
    assert "def utility_component_targets_from_expert_losses" in moe_src
    assert "def aggregate_episode_responsibilities" in moe_src
    assert "def routed_actions" in moe_src
    assert "def action_chunk_loss" in moe_src
    assert "def reconstruction_loss_components" in moe_src
    assert "def utility_calibration_objective" in moe_src
    assert "def utility_component_supervision_loss" in moe_src
    assert "expert_action_losses" in moe_src
    assert "pool_target_probs" in moe_src
    assert "utility_scores" in moe_src
    assert "lara_use_utility_head" in adapter_src
    assert "lara_inference_stickiness_weight" in adapter_src
    assert "inference_stickiness_weight" in moe_src
    assert "route_quality_metrics" in adapter_src
    assert "moe_route_quality_" in adapter_src
    assert "utility_from_expert_losses" in adapter_src
    assert "utility_component_targets_from_expert_losses" in adapter_src
    assert "self.use_state_utility" in adapter_src
    assert "def _expert_transition_loss_components" in adapter_src
    assert "moe_state_utility_error" in adapter_src
    assert "def expert_conditioning_tokens" in moe_src
    assert "self.direct_action_experts" in adapter_src
    assert "self.use_direct_action_output" in adapter_src
    assert 'state_dim=action_cfg.get("state_dim", None)' in adapter_src
    assert "direct_action_experts(conditioning_tokens, state=state_tensor)" in adapter_src
    assert "direct_expert_actions[:, :, : self.router_horizon, :]" in adapter_src
    assert "direct_expert_actions[:, :, : self.utility_horizon, :]" in adapter_src
    assert "moe_direct_expert_loss" in adapter_src
    assert "moe_direct_routed_action_loss" in adapter_src
    assert "def _trajectory_ids_to_tensor" in adapter_src
    assert "aggregate_episode_responsibilities(posterior_probs, trajectory_tensor)" in adapter_src
    assert "def select_resident_pool" in moe_src
    assert "def select_resident_pool" in adapter_src
    assert "pool_mask=pool_mask" in adapter_src
    assert "resident_pool_mask" in core_src
    assert '"resident_pool_mask"' in core_src
    assert "previous_router_probs" in core_src
    assert "forced_expert_id" in core_src
    assert "forced_router_probs" in core_src
    assert '"router_probs"' in core_src
    assert '"active_expert_mask"' in core_src
    assert "self._session_state" in server_src
    assert "resident_pool_mask" in server_src
    assert "previous_router_probs" in server_src
    assert '"router_probs"' in server_src
    assert 'mtype == "reset"' in server_src
    assert "rollout_trace_path" in server_src
    assert "--rollout_trace_path" in server_policy_src
    assert "rollout_trace_path=args.rollout_trace_path" in server_policy_src
    assert "record_outcome" in server_src
    assert "router_probs_sequence" in server_src
    assert "active_mask_sequence" in server_src
    assert "pool_mask_sequence" in server_src
    assert "forced_expert_id_sequence" in server_src
    assert "latency_ms_sequence" in server_src
    assert "vram_mb_sequence" in server_src
    assert "reset_peak_memory_stats" in server_src
    assert "has_training_teacher = expert_action_losses is not None or latent_action_tokens is not None" in moe_src
    assert "episode_pool_size" in moe_src
    assert "episode_pool_size_min" in moe_src
    assert "def _episode_pool_top_k" in moe_src
    assert "budget_features" in moe_src
    assert "self.budget_proj" in moe_src
    assert "def _pool_budget_features" in moe_src
    assert "def _pool_size_from_mask" in moe_src
    assert "active_mask = topk_mask(router_logits, top_k=self.top_k, allowed_mask=pool_mask)" in moe_src
    assert "router_probs, active_mask = forced_router_probs_from_scores" in moe_src
    assert "pool_loss_weight" in moe_src
    assert "posterior_temperature" in moe_src
    assert "posterior_uniform_floor" in moe_src
    assert "posterior_top_r" in moe_src
    assert "reduction: str = \"mean\"" in flow_src
    assert "action_mask: torch.Tensor = None" in flow_src
    assert "reduction=\"none\"" in adapter_src
    assert "def _expert_action_losses" in adapter_src
    assert "with torch.no_grad()" in adapter_src
    assert "use_lara_moe" in adapter_src
    assert "moe_router_loss" in adapter_src
    assert "moe_loss" in adapter_src
    assert "moe_route_distill_loss_raw" in adapter_src
    assert "moe_route_distill_loss_weighted" in adapter_src
    assert "moe_utility_loss_weighted" in adapter_src
    assert "moe_pool_distill_loss_weighted" in adapter_src
    assert "moe_pool_distill_loss" in adapter_src
    assert "moe_pool_coverage_loss_weighted" in adapter_src
    assert "moe_pool_teacher_mass" in adapter_src
    assert "moe_pool_critical_miss_rate" in adapter_src
    assert "moe_balance_loss" in adapter_src
    assert "moe_stickiness_loss" in adapter_src
    assert "moe_diversity_loss" in adapter_src
    assert "moe_entropy_loss" in adapter_src
    assert "moe_dead_expert_ratio" in adapter_src
    assert "moe_route_regret" in adapter_src
    assert "use_lara_moe: false" in config_src
    assert "lara_episode_pool_size: 4" in config_src
    assert "lara_episode_pool_size_min:" in config_src
    assert "lara_pool_target_avg_weight: 1.0" in config_src
    assert "lara_pool_target_max_weight: 1.0" in config_src
    assert "lara_pool_target_utility_weight: 0.0" in config_src
    assert "lara_pool_coverage_loss_weight: 0.0" in config_src
    assert "lara_pool_critical_threshold: 0.2" in config_src
    assert "lara_utility_loss_weight: 0.0" in config_src
    assert "lara_utility_rank_loss_weight: 0.0" in config_src
    assert "lara_utility_head_loss_weight: 0.0" in config_src
    assert "lara_use_action_loss_utility: false" in config_src
    assert "lara_use_action_loss_utility_components: false" in config_src
    assert "lara_use_state_utility: false" in config_src
    assert "lara_use_state_utility_components: false" in config_src
    assert "lara_state_utility_temperature: 1.0" in config_src
    assert "lara_state_utility_normalize: true" in config_src
    assert "lara_route_retention_fractions: [0.25, 0.5, 1.0]" in config_src
    assert "lara_use_utility_head: false" in config_src
    assert "lara_balance_loss_weight: 0.0" in config_src
    assert "lara_stickiness_loss_weight: 0.0" in config_src
    assert "lara_diversity_loss_weight: 0.0" in config_src
    assert "lara_entropy_loss_weight: 0.0" in config_src
    assert "lara_inference_stickiness_weight: 0.0" in config_src
    assert "lara_use_direct_action_experts: false" in config_src
    assert "lara_use_direct_action_output: false" in config_src
    assert "lara_direct_expert_loss_weight: 1.0" in config_src
    assert "lara_posterior_temperature: 1.0" in config_src
    assert "lara_posterior_uniform_floor: 0.0" in config_src
    assert "lara_posterior_top_r:" in config_src
    assert "lara_use_expert_loss_posterior: true" in config_src
    assert "include_episode_start: false" in config_src
    assert "counterfactual_utility_labels_path:" in config_src
    assert "counterfactual_utility_min_candidates_per_context: 2" in config_src
    assert "counterfactual_utility_sample_labeled_only: false" in config_src
    assert "use_lara_moe: true" in moe_cfg
    assert "lara_use_direct_action_experts: true" in moe_cfg
    assert "lara_use_direct_action_output: true" in moe_cfg
    assert "use_lara_moe: true" in utility_cfg
    assert "lara_utility_loss_weight: 1.0" in utility_cfg
    assert "lara_utility_rank_loss_weight: 0.25" in utility_cfg
    assert "include_episode_start: true" in utility_cfg
    assert "Stage-2 MoE/router scaffold" in gap
    assert "expert-diversity/entropy stabilizers" in gap
    assert "route-quality aggregation metrics" in gap
    assert "Action-head route-quality metrics are emitted" in gap
    assert "matched-compute rows, matched-resident rows, budget-match flags, Pareto frontier flags" in gap
    assert "scripts/summarize_lara_protocol.py" in gap
    assert "scripts/audit_lara_paper_readiness.py" in gap
    assert "structured JSON with required paper-stage flags" in gap
    assert "route-sequence diagnostics" in gap
    assert "training-time randomized resident-pool size" in gap
    assert "scripts/smoke_lara_real_components.py" in gap
    assert "structured error reporting" in gap
    assert "caches MoE `resident_pool_mask` and `router_probs` values per `session_id`" in gap


def test_paper_gap_is_explicit():
    readme = read("README.md")
    gap = read("document/IMPLEMENTATION_GAP.md")
    baseline_cfg = read("scripts/config/lara_so101_baseline.yaml")
    assert "implemented the SO101 VLA/action-baseline path and several default-off scaffolds" in readme
    assert "paper's latent-action MoE and two-level routing method should still be treated as unfinished" in readme
    assert "not the final latent-action MoE/router implementation" in readme
    assert "Experimental scaffolding exists but is not complete or validated" in readme
    assert "padded future action steps do not supervise" in readme
    assert "scripts/summarize_lara_protocol.py" in readme
    assert "Missing Paper Components" in gap
    assert "MoE action experts" in gap
    assert "Do not describe it as the completed LARA MoE or completed two-level router yet" in gap
    assert "future_action_mask" in gap
    assert "Trajectory ids are now passed" in gap
    assert "chunk-level top-k routing constrained to the resident pool" in gap
    assert "coverage-aware episode-level pool target aggregation" in gap
    assert "critical expert miss rate" in gap
    assert "per-expert action-loss posterior path" in gap
    assert "action-loss utility labels" in gap
    assert "transition-state consistency utility labels" in gap
    assert "rollout-record summarization" in gap
    assert "closed-loop route diagnostics" in gap
    assert "scripts/config/lara_so101_baseline.yaml" in readme
    assert "scripts/config/lara_so101_latent_vq.yaml" in readme
    assert "scripts/config/lara_so101_moe_direct.yaml" in readme
    assert "scripts/config/lara_so101_utility_pool.yaml" in readme
    assert "use_lara_moe: false" in baseline_cfg
