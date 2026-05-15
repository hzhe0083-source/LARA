# LARA Implementation Gap

This repository currently implements the SO101 VLA-JEPA action baseline and default-off research scaffolds, not the full LARA paper method. The MoE and two-level routing code paths are not yet paper-complete.

## Implemented Baseline

- SO101 LeRobot dataset wiring.
- VLA-JEPA Pretrain checkpoint loading.
- Qwen/V-JEPA latent action tokens and embodied action tokens.
- Latent-token conditioned flow-matching action head.
- Prediction horizon `H_p = 60` and execution horizon `H_e = 10` configuration.
- Weighted action loss for executable prefix versus long-horizon tail.
- `future_action_mask` propagation for SO101/LeRobot batches so padded future steps at trajectory boundaries do not supervise action, latent, transition, utility-proxy, or direct-expert reconstruction losses.

## Experimental Scaffolding

These files exist to make the next implementation steps concrete, but they are not complete LARA components and are disabled by default. They should be read as scaffolding and diagnostic wiring, not as evidence that the paper method is finished:

- Stage-1 latent action head scaffold: posterior encoder, VQ codebook, optional code-usage regularization, context-only prior, and optional execution/prediction boundary-state transition loss (`use_latent_action_head: false`, `lara_use_transition_head: false`).
- Stage-2 MoE/router scaffold: residual token experts, optional direct action-chunk experts, optional routed direct-expert action output, posterior responsibility from latent tokens or per-expert action reconstruction losses, optional posterior floor/top-r smoothing, episode-level resident pool targets from aggregated chunk responsibility, optional episode-start image encoding for the resident pool router, reusable episode-level resident pool masks, budget-conditioned episode pool routing, optional training-time randomized resident-pool size, chunk-level top-k routing constrained to the resident pool, optional inference stickiness, posterior-to-router distillation losses, balance/stickiness/expert-diversity/entropy stabilizers, and route-quality aggregation metrics (`use_lara_moe: false`).
- Utility calibration scaffold: optional action-loss utility labels, optional transition-state consistency utility labels, optional direct-expert action reconstruction or transition-state component labels for value/progress/uncertainty targets, optional dataset-provided utility/candidate/cost/component targets, strict counterfactual rollout-record to `utility_scores` / `utility_candidate_mask` matrix conversion, optional supervised route utility head, candidate value/progress/uncertainty/cost scoring helpers, centered utility regression, and pairwise ranking losses (`lara_utility_loss_weight: 0.0`, `lara_utility_head_loss_weight: 0.0`, `lara_use_action_loss_utility: false`, `lara_use_state_utility: false`, `lara_use_utility_head: false`).

## Missing Paper Components

- Production-ready latent action training and validation.
- Validated transition-state training with real SO101 boundary targets.
- Validated MoE action experts that directly produce or adapt action chunks in full SO101 training.
- Validated two-level routing: episode-level resident pool selection, chunk-level top-k routing inside that pool, and resident-pool reuse across real SO101 rollout horizons.
- closed-loop route diagnostics tied to real robot outcomes and subset-retention success curves.
- Real counterfactual utility scoring from latent-state or closed-loop evaluator signals beyond action reconstruction labels. The code can reject single-route records and build utility matrices from true multi-candidate logs, but the actual SO101 counterfactual evaluator labels still need to be produced.
- Validated matched-compute and matched-resident-expert experiments with real FLOPs, latency, VRAM, and rollout success.

## Baseline Reliability Fixes Applied

- VLA, video-only, and VLA/video co-training loops no longer zero gradients at the start of every `accelerator.accumulate` block; gradient clipping is gated on `accelerator.sync_gradients`; learning-rate schedulers are prepared by Accelerate alongside the optimizer so accumulation-aware optimizer skips do not leave a raw scheduler stepping on every micro-batch; checkpoint waits now use the trainer's accelerator instance.
- Distributed barriers and rank checks are guarded for single-process execution.
- Qwen latent/embodied special token counts are checked per sample before hidden-state reshaping and covered by a lightweight unit test.
- Qwen3/Qwen2.5 wrappers now respect `framework.qwenvl.attn_implementation`, and the real-component smoke script can temporarily override it for non-FlashAttention environments.
- V-JEPA world-model view count is explicit through `num_world_model_views`; SO101 single-camera batches and smoke dummy batches duplicate to two view streams before the predictor.
- Real-component `--run-step` smoke checks now mirror trainer/server device placement for V-JEPA and the action head instead of leaving them on CPU beside a CUDA Qwen model.
- Flow-matching timestep buckets are clamped to the valid range.
- `ActionHeadAdapter` has a dummy-batch forward/predict smoke test for basic loss and output shape.
- SO101 batches now expose `future_actions` and `current_state` explicitly, with `action` and `state` retained as compatibility aliases.
- `future_actions` is now treated as a strict future-only target and must match `action_horizon`; only the legacy `action` fallback is tail-sliced.
- SO101/LeRobot batches now expose `future_action_mask`; the action adapter passes it into flow-matching losses, latent-action posterior pooling, transition/action utility proxies, and direct-expert reconstruction losses so padded trajectory-end steps do not become training targets.
- The LeRobot v3 collator can be imported without the optional `lerobot` package and has unit coverage for explicit `future_actions/current_state` aliases.
- Action labels remain fp32 in the adapter.
- Static pytest coverage was added for the baseline guardrails in `tests/test_baseline_static.py`.
- Stage-1 latent action head code was added behind `use_latent_action_head`; posterior/VQ/prior train on the first `latent_action_horizon` steps of the explicit future action window, keeping the latent code aligned with the executable receding-horizon chunk while the action decoder still predicts `action_horizon`; optional code-usage regularization and optional transition-state shape/loss paths have unit coverage, but it is not yet validated in a full training run.
- SO101 and LeRobot v3 batch builders emit execution/prediction boundary state targets with valid masks for the optional transition head.
- Stage-2 MoE/router code was added behind `use_lara_moe`; it now has torch tests for resident-pool routing, resident-pool mask reuse across chunks, chunk top-k routing, `router_horizon`-prefix per-expert action-loss posterior responsibility with optional floor/top-r smoothing, routed direct-expert action output, episode-level pool target aggregation, and optional expert-diversity/entropy stabilizers, but it is not yet validated in a full training run.
- The pool router can optionally sample a smaller resident-pool size during training via `lara_episode_pool_size_min`, condition the episode-level logits on the active resident budget, then use the configured maximum pool size at evaluation/deployment.
- SO101 batches can optionally include `episode_start_image` via `datasets.vla_data.include_episode_start`; when MoE is enabled, `Lara_core` encodes that first observation into `initial_context_tokens` so the episode-level resident pool router can condition on `h_1` rather than the current chunk context. The default remains `false` to avoid a second Qwen encode in the baseline.
- Trajectory ids are now passed from the LeRobot dataloader through `Lara_core` into the action adapter so batch-local episode responsibility can supervise the pool router.
- SO101/LeRobot training samples can optionally load counterfactual utility sidecar labels from `datasets.vla_data.counterfactual_utility_labels_path`; records are keyed by `context_id` or `trajectory_id:base_index` and injected as `utility_scores` plus `utility_candidate_mask` only when the sidecar has at least two candidate experts for that context. `counterfactual_utility_sample_labeled_only` can restrict utility-calibration training batches to labeled steps.
- Utility calibration loss code, optional action-loss utility labels, optional transition-state consistency utility labels, dataset-provided utility/candidate/cost/component target pass-through, a generic candidate utility scorer, strict counterfactual rollout-record to utility-matrix conversion, and an optional supervised route utility head were added behind zero default weights; they still need real counterfactual labels or closed-loop evaluator signals before they can be considered the paper's calibration stage.
- Direct action experts can now split reconstruction losses into full-chunk, `utility_horizon` prefix, and long-horizon-tail components and use them as optional value/progress/uncertainty labels for the utility head (`lara_use_action_loss_utility_components: false` by default). This is still an action-reconstruction proxy, not closed-loop utility calibration.
- When `lara_use_state_utility` is enabled with the transition head, each residual expert can be scored by its execution/prediction boundary-state consistency error and those losses can optionally supervise the route utility head components (`lara_use_state_utility_components: false` by default). This is closer to the paper's latent-state consistency proxy, but it is still not a true closed-loop counterfactual evaluator.
- Optional direct action-chunk expert heads and routed direct-expert action output were added behind `lara_use_direct_action_experts: false` and `lara_use_direct_action_output: false`; they still need real SO101 training validation.
- Route-quality aggregation metrics were added for offline diagnostics: Spearman/Kendall ranking fidelity, top-k consistency, route regret, route-switch-rate, and retained probability mass; closed-loop subset-retention success curves still need real evaluation rollouts.
- Action-head route-quality metrics are emitted as scalar `metric/moe_route_quality_*` values by the trainers when MoE is enabled.
- Sparse active/resident expert budget helpers plus matched-compute rows, matched-resident rows, budget-match flags, Pareto frontier flags, subset-retention success aggregation, route-sequence diagnostics for resident-pool reuse across receding-horizon chunks, automatic diagnostic extraction from raw `router_probs_sequence` / `active_mask_sequence` / `pool_mask_sequence` rollout fields, strict `--require-paper-metrics` evidence auditing, and `scripts/summarize_lara_protocol.py` JSON/JSONL rollout-record summarization were added for matched-budget reporting; real FLOPs, latency, VRAM, and success measurements still require benchmark runs.
- Action-head auxiliary diagnostics are now returned as `metric/...` outputs and excluded from the differentiable loss sum by the trainers.
- The websocket deployment server caches MoE `resident_pool_mask` and `router_probs` values per `session_id`, feeds cached routes back as `previous_router_probs`, and supports `reset`, allowing closed-loop clients to reuse an episode-level expert pool and optionally bias chunk-level routing across receding-horizon chunks when `lara_inference_stickiness_weight > 0`. Inference can also force `forced_expert_id` / `forced_router_probs` for counterfactual candidate evaluation. When started with `--rollout_trace_path`, the server can write session-level JSONL records with raw `router_probs_sequence`, `active_mask_sequence`, `pool_mask_sequence`, `forced_expert_id_sequence`, measured `latency_ms_sequence`, optional CUDA `vram_mb_sequence`, and reset/outcome fields for later protocol summarization.
- `scripts/build_counterfactual_utility_labels.py` converts forced-expert rollout traces into the utility sidecar consumed by `counterfactual_utility_labels_path`, and validates the minimum candidate count per context before writing labels.
- `scripts/smoke_lara_real_components.py` now provides an explicit preflight for local Qwen/V-JEPA checkpoint paths and optional real `Lara` instantiate/one-step dummy forward-backward checks, with structured error reporting for missing runtime dependencies or model-load failures.
- The real-component smoke script can temporarily enable the default-off Stage-1 latent-action head and Stage-2 MoE/direct action expert scaffolds. Local `--run-step --attn-implementation sdpa` checks now complete dummy forward/backward for the latent-only path, the MoE/direct-output path, and the combined latent+MoE/direct-output path.
- The same smoke script can now use `--use-real-batch` to load examples from the configured SO101 LeRobot dataset, report real sample shapes, and run the model path on real SO101 batch contents instead of synthetic zeros. Passing this smoke only proves integration wiring; it is not evidence that the paper's latent-action MoE or two-level routing algorithm is trained or complete.
- The smoke script can also temporarily enable `--use-transition-head`; dummy smoke now supplies execution/prediction boundary state targets, and real SO101 batch smoke uses the dataloader-provided boundary targets to exercise the transition-state loss path.
- The smoke script can optionally run `--optimizer-step`, which performs one lightweight SGD update over `action_head` parameters after backward and reports gradient/update diagnostics. This validates a minimal parameter-update path, not a full trainer run.

## Remaining Engineering Risks

- Action target alignment is explicit for the current SO101 dataloader via strict `future_actions`. Future datasets that return past/current/future actions together must split out an `action_horizon`-length `future_actions` window before calling the action adapter.
- Full `Lara` model instantiation with real Qwen/V-JEPA checkpoints and real-component one-step training smoke tests now have a script entry point. The current local `--run-step --attn-implementation sdpa` checks complete dummy forward/backward through the VLA baseline and optional latent/MoE scaffolds, real SO101 batch forward/backward through the VLA baseline plus the combined latent+MoE/direct-output scaffold, and an optional action-head-only optimizer update; the default FlashAttention path still reports missing `flash_attn`.
- The latent action head is currently a Stage-1 skeleton that can pass real-component smoke, but it still needs empirical SO101 training validation, loss-weight tuning, and ablation against the token-conditioned flow baseline.
- The MoE/router path is currently a Stage-2 scaffold that can pass real-component smoke with optional utility labels, but it still needs full-train validation of direct experts and per-expert action-loss posterior paths, resident-pool success evaluation, and closed-loop validation beyond server-side pool reuse. Do not describe it as the completed LARA MoE or completed two-level router yet.
- Utility calibration currently validates action-loss utility labels, transition-state consistency utility labels, direct-expert action reconstruction component labels, generic utility composition, a trainable utility-head interface, and loss surfaces; it still does not supervise utility from true closed-loop evaluator labels.
- The optional transition head and dataloader boundary targets are wired and can be exercised by real-batch smoke, but transition-state training still needs empirical validation and loss-weight tuning.
- Matched-compute and matched-resident support currently covers reporting protocol, route-sequence diagnostics, rollout-record summarization, server-side latency/VRAM trace capture, and expert-budget accounting only; it does not measure real FLOPs or closed-loop success.
- The new pytest static tests were not run in the system Python because that interpreter lacks `pytest`; run them inside the project environment with `python -m pytest tests/test_baseline_static.py`.
- VJ2 video preprocessing still happens inside the forward path and may bottleneck training.
- `pyproject.toml` and `requirements.txt` do not pin the torch/CUDA runtime; the project still depends on a compatible prebuilt PyTorch environment.

## Suggested Implementation Order

1. Use the existing real-component smoke flags before heavy jobs to catch Qwen/V-JEPA/action-head integration failures in the baseline, latent-only, MoE-only, combined latent+MoE, dummy-batch, and real-SO101-batch paths.
2. Use the optional `past_actions` interface when a dataset needs history; `future_actions` and `current_state` are explicit for SO101.
3. Validate and tune the optional latent-action posterior/codebook/prior path in real SO101 fine-tuning.
4. Validate the optional MoE/router path with real trajectory-id batches, direct action experts, per-expert posterior labels, and route-quality diagnostics.
5. Add complete utility calibration supervision from real counterfactual or closed-loop evaluator signals.
6. Add matched-compute and matched-resident-expert experiments.
