# LARA Implementation Gap

This repository currently implements the SO101 VLA-JEPA action baseline, not the full LARA paper method.

## Implemented Baseline

- SO101 LeRobot dataset wiring.
- VLA-JEPA Pretrain checkpoint loading.
- Qwen/V-JEPA latent action tokens and embodied action tokens.
- Latent-token conditioned flow-matching action head.
- Prediction horizon `H_p = 60` and execution horizon `H_e = 10` configuration.
- Weighted action loss for executable prefix versus long-horizon tail.

## Experimental Scaffolding

These files exist to make the next implementation steps concrete, but they are not complete LARA components and are disabled by default:

- Stage-1 latent action head scaffold: posterior encoder, VQ codebook, optional code-usage regularization, context-only prior, and optional execution/prediction boundary-state transition loss (`use_latent_action_head: false`, `lara_use_transition_head: false`).
- Stage-2 MoE/router scaffold: residual token experts, optional direct action-chunk experts, optional routed direct-expert action output, posterior responsibility from latent tokens or per-expert action reconstruction losses, optional posterior floor/top-r smoothing, episode-level resident pool targets from aggregated chunk responsibility, reusable episode-level resident pool masks, optional training-time randomized resident-pool size, chunk-level top-k routing constrained to the resident pool, optional inference stickiness, posterior-to-router distillation losses, balance/stickiness/expert-diversity/entropy stabilizers, and route-quality aggregation metrics (`use_lara_moe: false`).
- Utility calibration scaffold: optional action-loss utility labels, optional direct-expert action reconstruction component labels for value/progress/uncertainty targets, optional dataset-provided utility/candidate/cost/component targets, optional supervised route utility head, candidate value/progress/uncertainty/cost scoring helpers, centered utility regression, and pairwise ranking losses (`lara_utility_loss_weight: 0.0`, `lara_utility_head_loss_weight: 0.0`, `lara_use_action_loss_utility: false`, `lara_use_action_loss_utility_components: false`, `lara_use_utility_head: false`).

## Missing Paper Components

- Production-ready latent action training and validation.
- Validated transition-state training with real SO101 boundary targets.
- Validated MoE action experts that directly produce or adapt action chunks in full SO101 training.
- Closed-loop route diagnostics and subset-retention success curves.
- Real counterfactual utility scoring from latent-state or closed-loop evaluator signals beyond action reconstruction labels.
- Validated matched-compute and matched-resident-expert experiments with real FLOPs, latency, VRAM, and rollout success.

## Baseline Reliability Fixes Applied

- VLA, video-only, and VLA/video co-training loops no longer zero gradients at the start of every `accelerator.accumulate` block; gradient clipping is gated on `accelerator.sync_gradients`, and checkpoint waits now use the trainer's accelerator instance.
- Distributed barriers and rank checks are guarded for single-process execution.
- Qwen latent/embodied special token counts are checked per sample before hidden-state reshaping and covered by a lightweight unit test.
- Qwen3/Qwen2.5 wrappers now respect `framework.qwenvl.attn_implementation`, and the real-component smoke script can temporarily override it for non-FlashAttention environments.
- V-JEPA world-model view count is explicit through `num_world_model_views`; SO101 single-camera batches and smoke dummy batches duplicate to two view streams before the predictor.
- Real-component `--run-step` smoke checks now mirror trainer/server device placement for V-JEPA and the action head instead of leaving them on CPU beside a CUDA Qwen model.
- Flow-matching timestep buckets are clamped to the valid range.
- `ActionHeadAdapter` has a dummy-batch forward/predict smoke test for basic loss and output shape.
- SO101 batches now expose `future_actions` and `current_state` explicitly, with `action` and `state` retained as compatibility aliases.
- `future_actions` is now treated as a strict future-only target and must match `action_horizon`; only the legacy `action` fallback is tail-sliced.
- The LeRobot v3 collator can be imported without the optional `lerobot` package and has unit coverage for explicit `future_actions/current_state` aliases.
- Action labels remain fp32 in the adapter.
- Static pytest coverage was added for the baseline guardrails in `tests/test_baseline_static.py`.
- Stage-1 latent action head code was added behind `use_latent_action_head`; posterior/VQ/prior, optional code-usage regularization, and optional transition-state shape/loss paths have unit coverage, but it is not yet validated in a full training run.
- SO101 and LeRobot v3 batch builders emit execution/prediction boundary state targets with valid masks for the optional transition head.
- Stage-2 MoE/router code was added behind `use_lara_moe`; it now has torch tests for resident-pool routing, resident-pool mask reuse across chunks, chunk top-k routing, per-expert action-loss posterior responsibility with optional floor/top-r smoothing, routed direct-expert action output, episode-level pool target aggregation, and optional expert-diversity/entropy stabilizers, but it is not yet validated in a full training run.
- The pool router can optionally sample a smaller resident-pool size during training via `lara_episode_pool_size_min`, then use the configured maximum pool size at evaluation/deployment.
- Trajectory ids are now passed from the LeRobot dataloader through `Lara_core` into the action adapter so batch-local episode responsibility can supervise the pool router.
- Utility calibration loss code, optional action-loss utility labels, dataset-provided utility/candidate/cost/component target pass-through, a generic candidate utility scorer, and an optional supervised route utility head were added behind zero default weights; they still need real counterfactual labels or closed-loop evaluator signals before they can be considered the paper's calibration stage.
- Direct action experts can now split reconstruction losses into full-chunk, executable-prefix, and long-horizon-tail components and use them as optional value/progress/uncertainty labels for the utility head (`lara_use_action_loss_utility_components: false` by default). This is still an action-reconstruction proxy, not closed-loop utility calibration.
- Optional direct action-chunk expert heads and routed direct-expert action output were added behind `lara_use_direct_action_experts: false` and `lara_use_direct_action_output: false`; they still need real SO101 training validation.
- Route-quality aggregation metrics were added for offline diagnostics: Spearman/Kendall ranking fidelity, top-k consistency, route regret, route-switch-rate, and retained probability mass; closed-loop subset-retention success curves still need real evaluation rollouts.
- Action-head route-quality metrics are emitted as scalar `metric/moe_route_quality_*` values by the trainers when MoE is enabled.
- Sparse active/resident expert budget helpers plus matched-compute rows, matched-resident rows, budget-match flags, Pareto frontier flags, and subset-retention success aggregation were added for matched-budget reporting; real FLOPs, latency, VRAM, and success measurements still require benchmark runs.
- Action-head auxiliary diagnostics are now returned as `metric/...` outputs and excluded from the differentiable loss sum by the trainers.
- The websocket deployment server caches MoE `resident_pool_mask` and `router_probs` values per `session_id`, feeds cached routes back as `previous_router_probs`, and supports `reset`, allowing closed-loop clients to reuse an episode-level expert pool and optionally bias chunk-level routing across receding-horizon chunks when `lara_inference_stickiness_weight > 0`.
- `scripts/smoke_lara_real_components.py` now provides an explicit preflight for local Qwen/V-JEPA checkpoint paths and optional real `Lara` instantiate/one-step dummy forward-backward checks, with structured error reporting for missing runtime dependencies or model-load failures.
- The real-component smoke script can temporarily enable the default-off Stage-1 latent-action head and Stage-2 MoE/direct action expert scaffolds. Local `--run-step --attn-implementation sdpa` checks now complete dummy forward/backward for the latent-only path, the MoE/direct-output path, and the combined latent+MoE/direct-output path.
- The same smoke script can now use `--use-real-batch` to load examples from the configured SO101 LeRobot dataset, report real sample shapes, and run the model path on real SO101 batch contents instead of synthetic zeros. Passing this smoke only proves integration wiring; it is not evidence that the paper's latent-action MoE or two-level routing algorithm is trained or complete.
- The smoke script can also temporarily enable `--use-transition-head`; dummy smoke now supplies execution/prediction boundary state targets, and real SO101 batch smoke uses the dataloader-provided boundary targets to exercise the transition-state loss path.
- The smoke script can optionally run `--optimizer-step`, which performs one lightweight SGD update over `action_head` parameters after backward and reports gradient/update diagnostics. This validates a minimal parameter-update path, not a full trainer run.

## Remaining Engineering Risks

- Action target alignment is explicit for the current SO101 dataloader via strict `future_actions`. Future datasets that return past/current/future actions together must split out an `action_horizon`-length `future_actions` window before calling the action adapter.
- Full `Lara` model instantiation with real Qwen/V-JEPA checkpoints and real-component one-step training smoke tests now have a script entry point. The current local `--run-step --attn-implementation sdpa` checks complete dummy forward/backward through the VLA baseline and optional latent/MoE scaffolds, real SO101 batch forward/backward through the VLA baseline plus the combined latent+MoE/direct-output scaffold, and an optional action-head-only optimizer update; the default FlashAttention path still reports missing `flash_attn`.
- The latent action head is currently a Stage-1 skeleton that can pass real-component smoke, but it still needs empirical SO101 training validation, loss-weight tuning, and ablation against the token-conditioned flow baseline.
- The MoE/router path is currently a Stage-2 scaffold that can pass real-component smoke with direct action experts and routed direct output, but it still needs full-train validation of direct experts and per-expert action-loss posterior paths, resident-pool success evaluation, and closed-loop validation beyond server-side pool reuse.
- Utility calibration currently validates action-loss utility labels, direct-expert action reconstruction component labels, generic utility composition, a trainable utility-head interface, and loss surfaces; it does not yet supervise utility from latent-state or closed-loop evaluator labels.
- The optional transition head and dataloader boundary targets are wired and can be exercised by real-batch smoke, but transition-state training still needs empirical validation and loss-weight tuning.
- Matched-compute and matched-resident support currently covers reporting protocol and expert-budget accounting only; it does not measure real FLOPs, latency, VRAM, or closed-loop success.
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
