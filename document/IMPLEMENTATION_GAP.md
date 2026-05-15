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

- Stage-1 latent action head scaffold: posterior encoder, VQ codebook, and context-only prior (`use_latent_action_head: false`).
- Stage-2 MoE/router scaffold: residual token experts, optional direct action-chunk experts, posterior responsibility from latent tokens or per-expert action reconstruction losses, episode-level resident pool targets from aggregated chunk responsibility, chunk-level top-k routing constrained to the resident pool, posterior-to-router distillation losses, and route collapse diagnostics (`use_lara_moe: false`).
- Utility calibration scaffold: centered utility regression and pairwise ranking losses for externally supplied candidate route utilities (`lara_utility_loss_weight: 0.0`).

## Missing Paper Components

- Production-ready latent action training and validation.
- Validated MoE action experts that directly produce or adapt action chunks in full training.
- Closed-loop route diagnostics and subset-retention curves.
- Real counterfactual utility scoring from value/progress/latent-state or closed-loop evaluator signals.
- Matched-compute and matched-resident-expert evaluation protocol.

## Baseline Reliability Fixes Applied

- Training loop no longer zeroes gradients at the start of every `accelerator.accumulate` block.
- Distributed barriers and rank checks are guarded for single-process execution.
- Qwen latent/embodied special token counts are checked per sample before hidden-state reshaping.
- Flow-matching timestep buckets are clamped to the valid range.
- `ActionHeadAdapter` has a dummy-batch forward/predict smoke test for basic loss and output shape.
- SO101 batches now expose `future_actions` explicitly, with `action` retained as a compatibility alias.
- Action labels remain fp32 in the adapter.
- Static pytest coverage was added for the baseline guardrails in `tests/test_baseline_static.py`.
- Stage-1 latent action head code was added behind `use_latent_action_head`; it is not yet validated in a full training run.
- Stage-2 MoE/router code was added behind `use_lara_moe`; it now has torch tests for resident-pool routing, chunk top-k routing, per-expert action-loss posterior responsibility, and episode-level pool target aggregation, but it is not yet validated in a full training run.
- Trajectory ids are now passed from the LeRobot dataloader through `Lara_core` into the action adapter so batch-local episode responsibility can supervise the pool router.
- Utility calibration loss code was added behind zero default weights; it still needs a real utility-score producer before it can be considered the paper's counterfactual calibration stage.
- Optional direct action-chunk expert heads were added behind `lara_use_direct_action_experts: false`; they still need real SO101 training validation.

## Remaining Engineering Risks

- Action target alignment is explicit for the current SO101 dataloader via `future_actions`. Future datasets that return past/current/future actions together must split out `future_actions` before calling the action adapter.
- Full `Lara` model instantiation and one-step training smoke tests still need to be run with local Qwen/V-JEPA checkpoints.
- The latent action head is currently a Stage-1 skeleton and still needs empirical validation, loss-weight tuning, and ablation against the token-conditioned flow baseline.
- The MoE/router path is currently a Stage-2 scaffold and still needs full-train validation of the direct expert and per-expert action-loss posterior paths, resident-pool evaluation, and closed-loop validation.
- Utility calibration currently validates only the loss surface; it does not yet compute candidate route utility from value/progress/latent-state or closed-loop evaluator signals.
- The new pytest static tests were not run in the system Python because that interpreter lacks `pytest`; run them inside the project environment with `python -m pytest tests/test_baseline_static.py`.
- VJ2 video preprocessing still happens inside the forward path and may bottleneck training.
- `pyproject.toml` and `requirements.txt` do not pin the torch/CUDA runtime; the project still depends on a compatible prebuilt PyTorch environment.

## Suggested Implementation Order

1. Add full-framework smoke tests for tokenizer integration and one fake-batch train step.
2. Make action batches explicit: `future_actions`, optional `past_actions`, and `current_state`.
3. Validate and tune the optional latent-action posterior/codebook/prior path.
4. Validate the optional MoE/router path with real trajectory-id batches and route-quality diagnostics.
5. Add complete router distillation/utility calibration losses.
6. Add matched-compute and matched-resident-expert experiments.
