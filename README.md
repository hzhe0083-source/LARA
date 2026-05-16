# LARA

LARA is the local research codebase for adapting VLA-JEPA pretraining to SO101 robot control.

The current implementation is an engineering baseline, not the full latent-action MoE system from the LARA paper. It uses the VLA-JEPA pretrain representation, extracts latent action tokens from the Qwen/V-JEPA token stream, and conditions a GR00T-style flow-matching action head on:

```text
latent action tokens + embodied action tokens -> continuous follower-arm action chunk
```

The SO101 path intentionally does not default to the VLA-JEPA Real-world checkpoint, because that checkpoint is adapted to other robot embodiments. For SO101, the training target is the follower arm `action` and `state` from the local LeRobot dataset.

## Implementation Status

The manuscript in [`document/LARA_collapse_paper.tex`](./document/LARA_collapse_paper.tex) describes the intended full LARA algorithm. The code in this repository has implemented the SO101 VLA/action-baseline path and several default-off scaffolds, but the paper's latent-action MoE and two-level routing method should still be treated as unfinished.

See [`document/IMPLEMENTATION_GAP.md`](./document/IMPLEMENTATION_GAP.md) for the current paper-to-code gap list and recommended implementation order.

Completed in code:

- SO101 single-arm LeRobot dataset support.
- VLA-JEPA Pretrain checkpoint loading.
- Latent-token conditioned flow action head baseline.
- Long-prediction / short-execution horizon setup:
  - `action_horizon: 60`
  - `execution_horizon: 10`
  - at 30 Hz, predict 2.0 seconds and execute the first 0.333 seconds before re-observing.
- SO101/LeRobot action-valid masks for trajectory-end chunks, so padded future action steps do not supervise the flow, latent-action, transition, or direct-expert losses.

Experimental scaffolding exists but is not complete or validated:

- Stage-1 latent action head scaffold with posterior encoder, VQ codebook, optional code-usage regularization, context-only prior, and optional execution/prediction boundary-state transition loss (`use_latent_action_head: false`, `lara_use_transition_head: false` by default). This is not yet production-ready latent-action training.
- Stage-2 MoE/router scaffold with residual token experts, optional direct action-chunk experts, optional routed direct-expert action output, posterior responsibility from latent tokens or per-expert action reconstruction losses, optional posterior floor/top-r smoothing, LeRobot trajectory ids for episode-level resident pool targets, optional episode-start image encoding for the resident pool router, reusable episode-level resident pool masks, budget-conditioned episode pool routing, optional training-time randomized resident-pool size, chunk-level top-k routing inside the resident pool, optional inference stickiness, optional balance/stickiness/expert-diversity/entropy stabilizers, and route-quality aggregation metrics (`use_lara_moe: false` by default). This is code scaffolding only; the MoE experts and two-level router have not been validated as the LARA paper method.
- Utility calibration scaffold with optional action-loss utility labels, optional transition-state consistency utility labels, optional direct-expert action reconstruction or transition-state component labels for value/progress/uncertainty targets, optional dataset-provided utility/candidate/cost/component targets, strict counterfactual rollout-record to `utility_scores` / `utility_candidate_mask` matrix conversion, an optional supervised route utility head, candidate value/progress/uncertainty/cost scoring helpers, centered utility regression, and pairwise ranking losses (`lara_utility_loss_weight: 0.0`, `lara_utility_head_loss_weight: 0.0`, `lara_use_action_loss_utility: false`, `lara_use_state_utility: false`, `lara_use_utility_head: false` by default). These labels are proxies, pass-through hooks, or offline label plumbing; real closed-loop counterfactual evaluator labels are still required before this becomes the paper's utility calibration stage.
- Action-head MoE diagnostics include `metric/moe_route_quality_*` scalars for posterior/utility ranking, top-k consistency, route regret, and retained probability mass when MoE is enabled.
- Matched-compute and matched-resident protocol helpers for subset-retention success aggregation, active/resident budget checks, result-table rows, compute-success Pareto flags, route-sequence diagnostics over receding-horizon chunks, automatic diagnostic extraction from raw `router_probs_sequence` / `active_mask_sequence` / `pool_mask_sequence` rollout fields, and JSON/JSONL rollout-record summarization via `scripts/summarize_lara_protocol.py`.
- Strict protocol evidence audit via `scripts/summarize_lara_protocol.py --require-paper-metrics`, which fails when rollout records lack paper-required success, FLOPs, latency, VRAM, route diagnostics, or required resident fractions.
- Paper-readiness audit via `scripts/audit_lara_paper_readiness.py`, which keeps the repository from treating default-off MoE/router scaffolding as the completed paper method until real utility sidecars, closed-loop protocol records, full SO101 training evidence, and real robot evaluation artifacts are present.
- Minimal dummy-batch smoke coverage exists for `ActionHeadAdapter` forward and prediction shapes.
- The real-component smoke script can temporarily enable the default-off Stage-1/Stage-2 scaffolds to prove they instantiate and complete a dummy forward/backward with local Qwen/V-JEPA checkpoints. This is an integration smoke check only; it is not full SO101 training or closed-loop validation.

Described in the paper but not implemented yet:

- production-ready latent action training
- validated MoE action experts that directly model or adapt action chunks in full SO101 training
- validated two-level routing behavior: episode-level resident pool selection plus chunk-level top-k routing inside that resident pool, under real SO101 training and rollout evaluation
- real counterfactual utility scoring from latent-state or closed-loop evaluator signals beyond action reconstruction labels; the code can now reject single-route records and build utility matrices from true multi-candidate logs, but the evaluator labels still need to be produced
- validated transition-state training with real SO101 boundary targets
- full resident-pool training/evaluation and closed-loop success/retention curves beyond static and unit tests
- matched-compute and matched-resident-expert experiments from real rollout records with measured FLOPs, latency, VRAM, and closed-loop success

In other words, the current code path is:

```text
VLA-JEPA/Qwen token stream
  -> latent action tokens + embodied action tokens
  -> flow-matching action head
  -> continuous SO101 follower-arm action chunk
```

It is a baseline adapter for SO101 fine-tuning, not the final latent-action MoE/router implementation.

## Repository Layout

```text
Lara/
  dataloader/                 LeRobot and video dataloaders
  model/framework/            LARA framework assembly
  model/modules/action_model/ Flow-matching and ACT-style action heads
  model/modules/world_model/  V-JEPA latent world model pieces
  training/                   Accelerate/DeepSpeed training loops
scripts/
  config/lara_so101_ft.yaml   compatibility alias for SO101 VLA baseline
  config/lara_so101_baseline.yaml
                                SO101 VLA/action baseline config
  config/lara_so101_latent_vq.yaml
                                SO101 latent-action VQ scaffold config
  config/lara_so101_moe_direct.yaml
                                SO101 direct MoE action scaffold config
  config/lara_so101_utility_pool.yaml
                                SO101 utility/pool scaffold config
  config/lara_libero100_baseline.yaml
                                LIBERO100 VLA baseline config
  config/lara_metaworld_mt50_baseline.yaml
                                MetaWorld MT50 VLA baseline config
  download_benchmark_data.py   Download/preflight benchmark LeRobot datasets
  summarize_lara_protocol.py  Summarize rollout JSON/JSONL into paper protocol rows
  audit_lara_paper_readiness.py
                                Fail paper-readiness claims when required evidence is missing
models/                       local checkpoints, ignored by git
```

## Environment

```bash
conda create -n Lara python=3.10 -y
conda activate Lara

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

The code expects CUDA-capable PyTorch, Accelerate, DeepSpeed, Qwen/V-JEPA dependencies, and FlashAttention2.

## Checkpoints

Place local model checkpoints under `models/`. This directory is intentionally ignored by git.

Expected default layout:

```text
models/
  Qwen3-VL-2B-Instruct/
  vjepa2-vitl-fpc64-256/
  VLA-JEPA/
    Pretrain/checkpoints/VLA-JEPA-pretrain.pt
```

The SO101 config uses:

```yaml
trainer:
  pretrained_checkpoint: ./models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt
  reload_modules: qwen,vj2
```

This means the VLA-JEPA pretrain representation is reused, while the SO101 action head is trained on SO101 follower-arm action/state data.

The SO101 configs are deliberately split by paper stage:

| Config | Intended use | Status |
| --- | --- | --- |
| `scripts/config/lara_so101_baseline.yaml` | VLA-JEPA pretrain + latent/body tokens + flow action head | Main baseline |
| `scripts/config/lara_so101_latent_vq.yaml` | Adds posterior/VQ/prior latent-action head and action-chunk reconstruction loss | Scaffold, needs SO101 validation |
| `scripts/config/lara_so101_moe_direct.yaml` | Adds latent head plus MoE/direct action experts and routed direct action output | Scaffold, needs SO101 validation |
| `scripts/config/lara_so101_utility_pool.yaml` | Adds utility/pool settings for counterfactual sidecar training | Scaffold, requires real forced-route utility labels |
| `scripts/config/lara_so101_ft.yaml` | Backward-compatible alias of the baseline config | Compatibility |

For trainer evaluation, add `datasets.vla_eval_data` with a separate validation dataset config. If it is absent, the trainer records `eval/skipped_no_eval_dataloader` instead of reusing the training iterator as validation.

Run the lightweight real-component smoke preflight before a heavy training job:

```bash
python scripts/smoke_lara_real_components.py --config scripts/config/lara_so101_ft.yaml
```

When the local Qwen/V-JEPA checkpoints and runtime dependencies are available, add `--instantiate` or `--run-step` to load the actual `Lara` framework and execute a one-step dummy forward/backward check. Use `--attn-implementation sdpa` or `--attn-implementation eager` to smoke-test environments that do not have FlashAttention2 installed. The `--run-step` path mirrors trainer/server device placement for V-JEPA and the action head. The smoke dummy follows the SO101 dataloader convention of two V-JEPA view streams; single-camera SO101 data is duplicated before entering the world model.
Those modes return structured JSON errors if dependency import or model loading fails, which makes missing runtime packages easier to diagnose before launching training.

To smoke-check paper-stage scaffolds without changing the default training config:

```bash
python scripts/smoke_lara_real_components.py \
  --config scripts/config/lara_so101_ft.yaml \
  --use-real-batch

python scripts/smoke_lara_real_components.py \
  --config scripts/config/lara_so101_ft.yaml \
  --run-step \
  --attn-implementation sdpa \
  --optimizer-step \
  --use-latent-action-head \
  --use-transition-head

python scripts/smoke_lara_real_components.py \
  --config scripts/config/lara_so101_ft.yaml \
  --run-step \
  --attn-implementation sdpa \
  --use-latent-action-head \
  --use-lara-moe \
  --use-direct-action-experts \
  --use-direct-action-output \
  --use-action-loss-utility-components
```

`--use-real-batch` loads a small batch from the configured SO101 LeRobot dataset instead of the synthetic dummy batch. It can be combined with `--run-step` and the paper-stage flags to check that real SO101 sample shapes feed the same model path. `--optimizer-step` implies the one-step smoke and runs one lightweight SGD update over `action_head` parameters after backward, reporting gradient/update diagnostics without pretending to be a full trainer run. `--include-episode-start` temporarily requests `episode_start_image` from the SO101 dataloader and makes the optional episode-level pool router condition on the first observation `h_1` instead of the current chunk context. `--use-transition-head` temporarily enables the boundary-state transition scaffold and gives it loss weight 1.0 when the config keeps the default zero weight; pass `--transition-loss-weight` to override that smoke weight. `--use-direct-action-experts` implies `--use-lara-moe`; `--use-direct-action-output` implies both. `--use-action-loss-utility-components` derives value/progress/uncertainty utility-head targets from direct-expert full/execution/tail action reconstruction losses and also enables the required direct experts, MoE, and utility head. `--use-state-utility` derives router utility labels from per-expert transition-state consistency errors and enables MoE plus the transition head; `--use-state-utility-components` also supervises the utility-head value/progress/uncertainty components from those transition-state errors. `--counterfactual-utility-labels-path` points smoke checks at a generated sidecar, enables MoE utility loss, and restricts real-batch sampling to labeled steps unless `--counterfactual-utility-all-steps` is set. These flags are smoke-time overrides only, and a passing smoke check does not mean the MoE/two-level routing method from the paper is complete or trained.

## SO101 Dataset

The default SO101 config points to:

```text
/home/ryan/Documents/robot/VLA_JEPA/runs/lerobot_so101
```

Expected dataset key:

```yaml
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_mix: so101_single_arm
    with_state: true
```

The configured SO101 action/state dimensions are:

```yaml
framework:
  action_model:
    action_dim: 7
    state_dim: 8
```

Action order:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

State order:

```text
[x, y, z, roll, pitch, yaw, pad, gripper]
```

## Horizon Settings

The SO101 baseline uses receding-horizon control:

```yaml
framework:
  action_model:
    future_action_window_size: 59
    action_horizon: 60
    execution_horizon: 10
    latent_action_horizon: 10
    router_horizon: 10
    utility_horizon: 10
    long_prediction_aux_horizon: 60
    execution_loss_weight: 1.0
    prediction_loss_weight: 0.5
```

`action_horizon` is the canonical prediction horizon. `future_action_window_size` is a legacy compatibility field and should stay equal to `action_horizon - 1` until it is removed.
SO101 batches pass `future_actions` as an explicit future-only supervision window, and that window must have exactly `action_horizon` steps. The older `action` fallback can still carry wider context and is tail-sliced only for compatibility.
SO101 batches also pass `future_action_mask`; near trajectory ends, invalid padded steps are masked out of the flow-matching loss, latent posterior pooling, transition/action utility proxies, and direct-expert reconstruction losses.
When the optional latent-action head is enabled, its posterior/codebook/prior is trained on the first `latent_action_horizon` steps of that window, so the latent code stays aligned with the executable receding-horizon chunk while the flow/direct action heads still predict the full `action_horizon`.
When the optional MoE/direct-expert path is enabled, posterior responsibility and pool-router targets use `router_horizon`, while action-loss utility proxies and value/progress/uncertainty component labels use `utility_horizon`.
For episode-level pool-router experiments, set `datasets.vla_data.include_episode_start: true` or use the smoke flag `--include-episode-start`; this adds the episode's first image and lets `Lara_core` encode an initial pool context for `p_\chi(P_\tau | g, h_1, b)`. The default remains `false` so the baseline path does not pay a second Qwen encode.
For utility-calibration experiments with real counterfactual evaluator labels, set `datasets.vla_data.counterfactual_utility_labels_path` to a JSON/JSONL file whose records contain either `context_id` or `trajectory_id` plus `base_index`, a candidate `expert_id`, and an outcome such as `success`, `return_score`, or `utility_score`. The loader injects matched labels as `utility_scores` and `utility_candidate_mask`; records with fewer than two candidate experts per context are rejected by default. Set `counterfactual_utility_sample_labeled_only: true` when training the utility router on a partial sidecar so every batch sample carries labels.

At 30 Hz:

```text
action_horizon = 60    -> predict 2.0 seconds
execution_horizon = 10 -> execute 0.333 seconds
```

During inference, `predict_action` returns:

- `normalized_actions`: full 60-frame prediction.
- `execution_normalized_actions`: first 10 frames for closed-loop execution.
- `resident_pool_mask`: when MoE routing is enabled and no mask is supplied, the episode-level resident expert pool selected for reuse on later chunks.
- `router_probs` and `active_expert_mask`: when MoE routing is enabled, the chunk-level route distribution and selected active experts.

For MoE experiments, pass the returned `resident_pool_mask` back into later `predict_action` calls for the same episode so the chunk router chooses sparse experts inside a stable episode pool.
For counterfactual expert evaluation, pass `forced_expert_id` or `forced_router_probs` to `predict_action`; this forces the MoE route for that request and returns the forced route fields so the evaluator can record candidate outcomes for the utility sidecar.
The websocket deployment server also caches `resident_pool_mask` and `router_probs` by `session_id`, feeds cached router probabilities back as `previous_router_probs`, and clears the cache on a `reset` request. This lets closed-loop clients reuse an episode pool and, when `lara_inference_stickiness_weight > 0`, bias chunk-level routing toward the previous route without manually echoing those tensors on every inference call.
For evaluation runs, start the server with `--rollout_trace_path /path/to/rollouts.jsonl`; each `infer` appends raw route outputs plus measured `latency_ms_sequence`, optional CUDA `vram_mb_sequence`, and any forced expert sequence to the session trace, and `reset` or `record_outcome` writes a JSONL record with `router_probs_sequence`, `active_mask_sequence`, `pool_mask_sequence`, aggregate latency/VRAM, and any provided outcome fields such as `success`, `return_score`, and `flops`.
Use `scripts/build_counterfactual_utility_labels.py` to convert forced-expert rollout traces into the JSONL sidecar consumed by `counterfactual_utility_labels_path`; it validates that every context has the configured minimum number of candidate experts before writing labels.

## Benchmark Datasets

Benchmark data is kept outside the git repository under:

```text
/home/ryan/Documents/robot/benchmark_data
```

The first benchmark targets are:

| Benchmark | Hugging Face dataset | Local directory | Config |
| --- | --- | --- | --- |
| LIBERO100 | `kevin-ys-zhang/libero100_lerobot` | `/home/ryan/Documents/robot/benchmark_data/raw/libero100/kevin_libero100_lerobot` | `scripts/config/lara_libero100_baseline.yaml` |
| MetaWorld MT50 | `lerobot/metaworld_mt50` | `/home/ryan/Documents/robot/benchmark_data/raw/metaworld/lerobot_metaworld_mt50` | `scripts/config/lara_metaworld_mt50_baseline.yaml` |

Download or resume with low concurrency:

```bash
python scripts/download_benchmark_data.py --dataset libero100 --download --max-workers 1
python scripts/download_benchmark_data.py --dataset metaworld --download --max-workers 1
```

The script fixes `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE` under `benchmark_data`, disables Xet by default, writes logs to `benchmark_data/logs`, and defaults to dry-run unless `--download` is passed. By default it downloads only the LeRobot v3 local-training layout: `meta/**` and `data/chunk-*/*.parquet`, not duplicate Hugging Face viewer split files such as `data/train-*`. Check local readiness without downloading:

```bash
python scripts/download_benchmark_data.py --dataset all --preflight-only
```

A dataset is not considered ready until `meta/info.json`, `meta/stats.json`, task metadata (`meta/tasks.parquet` or `meta/tasks.jsonl`), and the expected LeRobot v3 chunk parquet count are present: 279 `data/chunk-*/*.parquet` files for LIBERO100 and 492 for MetaWorld MT50. The preflight intentionally counts chunk parquet files, not duplicate split files under `data/train-*`, and exits nonzero while any selected dataset is incomplete. For progress-only supervision during a long download, add `--allow-incomplete`. The current benchmark configs use the baseline VLA path only; MoE and two-level routing remain disabled by default.

To check whether the repository has enough evidence to claim the full LARA paper method is complete, run:

```bash
python scripts/audit_lara_paper_readiness.py \
  --config scripts/config/lara_so101_ft.yaml \
  --counterfactual-utility-labels /path/to/utility.jsonl \
  --rollout-records /path/to/closed_loop_rollouts.jsonl \
  --full-so101-training-artifact /path/to/training_summary.json \
  --closed-loop-robot-eval-artifact /path/to/robot_eval_summary.json
```

The command intentionally exits nonzero when required evidence is missing. For preflight logs where an incomplete report is still useful, add `--allow-incomplete`.

The training artifact is a JSON object, not a free-form log. It must show a completed real-SO101 run, positive training steps, an existing checkpoint path, enabled latent/MoE/transition/direct-expert/expert-posterior flags, counterfactual utility labels with positive utility loss weight, and finite final metrics including `action_loss`, `transition_state_loss`, `moe_loss`, `moe_route_distill_loss_raw`, `moe_route_distill_loss_weighted`, `moe_pool_distill_loss_weighted`, and `moe_utility_loss_weighted`.

The robot evaluation artifact is also structured JSON. It must identify SO101 real-robot closed-loop evaluation, match the configured 60/10 horizons, cover the required resident fractions, report finite `success_rate`, include enough episodes for `--min-robot-eval-episodes`, and explicitly confirm route diagnostics, matched-compute metrics, and counterfactual utility evaluation.

## Training

Run SO101 fine-tuning with:

```bash
accelerate launch \
  --config_file ./Lara/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./Lara/training/train_lara.py \
  --config_yaml ./scripts/config/lara_so101_baseline.yaml
```

For fewer GPUs, change `--num_processes`.

You can also override config values from the CLI:

```bash
accelerate launch \
  --config_file ./Lara/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  ./Lara/training/train_lara.py \
  --config_yaml ./scripts/config/lara_so101_baseline.yaml \
  trainer.max_train_steps=1000 \
  datasets.vla_data.per_device_batch_size=1
```

## Action Head Baseline

The current adapter is implemented in:

```text
Lara/model/framework/act.py
```

It:

- deep-copies config before setting action-head cross-attention dim
- keeps action labels in fp32
- checks latent/body token shapes before concatenation
- prepends latent action tokens to embodied action tokens
- adds token-type embeddings for latent/body token streams
- trains the flow action head on the configured `action_horizon`
- treats `future_actions` as a strict horizon-aligned target while keeping legacy `action` tail-slicing as a fallback
- when the optional latent-action head is enabled, reconstructs the executable latent action chunk and logs raw, weight, and weighted latent/MoE loss components separately

The flow head implementation is in:

```text
Lara/model/modules/action_model/GR00T_ActionHeader.py
```

It now reads `action_horizon` directly and supports weighted action loss so the executable first 10 frames can receive higher weight than the auxiliary long-horizon tail.

## Verification Notes

The latest local verification performed:

```bash
python3 -m py_compile \
  Lara/model/framework/act.py \
  Lara/model/framework/Lara_core.py \
  Lara/model/modules/action_model/GR00T_ActionHeader.py \
  Lara/model/modules/action_model/LayerwiseFM_ActionHeader.py
```

Use `scripts/smoke_lara_real_components.py --instantiate --run-step` inside the training environment for the full Qwen/V-JEPA component smoke check.

## Acknowledgements

This codebase builds on:

- [VLA-JEPA](https://arxiv.org/abs/2602.10098)
- [starVLA](https://github.com/starVLA/starVLA)
- [V-JEPA2](https://github.com/facebookresearch/vjepa2)
- GR00T-style flow-matching action decoding ideas

## Citation

If you use the upstream VLA-JEPA components, cite:

```bibtex
@misc{vlajepa2026,
  title={VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
  author={Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
  year={2026},
  eprint={2602.10098},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2602.10098}
}
```
