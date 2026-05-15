# LARA

LARA is the local research codebase for adapting VLA-JEPA pretraining to SO101 robot control.

The current implementation is an engineering baseline, not the full latent-action MoE system from the LARA paper. It uses the VLA-JEPA pretrain representation, extracts latent action tokens from the Qwen/V-JEPA token stream, and conditions a GR00T-style flow-matching action head on:

```text
latent action tokens + embodied action tokens -> continuous follower-arm action chunk
```

The SO101 path intentionally does not default to the VLA-JEPA Real-world checkpoint, because that checkpoint is adapted to other robot embodiments. For SO101, the training target is the follower arm `action` and `state` from the local LeRobot dataset.

## Implementation Status

The manuscript in [`document/LARA_collapse_paper.tex`](./document/LARA_collapse_paper.tex) describes the intended full LARA algorithm. The code in this repository has only implemented the VLA/action-baseline part so far.

See [`document/IMPLEMENTATION_GAP.md`](./document/IMPLEMENTATION_GAP.md) for the current paper-to-code gap list and recommended implementation order.

Completed in code:

- SO101 single-arm LeRobot dataset support.
- VLA-JEPA Pretrain checkpoint loading.
- Latent-token conditioned flow action head baseline.
- Long-prediction / short-execution horizon setup:
  - `action_horizon: 60`
  - `execution_horizon: 10`
  - at 30 Hz, predict 2.0 seconds and execute the first 0.333 seconds before re-observing.

Experimental scaffolding exists but is not complete or validated:

- Stage-1 latent action head scaffold with posterior encoder, VQ codebook, optional code-usage regularization, context-only prior, and optional execution/prediction boundary-state transition loss (`use_latent_action_head: false`, `lara_use_transition_head: false` by default).
- Stage-2 MoE/router scaffold with residual token experts, optional direct action-chunk experts, optional routed direct-expert action output, posterior responsibility from latent tokens or per-expert action reconstruction losses, optional posterior floor/top-r smoothing, LeRobot trajectory ids for episode-level resident pool targets, reusable episode-level resident pool masks, chunk-level top-k routing inside the resident pool, optional balance/stickiness/expert-diversity/entropy stabilizers, and route collapse diagnostics (`use_lara_moe: false` by default).
- Utility calibration scaffold with optional action-loss utility labels, an optional supervised route utility head, candidate value/progress/uncertainty/cost scoring helpers, centered utility regression, and pairwise ranking losses (`lara_utility_loss_weight: 0.0`, `lara_utility_head_loss_weight: 0.0`, `lara_use_action_loss_utility: false`, `lara_use_utility_head: false` by default).
- Minimal dummy-batch smoke coverage exists for `ActionHeadAdapter` forward and prediction shapes.

Described in the paper but not implemented yet:

- production-ready latent action training
- validated MoE action experts that directly model or adapt action chunks in full SO101 training
- real counterfactual utility scoring from value/progress/latent-state or closed-loop evaluator signals beyond action-loss utility labels
- validated transition-state training with real SO101 boundary targets
- full resident-pool training/evaluation beyond static and unit tests
- matched-compute and matched-resident-expert experiments

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
  config/lara_so101_ft.yaml   SO101 fine-tuning config
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

At 30 Hz:

```text
action_horizon = 60    -> predict 2.0 seconds
execution_horizon = 10 -> execute 0.333 seconds
```

During inference, `predict_action` returns:

- `normalized_actions`: full 60-frame prediction.
- `execution_normalized_actions`: first 10 frames for closed-loop execution.
- `resident_pool_mask`: when MoE routing is enabled and no mask is supplied, the episode-level resident expert pool selected for reuse on later chunks.

For MoE experiments, pass the returned `resident_pool_mask` back into later `predict_action` calls for the same episode so the chunk router chooses sparse experts inside a stable episode pool.

## Training

Run SO101 fine-tuning with:

```bash
accelerate launch \
  --config_file ./Lara/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./Lara/training/train_lara.py \
  --config_yaml ./scripts/config/lara_so101_ft.yaml
```

For fewer GPUs, change `--num_processes`.

You can also override config values from the CLI:

```bash
accelerate launch \
  --config_file ./Lara/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  ./Lara/training/train_lara.py \
  --config_yaml ./scripts/config/lara_so101_ft.yaml \
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

A full model instantiation smoke test was not run in the system Python because that interpreter did not have `torch` installed. Use the `Lara` conda environment for training or runtime checks.

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
