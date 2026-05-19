# Cloud-to-local files for LIBERO eval debug

Copy these from the cloud instance into this local repository layout.

## Local runtime

The local `uv` environment in this repository has been checked with:

```bash
scripts/local/write_local_libero_config.sh
scripts/local/check_local_libero_eval_env.sh
```

Verified imports:

- `torch==2.6.0+cu124`
- `libero==0.1.1`
- `robosuite==1.4.0`
- `mujoco==3.8.1`
- `transformers==4.57.0`
- `opencv-python`
- `imageio`
- `websockets==15.0.1`

If using a container instead of `uv`, use the repo image definition:

```bash
docker build -f Dockerfile.lara-libero100 -t lara-libero100:local .
```

That Dockerfile is based on:

```text
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

It intentionally does not include model weights or LIBERO100 data; mount/copy the files below separately.

## Local all-stage layout

The local debug setup expects all four stages:

```text
runs/libero100_complete20g/dense_5000step_action30_bs4x2/
runs/libero100_complete20g/latent_5000step_action30_bs6x2_from_dense/
runs/libero100_complete20g/experts_residual_hardtop2_scale008_max006_warm800_cost004_norm001_7000step_action30_bs8x2_from_latent/
runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/
```

Each stage needs at least:

```text
config.yaml
final_model/pytorch_model.pt
```

The router stage also needs:

```text
dataset_statistics.json
```

Base models are expected at:

```text
models/Qwen3-VL-2B-Instruct
models/vjepa2-vitl-fpc64-256
models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt
```

The first two may be symlinks to an existing local HuggingFace cache. Run configs
are localized before checks/eval by:

```bash
uv run python scripts/local/localize_libero_run_configs.py
```

## Files to copy

Cloud source -> local destination:

- `/root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/config.yaml`
  -> `runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/config.yaml`
- `/root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/dataset_statistics.json`
  -> `runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/dataset_statistics.json`
- `/root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt`
  -> `runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt`
- `/root/.cache/libero/assets/`
  -> `.cache/libero/assets/`
- `/root/private_data/benchmark_data/raw/libero100/`
  -> `data/libero100/`

Suggested cloud packing command after the instance is back online. If this
repository has already been synced to the cloud instance, use:

```bash
cd /root/private_data/LARA
bash scripts/local/pack_remote_libero_eval_bundle.sh
```

If the cloud checkout does not contain `scripts/local/pack_remote_libero_eval_bundle.sh`,
run this self-contained command instead:

```bash
BUNDLE=/root/private_data/libero_eval_local_bundle.tgz
RUN_DIR=/root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2
for p in \
  "$RUN_DIR/config.yaml" \
  "$RUN_DIR/dataset_statistics.json" \
  "$RUN_DIR/final_model/pytorch_model.pt" \
  /root/.cache/libero/assets \
  /root/private_data/benchmark_data/raw/libero100; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 1; }
done
cd /
tar -czf "$BUNDLE" \
  root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/config.yaml \
  root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/dataset_statistics.json \
  root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt \
  root/.cache/libero/assets \
  root/private_data/benchmark_data/raw/libero100
(cd "$(dirname "$BUNDLE")" && sha256sum "$(basename "$BUNDLE")" > "$(basename "$BUNDLE").sha256")
du -sh "$BUNDLE"
cat "$BUNDLE.sha256"
```

This creates:

```text
/root/private_data/libero_eval_local_bundle.tgz
/root/private_data/libero_eval_local_bundle.tgz.sha256
```

Download both files with the platform file tool, `rayfile c`, or `scp`.
If the server is powered off but the RayFile file space is still accessible, you
do not need to boot the instance. Download the run files directly from RayFile:

```bash
RAYFILE_PASSWORD='...' scripts/local/pull_remote_libero_eval_rayfile.sh
```

To download all four stage checkpoints directly from RayFile:

```bash
RAYFILE_PASSWORD='...' scripts/local/pull_remote_libero_all_stages_rayfile.sh
```

To download base models from RayFile if the local `models/` symlinks or cache are
not available:

```bash
RAYFILE_PASSWORD='...' scripts/local/pull_remote_base_models_rayfile.sh
```

The RayFile file space maps `/root/private_data` to `/`, so this script pulls
from:

```text
/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/
```

The root user's `/root/.cache/libero/assets` directory may not be visible from
RayFile. In that case, use the official LIBERO assets mirror locally:

```bash
scripts/local/download_libero_assets.sh
```

For direct `scp` download, after the bundle exists remotely:

```bash
scripts/local/pull_remote_libero_eval_bundle_scp.sh root@qdai.scnet.cn 50199 .
```

After downloading the archive locally into this repository root, run:

```bash
bash scripts/local/unpack_cloud_libero_eval_bundle.sh libero_eval_local_bundle.tgz
scripts/local/run_local_eval_video_debug.sh
```

## Local verification

After downloads complete:

```bash
scripts/local/check_local_libero_all_stages.sh
scripts/local/check_local_libero_eval_env.sh
scripts/local/run_local_eval_video_debug.sh
```

The smoke eval writes:

```text
runs/local_eval_debug_*/eval/eval_summary.json
runs/local_eval_debug_*/eval/rollout_records.jsonl
runs/local_eval_debug_*/eval/sampled_route_traces.jsonl
runs/local_eval_debug_*/videos/*.mp4
```
