#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/local/write_local_libero_config.sh >/dev/null
uv run python scripts/local/localize_libero_run_configs.py >/dev/null
export MUJOCO_GL=${MUJOCO_GL:-egl}
export LIBERO_CONFIG_PATH="$PWD/.libero"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CKPT=${1:-runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt}
OUT=${2:-runs/local_eval_debug_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/eval" "$OUT/videos"
uv run python scripts/eval_libero100_headless.py \
  --checkpoint "$CKPT" \
  --output_dir "$OUT/eval" \
  --video_out_dir "$OUT/videos" \
  --task_suite_name libero_10 \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --start_server \
  --cuda 0 \
  --use_bf16 \
  --host 127.0.0.1 \
  --port 12000 \
  --server_startup_delay 60 \
  --sample_full_route_every 1
printf '\nOUT=%s\n' "$OUT"
find "$OUT/videos" -type f -maxdepth 1 -print
cat "$OUT/eval/eval_summary.json"
