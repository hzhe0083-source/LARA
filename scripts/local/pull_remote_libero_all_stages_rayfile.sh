#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RAYFILE_BIN=${RAYFILE_BIN:-rayfile-c}
if ! command -v "$RAYFILE_BIN" >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/rayfile-c" ]; then
    RAYFILE_BIN="$HOME/.local/bin/rayfile-c"
  else
    echo "missing rayfile-c; set RAYFILE_BIN=/path/to/rayfile-c" >&2
    exit 1
  fi
fi

RAYFILE_ADDRESS=${RAYFILE_ADDRESS:-qdefile.hpccube.com}
RAYFILE_PROXY_PORT=${RAYFILE_PROXY_PORT:-65012}
RAYFILE_USER=${RAYFILE_USER:-zhenghaoran}
RAYFILE_PASSWORD=${RAYFILE_PASSWORD:-}
if [ -z "$RAYFILE_PASSWORD" ]; then
  echo "missing RAYFILE_PASSWORD" >&2
  exit 2
fi

ROOT_REMOTE=${ROOT_REMOTE:-/vlajepa_runs/libero100_complete20g}
ROOT_LOCAL=${ROOT_LOCAL:-runs/libero100_complete20g}

STAGES=(
  dense_5000step_action30_bs4x2
  latent_5000step_action30_bs6x2_from_dense
  experts_residual_hardtop2_scale008_max006_warm800_cost004_norm001_7000step_action30_bs8x2_from_latent
  router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2
)

small_files=(
  config.yaml
  config.json
  dataset_statistics.json
  manifest.json
  metrics.jsonl
  preflight_report.json
  preflight_report.md
  guard_monitor_events.log
  router_guard.log
)

rayfile_download() {
  local src=$1
  local dst=$2
  mkdir -p "$dst"
  "$RAYFILE_BIN" \
    -a "$RAYFILE_ADDRESS" \
    -P "$RAYFILE_PROXY_PORT" \
    -u "$RAYFILE_USER" \
    -w "$RAYFILE_PASSWORD" \
    -no-meta \
    -symbolic-links follow \
    -retry 10 \
    -retrytimeout 30 \
    -o download \
    -s "$src" \
    -d "$dst"
}

maybe_download() {
  local src=$1
  local dst=$2
  if ! rayfile_download "$src" "$dst"; then
    echo "skip missing or failed optional file: $src" >&2
  fi
}

wait_for_existing_marker() {
  local local_file=$1
  local marker="$local_file.raysync.downloading"
  while [ -e "$marker" ]; do
    echo "waiting for existing RayFile download: $local_file"
    du -h "$local_file" 2>/dev/null || true
    sleep 60
  done
}

wait_for_any_existing_marker() {
  local markers
  while true; do
    markers=$(find "$ROOT_LOCAL" -name '*.raysync.downloading' -print 2>/dev/null || true)
    if [ -z "$markers" ]; then
      return
    fi
    echo "waiting for existing RayFile downloads before starting next stage:"
    echo "$markers"
    while IFS= read -r marker; do
      du -h "${marker%.raysync.downloading}" 2>/dev/null || true
    done <<<"$markers"
    sleep 60
  done
}

wait_for_any_existing_marker

for stage in "${STAGES[@]}"; do
  remote_stage="$ROOT_REMOTE/$stage"
  local_stage="$ROOT_LOCAL/$stage"
  ckpt="$local_stage/final_model/pytorch_model.pt"

  echo "===== $stage"
  mkdir -p "$local_stage/final_model"

  for file in "${small_files[@]}"; do
    if [ -e "$local_stage/$file" ]; then
      echo "exists: $local_stage/$file"
    else
      maybe_download "$remote_stage/$file" "$local_stage/"
    fi
  done

  wait_for_existing_marker "$ckpt"
  if [ -e "$ckpt" ]; then
    size_bytes=$(stat -c '%s' "$ckpt")
    if [ "$size_bytes" -ge 5000000000 ]; then
      echo "exists complete-ish: $ckpt ($size_bytes bytes)"
      continue
    fi
  fi

  rayfile_download "$remote_stage/final_model/pytorch_model.pt" "$local_stage/final_model/"
done

scripts/local/write_local_libero_config.sh >/dev/null
if [ ! -e .cache/libero/assets/scenes/libero_living_room_tabletop_base_style.xml ]; then
  scripts/local/download_libero_assets.sh
fi

scripts/local/check_local_libero_all_stages.sh
