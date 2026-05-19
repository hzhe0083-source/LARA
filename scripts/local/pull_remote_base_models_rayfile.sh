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

wait_for_any_marker() {
  local markers
  while true; do
    markers=$(find runs models -name '*.raysync.downloading' -print 2>/dev/null || true)
    if [ -z "$markers" ]; then
      return
    fi
    echo "waiting for existing RayFile downloads:"
    echo "$markers"
    while IFS= read -r marker; do
      du -h "${marker%.raysync.downloading}" 2>/dev/null || true
    done <<<"$markers"
    sleep 60
  done
}

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

has_model_weights() {
  local dir=$1
  find -L "$dir" -maxdepth 2 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -size +100M -print -quit 2>/dev/null | grep -q .
}

download_model_dir() {
  local remote_dir=$1
  local local_dir=$2
  local label=$3

  wait_for_any_marker
  if [ -e "$local_dir/config.json" ] && has_model_weights "$local_dir"; then
    echo "exists complete-ish: $local_dir"
    du -sh "$local_dir"
    return
  fi

  echo "===== downloading $label"
  rm -f "$local_dir"/*.raysync.downloading 2>/dev/null || true
  rayfile_download "$remote_dir" "$(dirname "$local_dir")/"
  wait_for_any_marker

  if [ ! -e "$local_dir/config.json" ]; then
    echo "missing $local_dir/config.json after download" >&2
    exit 3
  fi
  if ! has_model_weights "$local_dir"; then
    echo "missing large model weights under $local_dir after download" >&2
    exit 3
  fi
  du -sh "$local_dir"
}

download_model_dir /Qwen3-VL-2B-Instruct models/Qwen3-VL-2B-Instruct Qwen3-VL-2B-Instruct
download_model_dir /vjepa2-vitl-fpc64-256 models/vjepa2-vitl-fpc64-256 vjepa2-vitl-fpc64-256

uv run python scripts/local/localize_libero_run_configs.py
scripts/local/check_local_libero_eval_env.sh
