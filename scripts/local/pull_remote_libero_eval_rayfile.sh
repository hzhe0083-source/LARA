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
  echo "Example: RAYFILE_PASSWORD='...' scripts/local/pull_remote_libero_eval_rayfile.sh" >&2
  exit 2
fi

RUN_ID=${RUN_ID:-router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2}
REMOTE_RUN=${REMOTE_RUN:-/vlajepa_runs/libero100_complete20g/$RUN_ID}
LOCAL_RUN=${LOCAL_RUN:-runs/libero100_complete20g/$RUN_ID}

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

rayfile_download "$REMOTE_RUN/config.yaml" "$LOCAL_RUN/"
rayfile_download "$REMOTE_RUN/dataset_statistics.json" "$LOCAL_RUN/"
rayfile_download "$REMOTE_RUN/final_model/pytorch_model.pt" "$LOCAL_RUN/final_model/"

scripts/local/write_local_libero_config.sh >/dev/null
if [ ! -e .cache/libero/assets/scenes/libero_living_room_tabletop_base_style.xml ]; then
  echo "LIBERO assets are missing. Download them with:" >&2
  echo "  scripts/local/download_libero_assets.sh" >&2
fi

ls -lh \
  "$LOCAL_RUN/config.yaml" \
  "$LOCAL_RUN/dataset_statistics.json" \
  "$LOCAL_RUN/final_model/pytorch_model.pt"
