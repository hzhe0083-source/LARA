#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  cat >&2 <<'USAGE'
Usage:
  scripts/local/pull_remote_libero_eval_bundle_scp.sh <ssh_target> <port> [local_dir]

Example:
  scripts/local/pull_remote_libero_eval_bundle_scp.sh root@qdai.scnet.cn 50199 .

This downloads:
  /root/private_data/libero_eval_local_bundle.tgz
  /root/private_data/libero_eval_local_bundle.tgz.sha256
USAGE
  exit 2
fi
TARGET=$1
PORT=$2
LOCAL_DIR=${3:-.}
REMOTE_BUNDLE=/root/private_data/libero_eval_local_bundle.tgz
mkdir -p "$LOCAL_DIR"
scp -P "$PORT" "$TARGET:$REMOTE_BUNDLE" "$LOCAL_DIR/"
scp -P "$PORT" "$TARGET:$REMOTE_BUNDLE.sha256" "$LOCAL_DIR/" || true
if [ -f "$LOCAL_DIR/libero_eval_local_bundle.tgz.sha256" ]; then
  (cd "$LOCAL_DIR" && sha256sum -c libero_eval_local_bundle.tgz.sha256)
fi
printf 'Downloaded to: %s/libero_eval_local_bundle.tgz\n' "$LOCAL_DIR"
