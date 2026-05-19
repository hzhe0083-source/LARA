#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/local/write_local_libero_config.sh >/dev/null
BUNDLE=${1:-libero_eval_local_bundle.tgz}
if [ ! -f "$BUNDLE" ]; then
  echo "missing bundle: $BUNDLE" >&2
  exit 1
fi
if [ -f "${BUNDLE}.sha256" ]; then
  expected=$(awk '{print $1; exit}' "${BUNDLE}.sha256")
  actual=$(sha256sum "$BUNDLE" | awk '{print $1}')
  if [ "$expected" != "$actual" ]; then
    echo "sha256 mismatch for $BUNDLE" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
  echo "$BUNDLE: OK"
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$BUNDLE" -C "$TMP"
RUN_SRC="$TMP/root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2"
mkdir -p runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model
cp "$RUN_SRC/config.yaml" runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/config.yaml
cp "$RUN_SRC/dataset_statistics.json" runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/dataset_statistics.json
cp "$RUN_SRC/final_model/pytorch_model.pt" runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt
mkdir -p .cache/libero data
rm -rf .cache/libero/assets data/libero100
cp -a "$TMP/root/.cache/libero/assets" .cache/libero/assets
cp -a "$TMP/root/private_data/benchmark_data/raw/libero100" data/libero100
scripts/local/check_local_libero_eval_env.sh
