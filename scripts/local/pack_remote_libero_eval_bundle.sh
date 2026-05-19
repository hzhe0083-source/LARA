#!/usr/bin/env bash
set -euo pipefail
BUNDLE=${1:-/root/private_data/libero_eval_local_bundle.tgz}
SHA_FILE=${BUNDLE}.sha256
RUN_DIR=/root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2
REQUIRED=(
  "$RUN_DIR/config.yaml"
  "$RUN_DIR/dataset_statistics.json"
  "$RUN_DIR/final_model/pytorch_model.pt"
  "/root/.cache/libero/assets"
  "/root/private_data/benchmark_data/raw/libero100"
)
for p in "${REQUIRED[@]}"; do
  if [ ! -e "$p" ]; then
    echo "MISSING: $p" >&2
    exit 1
  fi
done
cd /
tar -czf "$BUNDLE" \
  root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/config.yaml \
  root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/dataset_statistics.json \
  root/private_data/vlajepa_runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt \
  root/.cache/libero/assets \
  root/private_data/benchmark_data/raw/libero100
(cd "$(dirname "$BUNDLE")" && sha256sum "$(basename "$BUNDLE")" > "$(basename "$SHA_FILE")")
du -sh "$BUNDLE"
cat "$SHA_FILE"
echo "Bundle: $BUNDLE"
echo "SHA256: $SHA_FILE"
