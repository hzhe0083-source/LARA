#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
scripts/local/write_local_libero_config.sh >/dev/null
uv run python scripts/local/localize_libero_run_configs.py >/dev/null

stages=(
  dense_5000step_action30_bs4x2
  latent_5000step_action30_bs6x2_from_dense
  experts_residual_hardtop2_scale008_max006_warm800_cost004_norm001_7000step_action30_bs8x2_from_latent
  router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2
)

ok=1
for stage in "${stages[@]}"; do
  base="runs/libero100_complete20g/$stage"
  echo "===== $stage"
  for p in "$base/config.yaml" "$base/final_model/pytorch_model.pt"; do
    if [ -e "$p.raysync.downloading" ]; then
      echo "INCOMPLETE $p (rayfile download marker exists)"
      ok=0
    elif [ -e "$p" ]; then
      if [ "$p" = "$base/final_model/pytorch_model.pt" ]; then
        size_bytes=$(stat -c '%s' "$p")
        if [ "$size_bytes" -lt 5000000000 ]; then
          echo "INCOMPLETE $p (${size_bytes} bytes)"
          ok=0
        else
          printf 'FOUND %s (%s)\n' "$p" "$(du -h "$p" | awk '{print $1}')"
        fi
      else
        echo "FOUND $p"
      fi
    else
      echo "MISSING $p"
      ok=0
    fi
  done
done

if [ -e .cache/libero/assets/scenes/libero_living_room_tabletop_base_style.xml ]; then
  echo "FOUND LIBERO assets"
else
  echo "MISSING LIBERO assets"
  ok=0
fi

exit $((1 - ok))
