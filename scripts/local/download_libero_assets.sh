#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
scripts/local/write_local_libero_config.sh >/dev/null

mkdir -p .cache/libero/assets .cache/hf_libero_assets

LIBERO_CONFIG_PATH="$PWD/.libero" \
HF_HOME="$PWD/.cache/hf_libero_assets" \
HF_HUB_CACHE="$PWD/.cache/hf_libero_assets/hub" \
HF_XET_CACHE="$PWD/.cache/hf_libero_assets/xet" \
HF_HUB_DISABLE_XET=1 \
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

out = Path(".cache/libero/assets").resolve()
snapshot_download(
    repo_id="jadechoghari/libero-assets",
    repo_type="model",
    local_dir=str(out),
    max_workers=4,
)

required = [
    "scenes/libero_living_room_tabletop_base_style.xml",
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
]
for rel in required:
    p = out / rel
    if not p.exists():
        raise SystemExit(f"missing asset path: {p}")
    print(f"FOUND {p}")
PY

du -sh .cache/libero/assets
