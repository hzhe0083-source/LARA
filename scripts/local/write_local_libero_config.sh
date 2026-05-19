#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LIBERO_ROOT=$(uv run python - <<'PY'
from pathlib import Path
import importlib.util
spec = importlib.util.find_spec('libero')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit('libero package is not installed')
root = Path(next(iter(spec.submodule_search_locations))) / 'libero'
print(root.resolve())
PY
)
mkdir -p .libero data/libero100 .cache/libero/assets
cat > .libero/config.yaml <<EOF
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PWD/data/libero100
assets: $PWD/.cache/libero/assets
EOF
echo "Wrote .libero/config.yaml"
cat .libero/config.yaml
