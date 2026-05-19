#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/local/write_local_libero_config.sh >/dev/null
uv run python scripts/local/localize_libero_run_configs.py >/dev/null
export LIBERO_CONFIG_PATH="$PWD/.libero"
uv run python - <<'PY'
import importlib
mods = ['torch','libero','robosuite','mujoco','transformers','cv2','imageio','websockets']
for m in mods:
    mod = importlib.import_module(m)
    print(f'{m}: OK {getattr(mod, "__version__", "")}')
from libero.libero import benchmark, get_libero_path
print('bddl_files:', get_libero_path('bddl_files'))
print('assets:', get_libero_path('assets'))
bench = benchmark.get_benchmark_dict()['libero_10']()
task = bench.get_task(0)
print('libero_10 tasks:', bench.n_tasks)
print('task0:', task.problem_folder, task.bddl_file, task.language)
print('init_states:', bench.get_task_init_states(0).shape)
PY
ok=1
for p in \
  runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/config.yaml \
  runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/dataset_statistics.json \
  .cache/libero/assets/scenes/libero_living_room_tabletop_base_style.xml \
  models/Qwen3-VL-2B-Instruct/config.json \
  models/vjepa2-vitl-fpc64-256/config.json; do
  if [ -e "$p" ]; then
    echo "FOUND $p"
  else
    echo "MISSING $p"
    ok=0
  fi
done

if find -L models/Qwen3-VL-2B-Instruct -maxdepth 2 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -size +100M -print -quit 2>/dev/null | grep -q .; then
  echo "FOUND Qwen3-VL model weights"
else
  echo "MISSING Qwen3-VL model weights"
  ok=0
fi

if find -L models/vjepa2-vitl-fpc64-256 -maxdepth 2 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -size +100M -print -quit 2>/dev/null | grep -q .; then
  echo "FOUND V-JEPA2 model weights"
else
  echo "MISSING V-JEPA2 model weights"
  ok=0
fi

CKPT=runs/libero100_complete20g/router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2/final_model/pytorch_model.pt
if [ -e "$CKPT.raysync.downloading" ]; then
  echo "INCOMPLETE $CKPT (rayfile download marker exists)"
  ok=0
elif [ -e "$CKPT" ]; then
  size_bytes=$(stat -c '%s' "$CKPT")
  if [ "$size_bytes" -lt 5000000000 ]; then
    echo "INCOMPLETE $CKPT (${size_bytes} bytes, expected about 5.91GB)"
    ok=0
  else
    echo "FOUND $CKPT (${size_bytes} bytes)"
  fi
else
  echo "MISSING $CKPT"
  ok=0
fi

exit $((1 - ok))
