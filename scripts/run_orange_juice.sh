#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [[ $# -gt 1 ]]; then
  echo "Expected at most one output directory argument." >&2
  exit 2
fi
if [[ $# -eq 1 && "$1" == "--help" ]]; then
  echo "Usage: bash scripts/run_orange_juice.sh [output_directory]"
  exit 0
fi
if [[ ! -x .venv/bin/lerobot-eval || ! -d models/smolvla_libero || ! -f .libero/config.yaml ]]; then
  echo "Installation is incomplete. Run: bash scripts/install.sh" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  output_dir="$1"
else
  output_dir="outputs/orange_juice_$(date +%Y%m%d_%H%M%S)"
fi
export LIBERO_CONFIG_PATH="$repo_root/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/lerobot-eval \
  --policy.path=models/smolvla_libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[9]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=1000 \
  --output_dir="$output_dir"

.venv/bin/python - "$output_dir/eval_info.json" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
data = json.loads(result_path.read_text())
task = next(item for item in data["per_task"] if item["task_group"] == "libero_object" and item["task_id"] == 9)
success = task["metrics"]["successes"][0]
print(f"LIBERO Object task 9 success: {success}")
print(f"Evaluator result: {result_path}")
print(f"Raw rollout video: {result_path.parent / 'videos/libero_object_9/eval_episode_0.mp4'}")
PY
