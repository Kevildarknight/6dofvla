#!/usr/bin/env bash
# Evaluate a policy on the custom LIBERO task "put all three items in the basket".
# Usage: bash scripts/run_triple_basket.sh <policy_path> [output_dir] [n_episodes]
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
policy="${1:?Usage: bash scripts/run_triple_basket.sh <policy_path> [output_dir] [n_episodes]}"
output_dir="${2:-outputs/triple_basket_$(date +%Y%m%d_%H%M%S)}"
episodes="${3:-10}"
[[ -x .venv/bin/lerobot-eval ]] || { echo "Run: bash scripts/install.sh" >&2; exit 1; }
.venv/bin/python -m pip install -q -e plugins/lerobot_env_triple_basket
export LIBERO_CONFIG_PATH="$repo_root/.libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
[[ -f plugins/lerobot_env_triple_basket/src/lerobot_env_triple_basket/data/init_states/put_all_three_items_in_the_basket.pruned_init ]] \
  || .venv/bin/python scripts/triple_basket/generate_init_states.py

.venv/bin/lerobot-eval \
  --policy.path="$policy" --policy.device=cuda \
  --env.type=libero --env.task=libero_triple_basket --env.task_ids='[0]' \
  --env.episode_length=900 \
  --eval.batch_size=1 --eval.n_episodes="$episodes" --seed=1000 \
  --output_dir="$output_dir"

.venv/bin/python - "$output_dir/eval_info.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text())
t = d["per_task"][0]; s = t["metrics"]["successes"]
print(f"successes: {s}  ({sum(map(bool, s))}/{len(s)})")
print(f"Evaluator result: {p}\nRaw videos: {p.parent / 'videos'}")
PY
