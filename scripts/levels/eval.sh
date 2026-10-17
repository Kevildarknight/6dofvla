#!/usr/bin/env bash
# Evaluate a policy on one difficulty-level task with lerobot-eval.
# Usage: bash scripts/levels/eval.sh <task_id> [policy_path] [n_episodes] [output_dir] [extra lerobot-eval args...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
task_id="${1:?task id required (see plugins/.../levels.py)}"
policy_path="${2:-models/smolvla_libero}"
n_episodes="${3:-5}"
output_dir="${4:-outputs/levels_eval_${task_id}_$(date +%Y%m%d_%H%M%S)}"

.venv/bin/lerobot-eval \
  --policy.path="$policy_path" \
  --policy.device=cuda \
  --env.type=libero_levels \
  --env.task_ids="[$task_id]" \
  --eval.batch_size=1 \
  --eval.n_episodes="$n_episodes" \
  --seed=1000 \
  --output_dir="$output_dir" \
  "${@:5}"

.venv/bin/python - "$output_dir" "$task_id" <<'PY'
import json
import sys
from pathlib import Path

out, task_id = Path(sys.argv[1]), int(sys.argv[2])
data = json.loads((out / "eval_info.json").read_text())
task = next(t for t in data["per_task"] if t["task_group"] == "libero_levels" and t["task_id"] == task_id)
successes = task["metrics"]["successes"]
print(f"libero_levels task {task_id}: {sum(successes)}/{len(successes)} episodes successful")
print(f"Evaluator result: {out / 'eval_info.json'}")
print(f"Raw rollout videos: {out / f'videos/libero_levels_{task_id}'}")
PY
