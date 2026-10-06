#!/usr/bin/env bash
# Stage 6: evaluate a policy on the three-butter task with lerobot-eval.
# Usage: bash scripts/three_items/eval.sh [policy_path] [n_episodes] [output_dir]
set -euo pipefail
source "$(dirname "$0")/env.sh"

policy_path="${1:-models/smolvla_libero}"
n_episodes="${2:-10}"
output_dir="${3:-outputs/three_items_eval_$(date +%Y%m%d_%H%M%S)}"

# Every episode starts from the same fixed layout; the seed only affects the policy's sampling noise.
.venv/bin/lerobot-eval \
  --policy.path="$policy_path" \
  --policy.device=cuda \
  --env.type=libero_three_items \
  --eval.batch_size=1 \
  --eval.n_episodes="$n_episodes" \
  --seed=1000 \
  --output_dir="$output_dir"

.venv/bin/python - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
data = json.loads((out / "eval_info.json").read_text())
task = next(t for t in data["per_task"] if t["task_group"] == "libero_three_items" and t["task_id"] == 0)
successes = task["metrics"]["successes"]
print(f"libero_three_items task 0: {sum(successes)}/{len(successes)} episodes successful")
for i, ok in enumerate(successes):
    print(f"  episode {i}: successes[{i}] = {ok}")
print(f"Evaluator result: {out / 'eval_info.json'}")
print(f"Raw rollout videos: {out / 'videos/libero_three_items_0'}")
PY
