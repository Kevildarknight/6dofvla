#!/usr/bin/env bash
# Record scripted demonstrations for one difficulty-level task.
# Usage: bash scripts/levels/collect_demos.sh <task_id> [extra collect_demos.py args...]
#   Example: bash scripts/levels/collect_demos.sh 4 --episodes 50 --noise 1.0
set -euo pipefail
source "$(dirname "$0")/env.sh"
task_id="${1:?task id required (see plugins/.../levels.py)}"
shift
name=$(.venv/bin/python -c "from lerobot_env_libero_three_items.levels import TASKS; print(TASKS[$task_id].name)")
.venv/bin/python scripts/three_items/collect_demos.py --suite levels --task-id "$task_id" \
  --root "data/levels_$name" --repo-id "local/levels_$name" "$@"
