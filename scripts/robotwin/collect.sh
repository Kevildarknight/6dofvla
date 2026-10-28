#!/usr/bin/env bash
# Collect Panthera demonstrations in RoboTwin with a task config kept in this repository.
# Usage: bash scripts/robotwin/collect.sh [task_name] [config_name] [episode_num]
# Defaults: move_pillbottle_pad, move_pillbottle_pad_panthera_single_30hz, the config's episode_num.
# RoboTwin reads configs only from its own task_config/, so the config is copied there first
# (a new file; no RoboTwin file is edited). Output: $ROBOTWIN_DIR/data/<task_name>/<config_name>/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/panthera/Panthera-HT_RoboTwin/RoboTwin}"
task_name="${1:-move_pillbottle_pad}"
config_name="${2:-move_pillbottle_pad_panthera_single_30hz}"
episode_args=()
if [[ -n "${3:-}" ]]; then
  episode_args=(--episode-num "$3")
fi

cp "$here/$config_name.yml" "$ROBOTWIN_DIR/task_config/$config_name.yml"
cd "$ROBOTWIN_DIR"
conda run --no-capture-output -n RoboTwin \
  python -u script/collect_data.py "$task_name" "$config_name" "${episode_args[@]}"
