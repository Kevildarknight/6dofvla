#!/usr/bin/env bash
# Evaluate a SmolVLA checkpoint in RoboTwin with RoboTwin's own evaluator and success check.
# Usage: bash scripts/robotwin/eval.sh <pretrained_model dir> [episodes] [seen|unseen]
# Env overrides: TASK_NAME, TASK_CONFIG, SEED, ACTION_STEPS (actions per request), DEVICE, PORT, ROBOTWIN_DIR,
# TEXTURES (unseen, the evaluator's default, or seen: the training backgrounds, via eval_seen_textures.py).
# SEED=-1 starts at seed 0, the same seeds as data collection.
# Starts policy_server.py in this repository's .venv, then runs RoboTwin's script/eval_policy.py
# in the RoboTwin conda env with smolvla_panthera.py as the policy. Test seeds start at
# 100000 * (SEED + 1), so they never overlap the collection seeds.
# Results: $ROBOTWIN_DIR/eval_result/<task>/smolvla_panthera/<config>/<checkpoint>/<time>/
# (_result.txt and one head-camera video per episode).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/panthera/Panthera-HT_RoboTwin/RoboTwin}"

checkpoint="$(cd "${1:?pass a pretrained_model directory}" && pwd)"
episodes="${2:-20}"
instruction_type="${3:-seen}"
task_name="${TASK_NAME:-move_pillbottle_pad}"
task_config="${TASK_CONFIG:-move_pillbottle_pad_panthera_single_30hz}"
port="${PORT:-6010}"
# e.g. outputs/robotwin_pillbottle_20k/checkpoints/020000/pretrained_model -> robotwin_pillbottle_20k_020000
ckpt_setting="$(basename "$(dirname "$(dirname "$(dirname "$checkpoint")")")")_$(basename "$(dirname "$checkpoint")")"
textures="${TEXTURES:-unseen}"
evaluator=script/eval_policy.py
if [[ "$textures" == seen ]]; then
  evaluator="$here/eval_seen_textures.py"
  ckpt_setting="${ckpt_setting}_seen-textures"
fi
if [[ "${SEED:-0}" != 0 ]]; then
  ckpt_setting="${ckpt_setting}_seed${SEED}"
fi

server_log="$(mktemp -t smolvla_server.XXXXXX.log)"
device="${DEVICE:-cuda}"
# On a 4 GiB GPU a CPU server must not open a CUDA context (about 400 MB), or
# the simulator's planner runs out of memory while training shares the GPU.
server_env=()
if [[ "$device" == cpu ]]; then
  server_env=(CUDA_VISIBLE_DEVICES=)
fi
env "${server_env[@]}" "$repo_root/.venv/bin/python" "$here/policy_server.py" "$checkpoint" --port "$port" \
  --action-steps "${ACTION_STEPS:-10}" --device "$device" > "$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
until grep -q "server ready" "$server_log"; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Policy server exited; log: $server_log" >&2
    tail -20 "$server_log" >&2
    exit 1
  fi
  sleep 2
done
echo "Policy server ready (log: $server_log)"

cp "$here/$task_config.yml" "$ROBOTWIN_DIR/task_config/$task_config.yml"
cd "$ROBOTWIN_DIR"
PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" conda run --no-capture-output -n RoboTwin \
  python -u "$evaluator" --config "$here/smolvla_panthera.yml" --overrides \
  --task_name "$task_name" \
  --task_config "$task_config" \
  --ckpt_setting "$ckpt_setting" \
  --seed "${SEED:-0}" \
  --policy_name smolvla_panthera \
  --test_num "$episodes" \
  --instruction_type "$instruction_type" \
  --port "$port"
