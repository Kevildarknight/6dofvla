#!/usr/bin/env bash
# Evaluate an ACT checkpoint from act_train.sh in RoboTwin with RoboTwin's own evaluator.
# Usage: bash scripts/robotwin/act_eval.sh <checkpoint dir> [episodes] [seen|unseen]
# Env overrides: TASK_NAME, TASK_CONFIG, SEED, TEXTURES, CHECKPOINT (policy_best.ckpt), ROBOTWIN_DIR;
# SEED and TEXTURES work as in eval.sh.
# Results: $ROBOTWIN_DIR/eval_result/<task>/act_panthera/<config>/<checkpoint dir name>/<time>/
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/panthera/Panthera-HT_RoboTwin/RoboTwin}"

ckpt_dir="$(cd "${1:?pass a checkpoint dir from act_train.sh}" && pwd)"
episodes="${2:-20}"
instruction_type="${3:-seen}"
task_name="${TASK_NAME:-move_pillbottle_pad}"
task_config="${TASK_CONFIG:-move_pillbottle_pad_panthera_single_30hz}"
checkpoint="${CHECKPOINT:-policy_best.ckpt}"
ckpt_setting="$(basename "$ckpt_dir")_${checkpoint%.ckpt}"
evaluator=script/eval_policy.py
if [[ "${TEXTURES:-unseen}" == seen ]]; then
  evaluator="$here/eval_seen_textures.py"
  ckpt_setting="${ckpt_setting}_seen-textures"
fi
if [[ "${SEED:-0}" != 0 ]]; then
  ckpt_setting="${ckpt_setting}_seed${SEED}"
fi

cp "$here/$task_config.yml" "$ROBOTWIN_DIR/task_config/$task_config.yml"
cd "$ROBOTWIN_DIR"
PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore::UserWarning \
  conda run --no-capture-output -n RoboTwin \
  python -u "$evaluator" --config "$here/act_panthera.yml" --overrides \
  --task_name "$task_name" \
  --task_config "$task_config" \
  --ckpt_setting "$ckpt_setting" \
  --ckpt_dir "$ckpt_dir" \
  --checkpoint_name "$checkpoint" \
  --seed "${SEED:-0}" \
  --policy_name act_panthera \
  --test_num "$episodes" \
  --instruction_type "$instruction_type"
