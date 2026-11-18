#!/usr/bin/env bash
# Train LeRobot's Diffusion Policy on a converted RoboTwin Panthera dataset.
# Usage: [DATASET=robotwin_<config>] bash scripts/robotwin/train_diffusion.sh [output_dir] [extra lerobot-train args...]
# Env overrides: BATCH_SIZE (16, the most that fits 4 GiB), STEPS (60000), SAVE_FREQ (20000).
# Both cameras are resized to 240x320 (panthera_policies.py) and randomly cropped to 90 %.
# U-Net down_dims (256, 512, 1024), the smaller variant of the paper, so training fits 4 GiB.
# Horizon 32 frames (about 1 s) from 2 observation steps; 16 actions are executed per
# prediction. Evaluate with eval.sh, which serves any LeRobot policy type.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

output_dir="${1:-outputs/robotwin_diffusion_$(date +%Y%m%d_%H%M%S)}"
shift || true
dataset="${DATASET:-robotwin_move_pillbottle_pad_panthera_single_30hz_200}"
if [[ ! -f "data/$dataset/meta/info.json" ]]; then
  echo "data/$dataset not found. Run collect.sh and convert.sh first." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/robotwin/train_diffusion.py \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.resize_shape='[240, 320]' \
  --policy.crop_ratio=0.9 \
  --policy.horizon=32 \
  --policy.n_action_steps=16 \
  --policy.drop_n_last_frames=15 \
  --policy.down_dims='[256, 512, 1024]' \
  --dataset.repo_id="local/$dataset" \
  --dataset.root="data/$dataset" \
  --batch_size="${BATCH_SIZE:-16}" \
  --steps="${STEPS:-60000}" \
  --save_freq="${SAVE_FREQ:-20000}" \
  --log_freq=100 \
  --seed=1000 \
  --output_dir="$output_dir" \
  "$@"
echo "Checkpoints: $output_dir/checkpoints/last/pretrained_model"
