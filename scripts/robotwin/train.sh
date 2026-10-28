#!/usr/bin/env bash
# Fine-tune SmolVLA base on a converted RoboTwin Panthera dataset.
# Usage: [DATASET=robotwin_<config>] bash scripts/robotwin/train.sh [output_dir] [extra lerobot-train args...]
# DATASET defaults to robotwin_move_pillbottle_pad_panthera_single_30hz (convert.sh output under data/).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

output_dir="${1:-outputs/robotwin_train_$(date +%Y%m%d_%H%M%S)}"
shift || true
dataset="${DATASET:-robotwin_move_pillbottle_pad_panthera_single_30hz}"
if [[ ! -f "data/$dataset/meta/info.json" ]]; then
  echo "data/$dataset not found. Run collect.sh and convert.sh first." >&2
  exit 1
fi

if [[ ! -f models/smolvla_base/model.safetensors ]]; then
  HF_HUB_DISABLE_XET=1 .venv/bin/hf download lerobot/smolvla_base \
    --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
    --local-dir models/smolvla_base
fi

# Starts from SmolVLA base, not the LIBERO checkpoint: the Panthera state and
# action are absolute joint positions, unlike LIBERO's end-effector deltas.
# Only the action expert and state projection are trained (base defaults); the
# vision-language backbone stays frozen. The base model names its cameras
# camera1..camera3; head and wrist are mapped to the first two and camera3 is
# left out. Normalisation statistics come from the new dataset.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/lerobot-train \
  --policy.path=models/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/$dataset" \
  --dataset.root="data/$dataset" \
  --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size="${BATCH_SIZE:-4}" \
  --steps="${STEPS:-20000}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --log_freq=100 \
  --seed=1000 \
  --output_dir="$output_dir" \
  "$@"
echo "Checkpoints: $output_dir/checkpoints/last/pretrained_model"
