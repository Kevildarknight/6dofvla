#!/usr/bin/env bash
# Fine-tune the LIBERO SmolVLA checkpoint on the three-butter data.
# Usage: [DATASET=three_items_mix|three_items_demos] bash scripts/three_items/train.sh [output_dir] [extra args...]
# DATASET defaults to three_items_mix: the demos plus part of the original LIBERO data (build_mix.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"

output_dir="${1:-outputs/three_items_train_$(date +%Y%m%d_%H%M%S)}"
shift || true
dataset="${DATASET:-three_items_mix}"
if [[ ! -f "data/$dataset/meta/info.json" ]]; then
  echo "data/$dataset not found. Run collect_demos.sh (and build_mix.sh for the mix) first." >&2
  exit 1
fi

# Starts from the LIBERO fine-tuned checkpoint, so only the action expert and
# state projection are trained (the checkpoint's defaults); the vision-language
# backbone stays frozen. Normalisation statistics come from the new dataset.
.venv/bin/lerobot-train \
  --policy.path=models/smolvla_libero \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/$dataset" \
  --dataset.root="data/$dataset" \
  --batch_size="${BATCH_SIZE:-4}" \
  --steps="${STEPS:-20000}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --log_freq=100 \
  --seed=1000 \
  --output_dir="$output_dir" \
  "$@"
echo "Checkpoints: $output_dir/checkpoints/last/pretrained_model"
