#!/usr/bin/env bash
# Stage 5b: fine-tune the LIBERO SmolVLA checkpoint on the three-butter demos.
# Usage: bash scripts/three_items/train.sh [output_dir] [extra lerobot-train args...]
set -euo pipefail
source "$(dirname "$0")/env.sh"

output_dir="${1:-outputs/three_items_train_$(date +%Y%m%d_%H%M%S)}"
shift || true
if [[ ! -d data/three_items_demos ]]; then
  echo "No demonstrations found. Run: bash scripts/three_items/collect_demos.sh" >&2
  exit 1
fi

# Starts from the LIBERO fine-tuned checkpoint, so only the action expert and
# state projection are trained (the checkpoint's defaults); the vision-language
# backbone stays frozen. Normalisation statistics come from the new dataset.
.venv/bin/lerobot-train \
  --policy.path=models/smolvla_libero \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/three_items_demos \
  --dataset.root=data/three_items_demos \
  --batch_size="${BATCH_SIZE:-4}" \
  --steps="${STEPS:-20000}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --log_freq=100 \
  --seed=1000 \
  --output_dir="$output_dir" \
  "$@"
echo "Checkpoints: $output_dir/checkpoints/last/pretrained_model"
