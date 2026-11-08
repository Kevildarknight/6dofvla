#!/usr/bin/env bash
# Train RoboTwin's ACT baseline on a dataset from act_convert.py (single-arm Panthera, 7-D).
# Usage: bash scripts/robotwin/act_train.sh <dataset dir> [checkpoint dir] [extra imitate_episodes.py args...]
# Env overrides: EPOCHS (6000), BATCH_SIZE (8), SEED (0), ROBOTWIN_DIR.
# Hyperparameters are those of RoboTwin's policy/ACT/train.sh. An "epoch" draws one random
# timestep per training episode, so 6000 epochs on 40 training episodes are 30,000 updates.
# Checkpoints: policy_best.ckpt (lowest validation loss, used by act_eval.sh), policy_last.ckpt,
# policy_epoch_<n>_seed_<s>.ckpt every 2000 epochs, and dataset_stats.pkl.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/panthera/Panthera-HT_RoboTwin/RoboTwin}"

dataset="$(cd "${1:?pass a dataset dir from act_convert.py}" && pwd)"
shift
ckpt_dir="${1:-$repo_root/outputs/act_$(basename "$dataset")_$(date +%Y%m%d_%H%M%S)}"
shift || true
mkdir -p "$ckpt_dir"
ckpt_dir="$(cd "$ckpt_dir" && pwd)"

cd "$ROBOTWIN_DIR"
conda run --no-capture-output -n RoboTwin python -u "$here/act_train.py" "$dataset" "$ckpt_dir" \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size "${BATCH_SIZE:-8}" \
  --dim_feedforward 3200 \
  --num_epochs "${EPOCHS:-6000}" \
  --lr 1e-5 \
  --save_freq 2000 \
  --seed "${SEED:-0}" \
  "$@"
echo "Checkpoints: $ckpt_dir"
