#!/usr/bin/env bash
# Step 2b: download part of HuggingFaceVLA/libero (basket tasks) as a local dataset.
set -euo pipefail
source "$(dirname "$0")/env.sh"
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/three_items/prepare_libero_subset.py "$@"
