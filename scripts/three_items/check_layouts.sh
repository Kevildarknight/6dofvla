#!/usr/bin/env bash
# Check the randomised train and held-out layouts.
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/three_items/check_layouts.py "$@"
