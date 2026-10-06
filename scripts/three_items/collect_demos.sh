#!/usr/bin/env bash
# Stage 5a: record scripted demonstrations into data/three_items_demos.
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/three_items/collect_demos.py "$@"
