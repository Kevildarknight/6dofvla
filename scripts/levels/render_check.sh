#!/usr/bin/env bash
# Render a scripted rollout of every difficulty-level task (GPU, ~10 min).
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/levels/render_check.py "$@"
