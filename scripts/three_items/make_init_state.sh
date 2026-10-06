#!/usr/bin/env bash
# Regenerate the fixed initial state after changing LAYOUT_XY in task.py.
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/three_items/make_init_state.py "$@"
