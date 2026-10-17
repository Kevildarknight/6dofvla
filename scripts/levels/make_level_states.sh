#!/usr/bin/env bash
# Write and check the initial state of every difficulty-level task (CPU only).
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/levels/make_level_states.py "$@"
