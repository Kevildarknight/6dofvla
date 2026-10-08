#!/usr/bin/env bash
# Regenerate the saved initial states (fixed, train, heldout layouts).
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/three_items/make_init_states.py "$@"
