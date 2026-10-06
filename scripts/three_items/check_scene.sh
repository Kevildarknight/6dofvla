#!/usr/bin/env bash
# Stage 4: validate the scene and success check with the scripted controller.
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/three_items/check_scene.py "$@"
