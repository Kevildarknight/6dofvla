#!/usr/bin/env bash
# Convert a RoboTwin Panthera single-arm output folder into a LeRobot dataset under data/.
# Usage: bash scripts/robotwin/convert.sh <RoboTwin data/<task>/<config> folder> [--overwrite ...]
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
.venv/bin/python scripts/robotwin/convert_to_lerobot.py "$@"
