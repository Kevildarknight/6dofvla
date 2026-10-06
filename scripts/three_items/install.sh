#!/usr/bin/env bash
# Install the three-butter task plugin into the project virtual environment.
# Run scripts/install.sh first.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python || ! -f .libero/config.yaml ]]; then
  echo "Run bash scripts/install.sh first." >&2
  exit 1
fi
.venv/bin/python -m pip install --no-deps -e plugins/lerobot_env_libero_three_items
.venv/bin/python -c 'import lerobot_env_libero_three_items' 2>/dev/null
echo "Installed. --env.type=libero_three_items is now available to lerobot-eval and lerobot-train."
