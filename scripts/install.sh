#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

for command_name in git python3.12; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

lerobot_commit=7e241bd630a3719a56157a497ce5d08f244784f1
if [[ ! -d lerobot ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v0.6.1 https://github.com/huggingface/lerobot.git lerobot
fi
if [[ "$(git -C lerobot rev-parse HEAD)" != "$lerobot_commit" ]]; then
  echo "lerobot/ must be LeRobot v0.6.1 at commit $lerobot_commit" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './lerobot[smolvla,libero]' \
  'torch==2.7.1' 'torchvision==0.22.1' \
  'hf-libero==0.1.4' 'transformers==5.5.4' 'mujoco==3.8.1'
.venv/bin/python -m pip check

if [[ ! -f .libero/config.yaml ]]; then
  mkdir -p .libero
  printf 'n\n' | LIBERO_CONFIG_PATH="$repo_root/.libero" .venv/bin/python -c 'import libero.libero'
fi

HF_HUB_DISABLE_XET=1 .venv/bin/hf download HuggingFaceVLA/smolvla_libero \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --local-dir models/smolvla_libero

HF_HUB_DISABLE_XET=1 .venv/bin/hf download lerobot/libero-assets \
  --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
  --local-dir .venv/lib/python3.12/site-packages/libero/libero/assets

echo "Installation complete. Run: bash scripts/run_orange_juice.sh"
