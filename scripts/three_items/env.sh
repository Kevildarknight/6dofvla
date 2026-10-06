# Shared environment for the three-butter scripts; source it, do not run it.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
if ! .venv/bin/python -c 'import lerobot_env_libero_three_items' 2>/dev/null; then
  echo "Plugin not installed. Run: bash scripts/three_items/install.sh" >&2
  exit 1
fi
export LIBERO_CONFIG_PATH="$repo_root/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
