"""Shared helpers for the triple-basket task scripts (run with the project .venv)."""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import lerobot_env_triple_basket as tb  # noqa: E402  (needs: pip install -e plugins/lerobot_env_triple_basket)
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

DUMMY = [0, 0, 0, 0, 0, 0, -1]


def make_env(size: int = 360) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=str(tb.BDDL_FILE), camera_heights=size, camera_widths=size, control_freq=20
    )
    env.seed(0)
    env.reset()
    return env


def body_pos(env, name: str) -> np.ndarray:
    return np.array(env.env.sim.data.body_xpos[env.env.obj_body_id[name]])


def settle(env, n: int = 10):
    obs = None
    for _ in range(n):
        obs, *_ = env.step(DUMMY)
    return obs


def fixed_reset(env):
    """Reset to the stored fixed init state (same path LeRobot's LiberoEnv uses)."""
    import torch

    env.reset()
    states = torch.load(tb.INIT_FILE, weights_only=False)
    obs = env.set_init_state(states[0])
    return settle(env, 10) or obs
