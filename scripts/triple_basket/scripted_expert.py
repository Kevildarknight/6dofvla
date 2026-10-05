"""Scripted pick-and-place expert (privileged state) used to sanity-check the scene and record demos.

Gains/heights are a starting point and must be tuned against the simulator.
"""
import numpy as np
from common import body_pos, tb

MAX_DELTA = 0.05  # OSC_POSE: action 1.0 == 5 cm per step
OPEN, CLOSE = -1.0, 1.0


def _move(env, target, grip, tol=0.012, max_steps=120, on_step=None):
    for _ in range(max_steps):
        eef = np.array(env.env.robots[0].controller.ee_pos)
        err = target - eef
        if np.linalg.norm(err) < tol:
            return True
        a = np.zeros(7)
        a[:3] = np.clip(err / MAX_DELTA * 2.0, -1, 1)
        a[6] = grip
        obs, _, _, _ = env.step(a)
        if on_step:
            on_step(obs, a)
    return False


def _hold(env, grip, n, on_step=None):
    for _ in range(n):
        a = np.array([0, 0, 0, 0, 0, 0, grip], dtype=float)
        obs, _, _, _ = env.step(a)
        if on_step:
            on_step(obs, a)


def run_episode(env, on_step=None, max_items=3):
    """Pick item_1..3 in order and drop each into the basket. Returns env.check_success()."""
    for k, item in enumerate(tb.ITEMS[:max_items]):
        obj = body_pos(env, item)
        basket = body_pos(env, tb.BASKET)
        hover = np.array([obj[0], obj[1], obj[2] + 0.15])
        _move(env, hover, OPEN, on_step=on_step)
        _move(env, obj + np.array([0, 0, 0.0]), OPEN, tol=0.008, on_step=on_step)
        _hold(env, CLOSE, 15, on_step)
        _move(env, np.array([obj[0], obj[1], obj[2] + 0.25]), CLOSE, on_step=on_step)
        drop = basket + np.array([(k - 1) * 0.04, 0.0, 0.25])
        _move(env, drop, CLOSE, on_step=on_step)
        _hold(env, OPEN, 15, on_step)
        _hold(env, OPEN, 10, on_step)
    _hold(env, OPEN, tb.SUCCESS_DWELL_STEPS + 5, on_step)
    return env.check_success()
