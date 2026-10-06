"""Gym env for the three-butter task, built on LeRobot's LiberoEnv wrapper.

Only reset() and step() change: LIBERO's own BDDL goal ("In" = touching the
basket and inside its box) is replaced by the stricter BasketSuccessChecker, and
the result is published as info["is_success"], which lerobot-eval writes to
eval_info.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lerobot.envs.libero import LiberoEnv

from .success import BasketSuccessChecker, SuccessStatus
from .task import ITEMS, SUITE_NAME, ThreeItemsSuite


class ThreeItemsLiberoEnv(LiberoEnv):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("task_suite", ThreeItemsSuite())
        kwargs.setdefault("task_id", 0)
        kwargs.setdefault("task_suite_name", SUITE_NAME)
        super().__init__(**kwargs)
        self._checker = BasketSuccessChecker(control_dt=1.0 / self.control_freq)
        self.last_status: SuccessStatus | None = None

    @property
    def sim_env(self):
        """The underlying robosuite/LIBERO environment (sim, robots, objects)."""
        self._ensure_env()
        return self._env.env

    def reset(self, seed=None, **kwargs):
        observation, info = super().reset(seed=seed, **kwargs)
        self._checker.reset(self.sim_env)
        self.last_status = None
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, _, truncated, info = super().step(action)
        status = self._checker.update(self.sim_env)
        self.last_status = status
        info["bddl_goal"] = info["is_success"]
        info["is_success"] = status.success
        info["num_items_in_basket"] = status.num_in_basket
        for name in ITEMS:
            info[f"{name}_in_basket"] = status.items[name].in_basket
        # LIBERO's "done" is its BDDL goal, which fires while the last item is still
        # falling, so only the strict check ends the episode. Time limits are
        # applied by the caller through _max_episode_steps.
        return observation, float(status.success), status.success, truncated, info
