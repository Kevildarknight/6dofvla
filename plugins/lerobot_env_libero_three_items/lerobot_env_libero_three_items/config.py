"""Registers --env.type=libero_three_items with LeRobot's EnvConfig registry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import gymnasium as gym
from lerobot.envs.configs import EnvConfig, LiberoEnv

from .task import EPISODE_LENGTH, SPLITS, SUITE_NAME, TASK_FIXED


@EnvConfig.register_subclass(SUITE_NAME)
@dataclass
class LiberoThreeItemsEnv(LiberoEnv):
    """Same observations, actions, and processors as LIBERO; one custom task.

    --env.task_ids selects the layouts: 0 fixed (default), 1 train, 2 heldout.
    Successive episodes of a task cycle through its saved layouts.
    """

    task: str = SUITE_NAME
    task_ids: list[int] | None = None
    episode_length: int | None = EPISODE_LENGTH
    # Demonstrations are recorded at 256x256, the resolution of the
    # HuggingFaceVLA/libero data the checkpoint was trained on.
    observation_height: int = 256
    observation_width: int = 256

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .env import ThreeItemsLiberoEnv

        task_ids = [TASK_FIXED] if self.task_ids is None else sorted(set(self.task_ids))
        for task_id in task_ids:
            if task_id not in SPLITS.values():
                raise ValueError(f"{SUITE_NAME} task ids are {SPLITS}, got {task_id}")
        gym_kwargs = {k: v for k, v in self.gym_kwargs.items() if k != "task_ids"}
        envs = {}
        for task_id in task_ids:
            fns = [
                partial(
                    ThreeItemsLiberoEnv,
                    task_id=task_id,
                    camera_name=self.camera_name,
                    init_states=self.init_states,
                    episode_length=self.episode_length,
                    episode_index=episode_index,
                    n_envs=n_envs,
                    control_mode=self.control_mode,
                    camera_name_mapping=self.camera_name_mapping,
                    **gym_kwargs,
                )
                for episode_index in range(n_envs)
            ]
            # Always synchronous: one MuJoCo/EGL context per process is what this
            # exercise was tested with, and batch size 1 is the evaluation setting.
            envs[task_id] = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        return {SUITE_NAME: envs}
