"""Registers --env.type=libero_three_items and --env.type=libero_levels with LeRobot."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import gymnasium as gym
from lerobot.envs.configs import EnvConfig, LiberoEnv

from .task import EPISODE_LENGTH, SPLITS, SUITE_NAME

LEVELS_SUITE = "libero_levels"  # levels.SUITE_NAME; kept here so importing config stays light


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

    def make_suite(self):
        from .task import ThreeItemsSuite

        return ThreeItemsSuite()

    def valid_task_ids(self) -> dict:
        return SPLITS

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .env import ThreeItemsLiberoEnv

        valid = self.valid_task_ids()
        task_ids = [min(valid.values())] if self.task_ids is None else sorted(set(self.task_ids))
        for task_id in task_ids:
            if task_id not in valid.values():
                raise ValueError(f"{self.task} task ids are {valid}, got {task_id}")
        suite = self.make_suite()
        gym_kwargs = {k: v for k, v in self.gym_kwargs.items() if k != "task_ids"}
        envs = {}
        for task_id in task_ids:
            fns = [
                partial(
                    ThreeItemsLiberoEnv,
                    task_suite=suite,
                    task_suite_name=self.task,
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
        return {self.task: envs}


@EnvConfig.register_subclass(LEVELS_SUITE)
@dataclass
class LiberoLevelsEnv(LiberoThreeItemsEnv):
    """Difficulty levels (levels.py): --env.task_ids picks a level x container pair.

    Task ids follow levels.TASKS: 0 one_object_tray_wide, 1 one_object_bin_narrow,
    2 three_same_tray_wide, ... 9 colors_sizes_bin_narrow.
    """

    task: str = LEVELS_SUITE

    def make_suite(self):
        from .levels import LevelsSuite

        return LevelsSuite()

    def valid_task_ids(self) -> dict:
        from .levels import TASKS

        return {task.name: i for i, task in enumerate(TASKS)}
