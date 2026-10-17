"""LeRobot env plugin: LIBERO task with three butters and a basket.

Importing this package registers --env.type=libero_three_items and
--env.type=libero_levels (difficulty levels with generated shapes, levels.py).
lerobot-eval and lerobot-train import it automatically because the distribution
name starts with "lerobot_env_".
"""

from .config import LiberoLevelsEnv, LiberoThreeItemsEnv
from .task import ITEMS, LANGUAGE, SUITE_NAME

__all__ = ["ITEMS", "LANGUAGE", "SUITE_NAME", "LiberoLevelsEnv", "LiberoThreeItemsEnv"]
