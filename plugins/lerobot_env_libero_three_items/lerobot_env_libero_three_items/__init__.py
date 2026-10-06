"""LeRobot env plugin: LIBERO task with three butters and a basket.

Importing this package registers --env.type=libero_three_items. lerobot-eval and
lerobot-train import it automatically because the distribution name starts with
"lerobot_env_".
"""

from .config import LiberoThreeItemsEnv
from .task import ITEMS, LANGUAGE, SUITE_NAME

__all__ = ["ITEMS", "LANGUAGE", "SUITE_NAME", "LiberoThreeItemsEnv"]
