"""Task definition: three identical butters, each with its own ID, go into one basket.

The BDDL file declares the scene (robot arena, basket, item_1..item_3) and the
coarse placement regions. LIBERO's sampler still jitters objects inside those
regions, so the exact layout is pinned by a saved MuJoCo state
(three_butters_basket.pruned_init) that every reset restores.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

SUITE_NAME = "libero_three_items"
TASK_NAME = "three_butters_basket"
LANGUAGE = "pick up the three butters one by one and place them all in the basket"

BASKET = "basket_1"
ITEMS = ("item_1", "item_2", "item_3")

# Fixed world-frame (x, y) of each body in the saved initial state. The robot base
# sits at x ~ -0.66; all four positions are inside the arm's workspace. Items are
# at least 0.15 m apart so the open gripper never touches a neighbour.
LAYOUT_XY = {
    "item_1": (-0.10, -0.18),
    "item_2": (0.06, -0.10),
    "item_3": (-0.15, 0.03),
    BASKET: (0.00, 0.25),
}
# Orientation comes from LIBERO's sampler, which places each butter flat with its
# 4 cm side along world y: the axis the Panda fingers close along when the
# gripper points straight down.

# A scripted demonstration finishes in about 525 control steps (20 Hz); the
# extra margin leaves room for a slower learned policy.
EPISODE_LENGTH = 900


class Task(NamedTuple):
    """Mirror of libero.libero.benchmark.Task so LeRobot's LiberoEnv accepts it."""

    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


# LeRobot joins problem_folder onto LIBERO's bddl/init-state roots. Joining an
# absolute path replaces the root, so the files are read from this package
# instead of the installed LIBERO tree.
TASK = Task(
    name=TASK_NAME,
    language=LANGUAGE,
    problem="Libero",
    problem_folder=str(ASSETS_DIR),
    bddl_file=f"{TASK_NAME}.bddl",
    init_states_file=f"{TASK_NAME}.pruned_init",
)


class ThreeItemsSuite:
    """Minimal one-task suite with the interface LeRobot's LiberoEnv reads."""

    name = SUITE_NAME

    def __init__(self):
        self.tasks = [TASK]
        self.n_tasks = 1

    def get_task(self, i: int) -> Task:
        return self.tasks[i]


def bddl_path() -> Path:
    return ASSETS_DIR / TASK.bddl_file


def init_states_path() -> Path:
    return ASSETS_DIR / TASK.init_states_file
