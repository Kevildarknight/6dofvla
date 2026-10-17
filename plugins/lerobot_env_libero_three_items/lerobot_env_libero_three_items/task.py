"""Task definition: three identical butters, each with its own ID, go into one basket.

The BDDL file declares the scene (robot arena, basket, item_1..item_3) and the
coarse placement regions. LIBERO's sampler still jitters objects inside those
regions, so exact layouts are pinned by saved MuJoCo states that resets restore.

Task IDs 0-2 share the three-butter scene and instruction and differ only in
their initial states:

    0  fixed    one hand-picked layout (the original demo)
    1  train    randomly sampled layouts for demonstrations
    2  heldout  separately sampled layouts, used only for evaluation

Task 3 is a reduced version for debugging the pipeline: one butter, one fixed
layout, and LIBERO Object's own sentence for picking up the butter.

    3  single   item_1 and the basket only

Within a task, successive resets cycle through its saved states.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

SUITE_NAME = "libero_three_items"
TASK_NAME = "three_butters_basket"
LANGUAGE = "pick up the three butters one by one and place them all in the basket"
# Other phrasings used for some demonstrations. Evaluation always uses LANGUAGE.
LANGUAGE_VARIANTS = (
    "put all three butters in the basket",
    "pick up each butter and put it in the basket",
    "place the three butters into the basket one at a time",
    "move all the butters into the basket",
)

BASKET = "basket_1"
ITEMS = ("item_1", "item_2", "item_3")

# World-frame (x, y) of each body in the fixed layout (task 0). The robot base
# sits at x ~ -0.66; all four positions are inside the arm's workspace. Items are
# at least 0.15 m apart so the open gripper never touches a neighbour.
LAYOUT_XY = {
    "item_1": (-0.10, -0.18),
    "item_2": (0.06, -0.10),
    "item_3": (-0.15, 0.03),
    BASKET: (0.00, 0.25),
}
SINGLE_TASK_NAME = "one_butter_basket"
SINGLE_LANGUAGE = "pick up the butter and place it in the basket"
SINGLE_LAYOUT_XY = {"item_1": LAYOUT_XY["item_1"], BASKET: LAYOUT_XY[BASKET]}
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


TASK_FIXED, TASK_TRAIN, TASK_HELDOUT, TASK_SINGLE = 0, 1, 2, 3
SPLITS = {"fixed": TASK_FIXED, "train": TASK_TRAIN, "heldout": TASK_HELDOUT, "single": TASK_SINGLE}
# Number of layouts and sampling seed per random split. Different seeds keep
# the held-out layouts independent of the training ones.
RANDOM_SPLITS = {"train": (40, 0), "heldout": (10, 1)}


def _task(split: str) -> Task:
    if split == "single":
        stem, scene, language = SINGLE_TASK_NAME, SINGLE_TASK_NAME, SINGLE_LANGUAGE
    else:
        stem = TASK_NAME if split == "fixed" else f"{TASK_NAME}_{split}"
        scene, language = TASK_NAME, LANGUAGE
    # LeRobot joins problem_folder onto LIBERO's bddl/init-state roots. Joining an
    # absolute path replaces the root, so the files are read from this package
    # instead of the installed LIBERO tree.
    return Task(
        name=stem,
        language=language,
        problem="Libero",
        problem_folder=str(ASSETS_DIR),
        bddl_file=f"{scene}.bddl",
        init_states_file=f"{stem}.pruned_init",
    )


TASKS = [_task(split) for split in SPLITS]


class ThreeItemsSuite:
    """Minimal suite with the interface LeRobot's LiberoEnv reads."""

    name = SUITE_NAME

    def __init__(self):
        self.tasks = list(TASKS)
        self.n_tasks = len(self.tasks)

    def get_task(self, i: int) -> Task:
        return self.tasks[i]


def bddl_path(split: str = "fixed") -> Path:
    return ASSETS_DIR / TASKS[SPLITS[split]].bddl_file


def scene_items(sim_env) -> tuple[str, ...]:
    """The item_* objects present in a built scene (three butters, or one in task 3)."""
    return tuple(sorted(name for name in sim_env.objects_dict if name.startswith("item_")))


def init_states_path(split: str) -> Path:
    return ASSETS_DIR / TASKS[SPLITS[split]].init_states_file


def layouts_path(split: str) -> Path:
    """JSON record of the layouts behind each saved state, in the same order."""
    return ASSETS_DIR / f"{Path(TASKS[SPLITS[split]].init_states_file).stem}.layouts.json"
