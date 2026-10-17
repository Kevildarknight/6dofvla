"""Difficulty levels: which objects to pick and which container to fill.

Each level is paired with each container, giving one task per pair under
--env.type=libero_levels (task ids in the order of TASKS):

    level           objects                                       what changes
    one_object      red cube                                      single pick
    three_same      3 red cubes                                   several identical picks
    three_shapes    red cube, red cylinder, red prism             shape (square, round, triangle)
    three_sizes     red cylinders: small, medium, large           size, same shape and colour
    colors_sizes    small red, medium green, large blue cylinder  colour and size

    container       tray_wide (wide and shallow), bin_narrow (narrow and tall)

The goal is always to put every object in the container. Objects start at the
three-butter task's fixed positions and the container at the basket's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import shapes  # noqa: F401  (registers the object categories)
from .layouts import Layout
from .task import BASKET, LAYOUT_XY, Task

LEVELS_DIR = Path(__file__).resolve().parent / "assets" / "levels"
SUITE_NAME = "libero_levels"
CONTAINER_WORDS = {"tray_wide": "tray", "bin_narrow": "bin"}


@dataclass(frozen=True)
class Level:
    name: str
    objects: dict[str, str]  # item id -> object category
    instruction: str  # "{container}" is replaced by tray / bin


LEVELS = (
    Level("one_object", {"item_1": "cube_red"}, "put the red cube in the {container}"),
    Level(
        "three_same",
        {"item_1": "cube_red", "item_2": "cube_red", "item_3": "cube_red"},
        "put the three red cubes in the {container}",
    ),
    Level(
        "three_shapes",
        {"item_1": "cube_red", "item_2": "cylinder_medium_red", "item_3": "prism_red"},
        "put the cube, the cylinder and the triangle in the {container}",
    ),
    Level(
        "three_sizes",
        {"item_1": "cylinder_small_red", "item_2": "cylinder_medium_red", "item_3": "cylinder_large_red"},
        "put the small, the medium and the large cylinder in the {container}",
    ),
    Level(
        "colors_sizes",
        {"item_1": "cylinder_small_red", "item_2": "cylinder_medium_green", "item_3": "cylinder_large_blue"},
        "put the red, the green and the blue cylinder in the {container}",
    ),
)


@dataclass(frozen=True)
class LevelTask:
    level: Level
    container: str

    @property
    def name(self) -> str:
        return f"{self.level.name}_{self.container}"

    @property
    def language(self) -> str:
        return self.level.instruction.format(container=CONTAINER_WORDS[self.container])

    @property
    def layout(self) -> Layout:
        xy = {item: LAYOUT_XY[item] for item in self.level.objects}
        xy[BASKET] = LAYOUT_XY[BASKET]
        return Layout(xy=xy, item_yaw={item: 0.0 for item in self.level.objects})

    def as_libero_task(self) -> Task:
        return Task(
            name=self.name,
            language=self.language,
            problem="Libero",
            problem_folder=str(LEVELS_DIR),
            bddl_file=f"{self.name}.bddl",
            init_states_file=f"{self.name}.pruned_init",
        )

    def bddl(self) -> str:
        regions = []
        for body, (x, y) in self.layout.xy.items():
            regions.append(
                f"      ({body}_region\n          (:target floor)\n"
                f"          (:ranges (\n              ({x - 0.005:.3f} {y - 0.005:.3f} {x + 0.005:.3f} {y + 0.005:.3f})\n"
                "            )\n          )\n      )"
            )
        regions.append(f"      (contain_region\n          (:target {BASKET})\n      )")
        # LIBERO's parser keeps only the last line per category, so items that
        # share a category must be declared on one line.
        by_category: dict[str, list[str]] = {}
        for item, category in self.level.objects.items():
            by_category.setdefault(category, []).append(item)
        objects = [f"    {' '.join(items)} - {category}" for category, items in by_category.items()]
        objects.append(f"    {BASKET} - {self.container}")
        bodies = [*self.level.objects, BASKET]
        init = [f"    (On {body} floor_{body}_region)" for body in bodies]
        goal = " ".join(f"(In {item} {BASKET}_contain_region)" for item in self.level.objects)
        return (
            "(define (problem LIBERO_Floor_Manipulation)\n  (:domain robosuite)\n"
            f"  (:language {self.language})\n    (:regions\n" + "\n".join(regions) + "\n    )\n\n"
            "  (:fixtures\n    floor - floor\n  )\n\n"
            "  (:objects\n" + "\n".join(objects) + "\n  )\n\n"
            "  (:obj_of_interest\n" + "\n".join(f"    {b}" for b in bodies) + "\n  )\n\n"
            "  (:init\n" + "\n".join(init) + "\n  )\n\n"
            f"  (:goal\n    (And {goal})\n  )\n\n)\n"
        )


TASKS = tuple(LevelTask(level, container) for level in LEVELS for container in CONTAINER_WORDS)


def write_bddl_files() -> None:
    LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = LEVELS_DIR / f"{task.name}.bddl"
        text = task.bddl()
        if not path.exists() or path.read_text() != text:
            path.write_text(text)


class LevelsSuite:
    """Suite with the interface LeRobot's LiberoEnv reads."""

    name = SUITE_NAME

    def __init__(self):
        write_bddl_files()
        self.tasks = [task.as_libero_task() for task in TASKS]
        self.n_tasks = len(self.tasks)

    def get_task(self, i: int) -> Task:
        return self.tasks[i]
