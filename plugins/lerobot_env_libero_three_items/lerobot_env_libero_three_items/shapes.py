"""Primitive objects and containers for the difficulty levels.

LIBERO builds objects from MuJoCo XML files registered under a category name.
This module writes those XML files (and an OBJ mesh for the triangular prism)
into assets/shapes/ and registers one LIBERO object class per category, so a
BDDL file can declare e.g. `item_1 - cylinder_large_red` or `basket_1 - tray_wide`.

Objects stand upright on their flat face. All of them fit in the open Panda
gripper (about 8 cm) when grasped from above:

    cube       4 cm cube
    cylinder   4 cm tall; diameter 3 cm (small), 4 cm (medium), 5.6 cm (large)
    prism      4 cm tall equilateral triangular prism, 4.6 cm sides

Containers are open boxes with a `contain_region` site filling the inside, the
site the success check uses:

    tray_wide   inner 20 x 14 cm, 4 cm walls   (wide and shallow)
    bin_narrow  inner 10 x 10 cm, 12 cm walls  (narrow and tall)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from libero.libero.envs.base_object import OBJECTS_DICT, register_object
from robosuite.models.objects import MujocoXMLObject

SHAPES_DIR = Path(__file__).resolve().parent / "assets" / "shapes"

COLORS = {
    "red": "0.85 0.18 0.15 1",
    "green": "0.20 0.65 0.30 1",
    "blue": "0.18 0.38 0.85 1",
    "yellow": "0.95 0.80 0.15 1",
}
CYLINDER_RADIUS = {"small": 0.015, "medium": 0.02, "large": 0.028}
OBJECT_HALF_HEIGHT = 0.02
CUBE_HALF = 0.02
PRISM_SIDE = 0.046
WALL = 0.008  # container wall and floor thickness


@dataclass(frozen=True)
class ContainerSpec:
    inner_x: float
    inner_y: float
    wall_height: float
    rgba: str


CONTAINERS = {
    "tray_wide": ContainerSpec(0.20, 0.14, 0.04, "0.55 0.42 0.30 1"),
    "bin_narrow": ContainerSpec(0.10, 0.10, 0.12, "0.45 0.47 0.52 1"),
}


def object_categories() -> dict[str, tuple[str, str, str]]:
    """category -> (shape, size, color) for every primitive object."""
    cats = {}
    for color in COLORS:
        cats[f"cube_{color}"] = ("cube", "medium", color)
        cats[f"prism_{color}"] = ("prism", "medium", color)
        for size in CYLINDER_RADIUS:
            cats[f"cylinder_{size}_{color}"] = ("cylinder", size, color)
    return cats


# A parallel gripper closing along world y grips these shapes the same way
# after turning by this period (radians), so the controller turns the wrist by
# at most half of it. None means any heading works.
GRASP_PERIOD = {"cube": math.pi / 2, "cylinder": None, "prism": 2 * math.pi / 3}


def _sites() -> str:
    h = OBJECT_HALF_HEIGHT
    return (
        f'<site rgba="0 0 0 0" size="0.005" pos="0 0 {-h:.4f}" name="bottom_site"/>'
        f'<site rgba="0 0 0 0" size="0.005" pos="0 0 {h:.4f}" name="top_site"/>'
        '<site rgba="0 0 0 0" size="0.005" pos="0.02 0.02 0" name="horizontal_radius_site"/>'
    )


def _prism_obj() -> str:
    """Triangular prism with one edge along x and its y extent centred on the origin.

    Centring the y extent (not the centroid) puts the grip midway between the
    edge and the opposite vertex, where the symmetric Panda fingers close.
    """
    a, h = PRISM_SIDE, OBJECT_HALF_HEIGHT
    tri_h = a * math.sqrt(3) / 2
    tri = [(-a / 2, -tri_h / 2), (a / 2, -tri_h / 2), (0.0, tri_h / 2)]
    verts = [(x, y, -h) for x, y in tri] + [(x, y, h) for x, y in tri]
    faces = [(1, 3, 2), (4, 5, 6), (1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5), (3, 1, 4), (3, 4, 6)]
    lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts] + [f"f {a} {b} {c}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def _object_xml(category: str, shape: str, size: str, color: str) -> str:
    rgba = COLORS[color]
    common = f'rgba="{rgba}" density="400" friction="1.2 0.01 0.001" solref="0.004 1" condim="4" group="0"'
    asset = ""
    if shape == "cube":
        geom = f'<geom type="box" size="{CUBE_HALF} {CUBE_HALF} {OBJECT_HALF_HEIGHT}" {common}/>'
    elif shape == "cylinder":
        geom = f'<geom type="cylinder" size="{CYLINDER_RADIUS[size]} {OBJECT_HALF_HEIGHT}" {common}/>'
    else:
        asset = '<asset><mesh name="prism_mesh" file="prism.obj"/></asset>'
        geom = f'<geom type="mesh" mesh="prism_mesh" {common}/>'
    return (
        f'<mujoco model="{category}">{asset}<worldbody><body><body name="object">{geom}</body>'
        f"{_sites()}</body></worldbody></mujoco>\n"
    )


def _container_xml(category: str, spec: ContainerSpec) -> str:
    hx, hy, t, H = spec.inner_x / 2, spec.inner_y / 2, WALL, spec.wall_height
    g = f'rgba="{spec.rgba}" density="300" friction="1 0.01 0.001" group="0"'
    wall_z = H / 2
    geoms = [
        f'<geom type="box" pos="0 0 {t / 2:.4f}" size="{hx + t:.4f} {hy + t:.4f} {t / 2:.4f}" {g}/>',
        f'<geom type="box" pos="{hx + t / 2:.4f} 0 {wall_z:.4f}" size="{t / 2:.4f} {hy + t:.4f} {H / 2:.4f}" {g}/>',
        f'<geom type="box" pos="{-hx - t / 2:.4f} 0 {wall_z:.4f}" size="{t / 2:.4f} {hy + t:.4f} {H / 2:.4f}" {g}/>',
        f'<geom type="box" pos="0 {hy + t / 2:.4f} {wall_z:.4f}" size="{hx:.4f} {t / 2:.4f} {H / 2:.4f}" {g}/>',
        f'<geom type="box" pos="0 {-hy - t / 2:.4f} {wall_z:.4f}" size="{hx:.4f} {t / 2:.4f} {H / 2:.4f}" {g}/>',
    ]
    # Inside the walls, from the floor's top face to the rim.
    site_half_z = (H - t) / 2
    site = (
        f'<site type="box" pos="0 0 {t + site_half_z:.4f}" size="{hx:.4f} {hy:.4f} {site_half_z:.4f}" '
        'quat="1 0 0 0" rgba="0.8 0 0 0" group="0" name="contain_region"/>'
    )
    sites = (
        '<site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>'
        f'<site rgba="0 0 0 0" size="0.005" pos="0 0 {H:.4f}" name="top_site"/>'
        f'<site rgba="0 0 0 0" size="0.005" pos="{hx:.4f} {hy:.4f} 0" name="horizontal_radius_site"/>'
    )
    return (
        f'<mujoco model="{category}"><worldbody><body><body name="object">{"".join(geoms)}{site}</body>'
        f"{sites}</body></worldbody></mujoco>\n"
    )


def write_assets() -> None:
    """(Re)write every XML/OBJ file. Deterministic, so it is safe to call at import."""
    SHAPES_DIR.mkdir(parents=True, exist_ok=True)
    files = {"prism.obj": _prism_obj()}
    for category, (shape, size, color) in object_categories().items():
        files[f"{category}.xml"] = _object_xml(category, shape, size, color)
    for category, spec in CONTAINERS.items():
        files[f"{category}.xml"] = _container_xml(category, spec)
    for name, text in files.items():
        path = SHAPES_DIR / name
        if not path.exists() or path.read_text() != text:
            path.write_text(text)


class _ShapeObject(MujocoXMLObject):
    """LIBERO-style object built from one of the generated XML files."""

    category: str = ""
    shape: str = ""

    def __init__(self, name: str):
        super().__init__(
            str(SHAPES_DIR / f"{self.category}.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            # The generated geoms are collision geoms (group 0), which cameras do
            # not draw; robosuite adds a drawn copy (group 1) of each one.
            duplicate_collision_geoms=True,
        )
        self.category_name = self.category
        # Keep the generated orientation; layouts set the yaw explicitly.
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.grasp_period = GRASP_PERIOD.get(self.shape)


def _register(category: str, shape: str) -> None:
    if category in OBJECTS_DICT:
        return
    class_name = "".join(part.capitalize() for part in category.split("_"))
    cls = type(class_name, (_ShapeObject,), {"category": category, "shape": shape})
    register_object(cls)
    # register_object derives the key from the class name; make sure it matches.
    assert category in OBJECTS_DICT, f"{class_name} registered under an unexpected key"


def register_all() -> None:
    write_assets()
    for category, (shape, _, _) in object_categories().items():
        _register(category, shape)
    for category in CONTAINERS:
        _register(category, "container")


register_all()
