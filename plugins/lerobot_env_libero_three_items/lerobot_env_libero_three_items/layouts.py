"""Object layouts: the fixed demo layout and randomly sampled ones.

A layout gives the world (x, y) of item_1..item_3 and the basket, plus a yaw
for each item. Sampled layouts keep every item reachable from above, far enough
from its neighbours for the open gripper, and clear of the basket.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .task import BASKET, ITEMS, LAYOUT_XY

# Sampling ranges in world metres. The robot base is at x ~ -0.66, so these
# stay inside the region the scripted controller reaches with a vertical gripper.
ITEM_X = (-0.20, 0.10)
ITEM_Y = (-0.22, 0.08)
BASKET_X = (-0.08, 0.08)
BASKET_Y = (0.18, 0.28)
MAX_ITEM_YAW = math.pi / 4  # a butter is symmetric under 180 deg, so +-45 deg covers half the cases

# Butter is 7.6 cm long: 13 cm between centres leaves room for the fingers even
# when two butters point at each other. The basket's outer half-width is ~7.5 cm.
MIN_ITEM_GAP = 0.13
MIN_ITEM_BASKET_GAP = 0.18


@dataclass
class Layout:
    xy: dict[str, tuple[float, float]]
    item_yaw: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Layout:
        return cls(
            xy={k: tuple(v) for k, v in data["xy"].items()},
            item_yaw=dict(data["item_yaw"]),
        )


FIXED_LAYOUT = Layout(xy=dict(LAYOUT_XY), item_yaw={name: 0.0 for name in ITEMS})


def is_valid(layout: Layout) -> bool:
    items = [np.array(layout.xy[name]) for name in ITEMS]
    basket = np.array(layout.xy[BASKET])
    gaps_ok = all(
        np.linalg.norm(items[i] - items[j]) >= MIN_ITEM_GAP
        for i in range(len(items))
        for j in range(i + 1, len(items))
    )
    return gaps_ok and all(np.linalg.norm(p - basket) >= MIN_ITEM_BASKET_GAP for p in items)


def sample_layout(rng: np.random.Generator, max_tries: int = 10_000) -> Layout:
    for _ in range(max_tries):
        layout = Layout(
            xy={
                BASKET: (float(rng.uniform(*BASKET_X)), float(rng.uniform(*BASKET_Y))),
                **{name: (float(rng.uniform(*ITEM_X)), float(rng.uniform(*ITEM_Y))) for name in ITEMS},
            },
            item_yaw={name: float(rng.uniform(-MAX_ITEM_YAW, MAX_ITEM_YAW)) for name in ITEMS},
        )
        if is_valid(layout):
            return layout
    raise RuntimeError("Could not sample a valid layout; the sampling ranges are too tight.")


def apply_layout(sim_env, layout: Layout) -> None:
    """Teleport the basket and items to a layout, keeping their sampled height.

    Item yaw is applied on top of the orientation LIBERO's sampler gives a
    butter (lying flat, long side along world x).
    """
    sim = sim_env.sim
    for name, (x, y) in layout.xy.items():
        joint = sim_env.get_object(name).joints[0]
        qpos = sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = (x, y)
        yaw = layout.item_yaw.get(name, 0.0)
        if yaw:
            # MuJoCo quaternions are (w, x, y, z); rotate about world z.
            rot = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
            qpos[3:7] = _quat_mul_wxyz(rot, qpos[3:7])
        sim.data.set_joint_qpos(joint, qpos)
        sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )
