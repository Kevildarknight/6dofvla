"""Scripted expert for the three-butter task.

It reads object positions from the simulator (privileged state that the policy
never sees) and drives the end effector through fixed waypoints with a
proportional controller. Each item goes through the same cycle:

    above item -> down -> close -> lift -> above basket -> lower -> open -> lift

Before descending, the gripper turns about the vertical axis so its fingers
close across the butter's 4 cm side, whatever the butter's yaw. It turns back
to its starting orientation while carrying, so every butter is dropped the same
way. Horizontal moves happen only at CARRY_Z, above the basket rim; a diagonal path
from the basket to the next item would clip the rim and drag the basket.

Actions use the LIBERO/robosuite OSC_POSE delta convention that the SmolVLA
checkpoint predicts: [dx, dy, dz, drx, dry, drz, gripper] in [-1, 1], where one
unit is 0.05 m / 0.5 rad per control step and gripper -1 opens, +1 closes.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
import robosuite.utils.transform_utils as T

from .task import BASKET, ITEMS

POS_SCALE = 0.05  # metres per action unit (robosuite OSC_POSE output_max)
ROT_SCALE = 0.5  # radians per action unit
OPEN, CLOSE = -1.0, 1.0

GRASP_Z = 0.012  # end-effector height when the fingers close around a flat butter
CARRY_Z = 0.22  # clears the basket rim (0.141 m) with the butter hanging below
RELEASE_Z = 0.17  # opening the fingers below the rim pushes the basket wall
GRIP_STEPS = 15
ROT_TOLERANCE = 0.05  # rad; the approach waits for the wrist to finish turning
# Segments that use the item-aligned gripper yaw; the others use the start orientation.
ITEM_ALIGNED = {"approach", "descend", "grasp", "lift"}
# Small offsets inside the basket so later butters land beside earlier ones,
# indexed by pick order (first, second, third).
DROP_OFFSETS = ((0.0, -0.02), (0.0, 0.02), (0.0, 0.0))


@dataclass
class Segment:
    name: str
    item: str
    slot: int  # position in the pick order; selects the drop offset
    target: str  # "item" | "basket" | "hold"
    z: float
    gripper: float
    max_speed: float = 1.0
    tolerance: float = 0.008
    fixed_steps: int = 0  # >0: hold position for this many steps instead of moving
    max_steps: int = 120


def _plan(order: tuple[str, ...]) -> list[Segment]:
    plan = []
    for slot, item in enumerate(order):
        plan += [
            Segment("approach", item, slot, "item", CARRY_Z, OPEN),
            Segment("descend", item, slot, "item", GRASP_Z, OPEN, max_speed=0.4, tolerance=0.005),
            Segment("grasp", item, slot, "hold", GRASP_Z, CLOSE, fixed_steps=GRIP_STEPS),
            Segment("lift", item, slot, "hold", CARRY_Z, CLOSE, max_speed=0.6),
            Segment("carry", item, slot, "basket", CARRY_Z, CLOSE),
            Segment("lower", item, slot, "basket", RELEASE_Z, CLOSE, max_speed=0.4),
            Segment("release", item, slot, "hold", RELEASE_Z, OPEN, fixed_steps=GRIP_STEPS),
            Segment("retreat", item, slot, "hold", CARRY_Z, OPEN, max_speed=0.6),
        ]
    return plan


class ScriptedThreeItemsExpert:
    """Call reset(env) after env.reset(), then act(env) once per control step."""

    def __init__(self, rng: np.random.Generator | None = None, jitter: float = 0.0, shuffle_order: bool = False):
        self.rng = rng or np.random.default_rng(0)
        self.jitter = jitter
        self.shuffle_order = shuffle_order
        self.order = ITEMS
        self.plan = _plan(self.order)

    def reset(self, env) -> None:
        # The butters look identical, so any pick order is valid; shuffling it
        # keeps a policy from tying "first" to one particular ID or position.
        self.order = tuple(self.rng.permutation(ITEMS)) if self.shuffle_order else ITEMS
        self.plan = _plan(self.order)
        self.index = 0
        self.steps_in_segment = 0
        self.base_quat = self._eef_quat(env)
        self._item_quat: dict[str, np.ndarray] = {}
        self._hold_xy: np.ndarray | None = None
        # Per-episode waypoint noise gives the demonstrations some variety.
        self._noise = {
            (seg.item, seg.target): self.rng.uniform(-self.jitter, self.jitter, size=2) for seg in self.plan
        }

    @property
    def done(self) -> bool:
        return self.index >= len(self.plan)

    @property
    def segment(self) -> Segment | None:
        return None if self.done else self.plan[self.index]

    def act(self, env) -> np.ndarray:
        if self.done:
            return np.array([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32)
        seg = self.plan[self.index]
        eef = self._eef_pos(env)
        target = self._target(env, seg, eef)
        error = target - eef

        delta = np.clip(error / POS_SCALE, -seg.max_speed, seg.max_speed)
        rot_error = self._rotation_error(env, self._target_quat(env, seg))
        rot = np.clip(rot_error / ROT_SCALE, -0.5, 0.5)
        action = np.concatenate([delta, rot, [seg.gripper]]).astype(np.float32)

        self.steps_in_segment += 1
        arrived = np.linalg.norm(error) < seg.tolerance
        if seg.name == "approach":
            arrived = arrived and np.linalg.norm(rot_error) < ROT_TOLERANCE
        finished = (
            self.steps_in_segment >= seg.fixed_steps
            if seg.fixed_steps
            else arrived or self.steps_in_segment >= seg.max_steps
        )
        if finished:
            self.index += 1
            self.steps_in_segment = 0
            self._hold_xy = None
        return action

    def _target(self, env, seg: Segment, eef: np.ndarray) -> np.ndarray:
        if seg.target == "item":
            xy = self._body_pos(env, seg.item)[:2]
        elif seg.target == "basket":
            xy = self._body_pos(env, BASKET)[:2] + np.array(DROP_OFFSETS[seg.slot])
        else:  # hold the xy where this segment started, move only in z
            if self._hold_xy is None:
                self._hold_xy = eef[:2].copy()
            xy = self._hold_xy
        if seg.target != "hold":
            xy = xy + self._noise[(seg.item, seg.target)]
        return np.array([xy[0], xy[1], seg.z])

    @staticmethod
    def _body_pos(env, name: str) -> np.ndarray:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])

    @staticmethod
    def _eef_pos(env) -> np.ndarray:
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])

    @staticmethod
    def _eef_quat(env) -> np.ndarray:
        return T.mat2quat(env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3))

    def _target_quat(self, env, seg: Segment) -> np.ndarray:
        if seg.name not in ITEM_ALIGNED:
            return self.base_quat
        if seg.item not in self._item_quat:
            # Fixed once per item, at the start of its approach, while it lies still.
            yaw = self.item_yaw(env, seg.item)
            turn = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])  # xyzw
            self._item_quat[seg.item] = T.quat_multiply(turn, self.base_quat)
        return self._item_quat[seg.item]

    @staticmethod
    def item_yaw(env, name: str) -> float:
        """Yaw of the butter's long side relative to world x, wrapped to [-90, 90] deg.

        The start orientation closes the fingers along world y, which grips a
        butter whose long side lies along world x; this is the turn needed for
        any other heading.
        """
        model, data = env.sim.model, env.sim.data
        body = env.obj_body_id[name]
        box = max(
            (g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_type[g] == 6),
            key=lambda g: model.geom_size[g].max(),
        )
        long_axis = data.geom_xmat[box].reshape(3, 3)[:, int(np.argmax(model.geom_size[box]))]
        yaw = math.atan2(long_axis[1], long_axis[0])
        return (yaw + math.pi / 2) % math.pi - math.pi / 2

    def _rotation_error(self, env, target_quat: np.ndarray) -> np.ndarray:
        """Axis-angle rotation (world frame) from the current to the target gripper orientation."""
        q = T.quat_multiply(target_quat, T.quat_inverse(self._eef_quat(env)))
        if q[3] < 0:
            q = -q
        return T.quat2axisangle(q)
