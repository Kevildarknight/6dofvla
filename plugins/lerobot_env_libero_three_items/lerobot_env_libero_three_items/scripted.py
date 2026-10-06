"""Scripted expert for the three-butter task.

It reads object positions from the simulator (privileged state that the policy
never sees) and drives the end effector through fixed waypoints with a
proportional controller. Each item goes through the same cycle:

    above item -> down -> close -> lift -> above basket -> lower -> open -> lift

Horizontal moves happen only at CARRY_Z, above the basket rim; a diagonal path
from the basket to the next item would clip the rim and drag the basket.

Actions use the LIBERO/robosuite OSC_POSE delta convention that the SmolVLA
checkpoint predicts: [dx, dy, dz, drx, dry, drz, gripper] in [-1, 1], where one
unit is 0.05 m / 0.5 rad per control step and gripper -1 opens, +1 closes.
"""

from __future__ import annotations

from dataclasses import dataclass

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
# Small offsets inside the basket so later butters land beside earlier ones.
DROP_OFFSETS = {"item_1": (0.0, -0.02), "item_2": (0.0, 0.02), "item_3": (0.0, 0.0)}


@dataclass
class Segment:
    name: str
    item: str
    target: str  # "item" | "basket" | "hold"
    z: float
    gripper: float
    max_speed: float = 1.0
    tolerance: float = 0.008
    fixed_steps: int = 0  # >0: hold position for this many steps instead of moving
    max_steps: int = 120


def _plan() -> list[Segment]:
    plan = []
    for item in ITEMS:
        plan += [
            Segment("approach", item, "item", CARRY_Z, OPEN),
            Segment("descend", item, "item", GRASP_Z, OPEN, max_speed=0.4, tolerance=0.005),
            Segment("grasp", item, "hold", GRASP_Z, CLOSE, fixed_steps=GRIP_STEPS),
            Segment("lift", item, "hold", CARRY_Z, CLOSE, max_speed=0.6),
            Segment("carry", item, "basket", CARRY_Z, CLOSE),
            Segment("lower", item, "basket", RELEASE_Z, CLOSE, max_speed=0.4),
            Segment("release", item, "hold", RELEASE_Z, OPEN, fixed_steps=GRIP_STEPS),
            Segment("retreat", item, "hold", CARRY_Z, OPEN, max_speed=0.6),
        ]
    return plan


class ScriptedThreeItemsExpert:
    """Call reset(env) after env.reset(), then act(env) once per control step."""

    def __init__(self, rng: np.random.Generator | None = None, jitter: float = 0.0):
        self.rng = rng or np.random.default_rng(0)
        self.jitter = jitter
        self.plan = _plan()

    def reset(self, env) -> None:
        self.index = 0
        self.steps_in_segment = 0
        self.target_quat = self._eef_quat(env)
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
        rot = np.clip(self._rotation_error(env) / ROT_SCALE, -0.5, 0.5)
        action = np.concatenate([delta, rot, [seg.gripper]]).astype(np.float32)

        self.steps_in_segment += 1
        finished = (
            self.steps_in_segment >= seg.fixed_steps
            if seg.fixed_steps
            else np.linalg.norm(error) < seg.tolerance or self.steps_in_segment >= seg.max_steps
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
            xy = self._body_pos(env, BASKET)[:2] + np.array(DROP_OFFSETS[seg.item])
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

    def _rotation_error(self, env) -> np.ndarray:
        """Axis-angle rotation (world frame) from the current to the initial gripper orientation."""
        q = T.quat_multiply(self.target_quat, T.quat_inverse(self._eef_quat(env)))
        if q[3] < 0:
            q = -q
        return T.quat2axisangle(q)
