"""Success check: every item rests inside the basket, not just the last one.

An item counts as "in the basket" at a control step when all of these hold:
  * its centre lies inside the basket's contain_region box (expressed in the
    basket frame, shrunk by a small margin so an item balanced on the rim fails),
  * it is not touching the gripper (the robot has let go of it),
  * it moved less than SETTLE_SPEED since the previous control step.

The task succeeds once all three items satisfy this for HOLD_STEPS consecutive
control steps, so an item swinging through the basket or still held over it
never triggers success.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .task import BASKET, ITEMS

CONTAIN_SITE = f"{BASKET}_contain_region"
XY_MARGIN = 0.005  # metres trimmed from each side of the contain box
SETTLE_SPEED = 0.02  # m/s
HOLD_STEPS = 10  # 0.5 s at 20 Hz


@dataclass
class ItemStatus:
    inside: bool
    released: bool
    settled: bool

    @property
    def in_basket(self) -> bool:
        return self.inside and self.released and self.settled


@dataclass
class SuccessStatus:
    items: dict[str, ItemStatus]
    hold_count: int
    success: bool

    @property
    def num_in_basket(self) -> int:
        return sum(s.in_basket for s in self.items.values())


@dataclass
class BasketSuccessChecker:
    """Stateful checker; call reset() after every env reset and update() after every step."""

    control_dt: float = 0.05
    hold_steps: int = HOLD_STEPS
    _prev_pos: dict[str, np.ndarray] = field(default_factory=dict)
    _hold: int = 0

    def reset(self, env) -> None:
        self._prev_pos = {name: self._item_pos(env, name) for name in ITEMS}
        self._hold = 0

    def update(self, env) -> SuccessStatus:
        items = {}
        for name in ITEMS:
            pos = self._item_pos(env, name)
            speed = np.linalg.norm(pos - self._prev_pos[name]) / self.control_dt
            self._prev_pos[name] = pos
            items[name] = ItemStatus(
                inside=self.inside_basket(env, pos),
                released=not env.check_contact(env.robots[0].gripper, env.get_object(name)),
                settled=bool(speed < SETTLE_SPEED),
            )
        self._hold = self._hold + 1 if all(s.in_basket for s in items.values()) else 0
        return SuccessStatus(items=items, hold_count=self._hold, success=self._hold >= self.hold_steps)

    @staticmethod
    def _item_pos(env, name: str) -> np.ndarray:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])

    @staticmethod
    def inside_basket(env, item_pos: np.ndarray) -> bool:
        """Point-in-box test against the basket's contain_region site, in the site frame."""
        site_id = env.sim.model.site_name2id(CONTAIN_SITE)
        centre = env.sim.data.site_xpos[site_id]
        rot = env.sim.data.site_xmat[site_id].reshape(3, 3)
        half = np.array(env.sim.model.site_size[site_id])
        local = rot.T @ (item_pos - centre)
        # The site's local z axis may be tilted with the basket, so test each axis
        # in the site frame. xy shrink rejects items resting on the rim.
        limit = half - np.array([XY_MARGIN, XY_MARGIN, 0.0])
        return bool(np.all(np.abs(local) < limit))
