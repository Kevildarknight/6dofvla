"""Write the fixed initial state for the three-butter task.

LIBERO samples object poses inside the BDDL regions, which moves them by a
centimetre or two between resets. This script places item_1..item_3 and the
basket at task.LAYOUT_XY, lets the scene settle, and saves the MuJoCo state.
Every env reset restores that state, so all episodes start from the same layout.
"""

import argparse

import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
from lerobot.envs.libero import get_libero_dummy_action
from lerobot_env_libero_three_items.task import LAYOUT_XY, bddl_path, init_states_path

SETTLE_STEPS = 50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(init_states_path()))
    args = parser.parse_args()

    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path()), camera_heights=128, camera_widths=128)
    env.reset()
    sim = env.env.sim
    for name, (x, y) in LAYOUT_XY.items():
        joint = env.env.get_object(name).joints[0]
        qpos = sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = (x, y)
        sim.data.set_joint_qpos(joint, qpos)
        sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()
    for _ in range(SETTLE_STEPS):
        env.step(get_libero_dummy_action())

    for name, (x, y) in LAYOUT_XY.items():
        pos = sim.data.body_xpos[env.env.obj_body_id[name]]
        drift = np.linalg.norm(pos[:2] - (x, y))
        print(f"{name}: settled at {np.round(pos, 4)} (drift {drift * 1000:.1f} mm)")
        if drift > 0.005:
            raise SystemExit(f"{name} moved {drift:.3f} m while settling; check the layout.")

    state = np.asarray(env.get_sim_state(), dtype=np.float64)[None]
    torch.save(state, args.output)
    print(f"Saved {state.shape} initial state to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
