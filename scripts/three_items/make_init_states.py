"""Write the saved initial states for the three-butter task.

Splits (task IDs of --env.type=libero_three_items):
  fixed    task 0: the single layout in task.LAYOUT_XY
  train    task 1: randomly sampled layouts for demonstrations
  heldout  task 2: layouts sampled with another seed, for evaluation only

For each layout the script resets the scene, teleports the basket and butters
into place, lets them settle, and records the MuJoCo state. A layout is kept
only if the scripted controller, started from that state, gets all three
butters into the basket (the strict success check). Each split is saved as
<name>.pruned_init plus <name>.layouts.json with the layouts in the same order.
"""

import argparse
import json

import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
from lerobot.envs.libero import get_libero_dummy_action
from lerobot_env_libero_three_items.layouts import FIXED_LAYOUT, Layout, apply_layout, sample_layout
from lerobot_env_libero_three_items.scripted import ScriptedThreeItemsExpert
from lerobot_env_libero_three_items.success import BasketSuccessChecker
from lerobot_env_libero_three_items.task import (
    EPISODE_LENGTH,
    RANDOM_SPLITS,
    SPLITS,
    bddl_path,
    init_states_path,
    layouts_path,
)

SETTLE_STEPS = 50
MAX_DRIFT = 0.005  # metres an object may move while settling
IDLE = get_libero_dummy_action()


def settle_layout(env, layout: Layout) -> np.ndarray | None:
    """Place a layout and return the settled state, or None if objects drifted."""
    env.reset()
    sim_env = env.env
    apply_layout(sim_env, layout)
    for _ in range(SETTLE_STEPS):
        env.step(IDLE)
    for name, (x, y) in layout.xy.items():
        pos = sim_env.sim.data.body_xpos[sim_env.obj_body_id[name]]
        if np.linalg.norm(pos[:2] - (x, y)) > MAX_DRIFT:
            return None
    return np.asarray(env.get_sim_state(), dtype=np.float64)


def scripted_success_step(env) -> int | None:
    """Run the scripted controller from the current state; return the success step."""
    sim_env = env.env
    expert = ScriptedThreeItemsExpert()
    expert.reset(sim_env)
    checker = BasketSuccessChecker()
    checker.reset(sim_env)
    for t in range(EPISODE_LENGTH):
        env.step(expert.act(sim_env))
        if checker.update(sim_env).success:
            return t
    return None


def make_split(env, split: str, verify: bool) -> None:
    if split == "fixed":
        candidates = iter([FIXED_LAYOUT])
        count = 1
    else:
        count, seed = RANDOM_SPLITS[split]
        rng = np.random.default_rng(seed)
        candidates = iter(lambda: sample_layout(rng), None)

    states, records = [], []
    rejected = 0
    while len(states) < count:
        layout = next(candidates)
        state = settle_layout(env, layout)
        step = scripted_success_step(env) if state is not None and verify else None
        if state is None or (verify and step is None):
            rejected += 1
            print(f"[{split}] rejected layout ({'drifted' if state is None else 'scripted run failed'})")
            continue
        states.append(state)
        records.append({**layout.to_dict(), "scripted_success_step": step})
        print(f"[{split}] layout {len(states)}/{count} kept (scripted success at step {step})")

    torch.save(np.stack(states), init_states_path(split))
    layouts_path(split).write_text(json.dumps({"rejected": rejected, "layouts": records}, indent=2))
    print(f"[{split}] saved {len(states)} states to {init_states_path(split)} ({rejected} rejected)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--no-verify", action="store_true", help="skip the scripted rollout check")
    args = parser.parse_args()

    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path()), camera_heights=64, camera_widths=64)
    for split in SPLITS if args.split == "all" else [args.split]:
        make_split(env, split, verify=not args.no_verify)
    env.close()


if __name__ == "__main__":
    main()
