"""Write and check the initial state of every difficulty-level task.

For each level x container pair (levels.TASKS) the script builds the scene,
places the objects and the container at the task's layout, lets them settle,
saves the MuJoCo state as assets/levels/<task>.pruned_init, and then runs the
scripted controller from that state with the strict success check. The scene
is simulated without cameras, so this runs on the CPU and does not need the GPU.

A summary is printed and written to assets/levels/check.json: for every task,
whether the scripted run succeeded, at which step, and which items ended in the
container.
"""

import argparse
import json

import numpy as np
import torch
from libero.libero.envs.env_wrapper import ControlEnv
from lerobot.envs.libero import get_libero_dummy_action
from lerobot_env_libero_three_items.layouts import apply_layout
from lerobot_env_libero_three_items.levels import LEVELS_DIR, TASKS, write_bddl_files
from lerobot_env_libero_three_items.scripted import ScriptedThreeItemsExpert
from lerobot_env_libero_three_items.success import BasketSuccessChecker

SETTLE_STEPS = 50
MAX_STEPS = 900
IDLE = get_libero_dummy_action()


def run(task, save: bool) -> dict:
    env = ControlEnv(
        bddl_file_name=str(LEVELS_DIR / f"{task.name}.bddl"),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        camera_names=[],
    )
    env.reset()
    sim_env = env.env
    apply_layout(sim_env, task.layout)
    for _ in range(SETTLE_STEPS):
        env.step(IDLE)
    drift = max(
        float(np.linalg.norm(sim_env.sim.data.body_xpos[sim_env.obj_body_id[b]][:2] - xy))
        for b, xy in task.layout.xy.items()
    )
    state = np.asarray(env.get_sim_state(), dtype=np.float64)[None]
    if save:
        torch.save(state, LEVELS_DIR / f"{task.name}.pruned_init")

    expert = ScriptedThreeItemsExpert()
    expert.reset(sim_env)
    checker = BasketSuccessChecker()
    checker.reset(sim_env)
    success_step, status = None, None
    for t in range(MAX_STEPS):
        env.step(expert.act(sim_env))
        status = checker.update(sim_env)
        if status.success:
            success_step = t
            break
    env.close()
    return {
        "task": task.name,
        "language": task.language,
        "settle_drift_mm": round(drift * 1000, 2),
        "success_step": success_step,
        "in_container": {k: v.in_basket for k, v in status.items.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="run a single task by name")
    parser.add_argument("--no-save", action="store_true", help="check without writing .pruned_init files")
    args = parser.parse_args()

    write_bddl_files()
    results = []
    for task in TASKS:
        if args.only and task.name != args.only:
            continue
        r = run(task, save=not args.no_save)
        results.append(r)
        ok = "PASS" if r["success_step"] is not None else "FAIL"
        print(f"{ok}  {r['task']:<28} step {r['success_step']}  drift {r['settle_drift_mm']} mm  {r['in_container']}", flush=True)
    if not args.only:
        (LEVELS_DIR / "check.json").write_text(json.dumps(results, indent=2))
    raise SystemExit(0 if all(r["success_step"] is not None for r in results) else 1)


if __name__ == "__main__":
    main()
