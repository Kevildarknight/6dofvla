"""Check the randomised layouts through the same env class lerobot-eval uses.

For the train (task 1) and heldout (task 2) splits:
  1. restore    - resetting to state i puts every body at layout i's x, y
                  (within 1 mm) and every butter at its yaw (within 1 deg).
  2. valid      - each layout respects the spacing rules in layouts.py.
  3. scripted   - the scripted controller succeeds from the first N layouts.
And across splits:
  4. separate   - no held-out layout is a near copy of a training layout.

Writes report.json and one scripted rollout video per split.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
from lerobot_env_libero_three_items.env import ThreeItemsLiberoEnv
from lerobot_env_libero_three_items.layouts import Layout, is_valid
from lerobot_env_libero_three_items.scripted import ScriptedThreeItemsExpert
from lerobot_env_libero_three_items.task import EPISODE_LENGTH, ITEMS, SPLITS, layouts_path

XY_TOLERANCE = 0.001
YAW_TOLERANCE = math.radians(1.0)
# Two layouts count as near copies when every body is within this distance.
NEAR_COPY = 0.03


def load_layouts(split):
    return [Layout.from_dict(d) for d in json.loads(layouts_path(split).read_text())["layouts"]]


def reset_to(env, index):
    env.init_state_id = index  # LeRobot's LiberoEnv restores state init_state_id on reset
    env.reset(seed=index)


def yaw_error(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def check_restore(env, layouts):
    records = []
    for i, layout in enumerate(layouts):
        reset_to(env, i)
        sim_env = env.sim_env
        xy_err = max(
            float(np.linalg.norm(sim_env.sim.data.body_xpos[sim_env.obj_body_id[n]][:2] - xy))
            for n, xy in layout.xy.items()
        )
        yaw_err = max(
            yaw_error(ScriptedThreeItemsExpert.item_yaw(sim_env, n), layout.item_yaw[n]) for n in ITEMS
        )
        records.append(
            {
                "layout": i,
                "max_xy_error_m": xy_err,
                "max_yaw_error_deg": math.degrees(yaw_err),
                "valid_spacing": is_valid(layout),
                "passed": xy_err < XY_TOLERANCE and yaw_err < YAW_TOLERANCE and is_valid(layout),
            }
        )
    return {"passed": all(r["passed"] for r in records), "layouts": records}


def check_scripted(env, n_rollouts, video_path):
    records = []
    for i in range(n_rollouts):
        reset_to(env, i)
        expert = ScriptedThreeItemsExpert()
        expert.reset(env.sim_env)
        frames = [env.render()] if i == 0 else None
        success_step = None
        for t in range(EPISODE_LENGTH):
            _, _, terminated, _, info = env.step(expert.act(env.sim_env))
            if frames is not None:
                frames.append(env.render())
            if terminated:
                success_step = t if info["is_success"] else None
                break
        if frames is not None:
            imageio.mimsave(video_path, frames, fps=20)
        records.append({"layout": i, "success_step": success_step, "passed": success_step is not None})
    return {"passed": all(r["passed"] for r in records), "rollouts": records, "video": str(video_path)}


def check_separate(train, heldout):
    def distance(a, b):
        return max(float(np.linalg.norm(np.subtract(a.xy[n], b.xy[n]))) for n in a.xy)

    closest = min(distance(h, t) for h in heldout for t in train)
    return {"passed": closest > NEAR_COPY, "closest_pair_max_body_distance_m": closest}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollouts", type=int, default=3, help="scripted rollouts per split")
    args = parser.parse_args()
    out = args.output_dir or Path("outputs") / f"three_items_layouts_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    report = {}
    layouts = {}
    for split in ("train", "heldout"):
        layouts[split] = load_layouts(split)
        env = ThreeItemsLiberoEnv(
            task_id=SPLITS[split],
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_length=EPISODE_LENGTH,
        )
        report[f"{split}_restore"] = check_restore(env, layouts[split])
        report[f"{split}_scripted"] = check_scripted(env, args.rollouts, out / f"{split}_layout0.mp4")
        env.close()
    report["separate"] = check_separate(layouts["train"], layouts["heldout"])
    report["passed"] = all(section["passed"] for section in report.values())
    (out / "report.json").write_text(json.dumps(report, indent=2))

    for name, section in report.items():
        if name != "passed":
            print(f"{'PASS' if section['passed'] else 'FAIL'}  {name}")
    print(f"Report: {out / 'report.json'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
