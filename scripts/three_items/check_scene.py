"""Validate the three-butter scene and its success check before collecting data.

Checks, each recorded in report.json:
  1. reset_layout   - every reset restores the same robot pose and object layout.
  2. reachability   - a scripted controller reaches, lifts, and drops every item.
  3. per_item       - placing item k in the basket flags item k only; an item
                      outside the basket, or still falling, is not counted.
  4. all_required   - success stays False with one or two items in the basket and
                      turns True only after all three rest there for HOLD_STEPS.
  5. scripted_full  - a full scripted episode ends with is_success == True, and
                      success never fires before the third item is in.

The scripted rollout is saved as scripted_episode.mp4 next to the report.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
from lerobot.envs.libero import get_libero_dummy_action
from lerobot_env_libero_three_items.env import ThreeItemsLiberoEnv
from lerobot_env_libero_three_items.scripted import CARRY_Z, ScriptedThreeItemsExpert
from lerobot_env_libero_three_items.success import HOLD_STEPS
from lerobot_env_libero_three_items.task import BASKET, EPISODE_LENGTH, ITEMS, LAYOUT_XY

LAYOUT_TOLERANCE = 0.001  # metres
IDLE = np.array(get_libero_dummy_action(), dtype=np.float32)


def body_pos(sim_env, name):
    return np.array(sim_env.sim.data.body_xpos[sim_env.obj_body_id[name]])


def place(sim_env, name, xyz):
    """Teleport a free-jointed object, keeping its orientation, and stop it."""
    joint = sim_env.get_object(name).joints[0]
    qpos = sim_env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = xyz
    sim_env.sim.data.set_joint_qpos(joint, qpos)
    sim_env.sim.data.set_joint_qvel(joint, np.zeros(6))
    sim_env.sim.forward()


def basket_slot(sim_env, k):
    """A resting spot inside the basket for the k-th item (side by side along y)."""
    centre = body_pos(sim_env, BASKET)
    return centre + np.array([0.0, (k - 1) * 0.04, 0.04])


def run_idle(env, steps):
    """Step with the robot idle; return per-step (num_in_basket, is_success)."""
    history = []
    for _ in range(steps):
        _, _, _, _, info = env.step(IDLE)
        history.append((int(info["num_items_in_basket"]), bool(info["is_success"])))
    return history


def check_reset_layout(env, n_resets):
    records = []
    first_qpos = None
    for i in range(n_resets):
        env.reset(seed=i)
        sim_env = env.sim_env
        errors = {name: float(np.linalg.norm(body_pos(sim_env, name)[:2] - xy)) for name, xy in LAYOUT_XY.items()}
        qpos = np.array(sim_env.sim.data.qpos[sim_env.robots[0]._ref_joint_pos_indexes])
        first_qpos = qpos if first_qpos is None else first_qpos
        records.append(
            {
                "reset": i,
                "xy_error_m": errors,
                "robot_qpos_diff": float(np.abs(qpos - first_qpos).max()),
            }
        )
    passed = all(
        max(r["xy_error_m"].values()) < LAYOUT_TOLERANCE and r["robot_qpos_diff"] < 1e-6 for r in records
    )
    return {"passed": passed, "resets": records}


def check_per_item(env):
    cases = []
    for k, name in enumerate(ITEMS):
        env.reset(seed=0)
        place(env.sim_env, name, basket_slot(env.sim_env, k))
        history = run_idle(env, 40)
        status = env.last_status
        flags = {n: status.items[n].in_basket for n in ITEMS}
        cases.append(
            {
                "case": f"only {name} in basket",
                "in_basket": flags,
                "success_ever": any(s for _, s in history),
                "passed": flags == {n: n == name for n in ITEMS} and not any(s for _, s in history),
            }
        )

    # Next to the basket, touching its outer wall: must not count.
    env.reset(seed=0)
    beside = body_pos(env.sim_env, BASKET) + np.array([0.0, -0.10, 0.02])
    place(env.sim_env, ITEMS[0], beside)
    run_idle(env, 40)
    flag = env.last_status.items[ITEMS[0]].in_basket
    cases.append({"case": f"{ITEMS[0]} beside basket", "in_basket": flag, "passed": not flag})

    # Dropped from above: inside the box while falling, but must not count until it rests.
    env.reset(seed=0)
    place(env.sim_env, ITEMS[0], body_pos(env.sim_env, BASKET) + np.array([0.0, 0.0, 0.12]))
    falling = []
    for _ in range(40):
        env.step(IDLE)
        s = env.last_status.items[ITEMS[0]]
        falling.append({"inside": s.inside, "settled": s.settled, "in_basket": s.in_basket})
    counted_while_moving = any(f["inside"] and f["in_basket"] and not f["settled"] for f in falling)
    first_inside = next(i for i, f in enumerate(falling) if f["inside"])
    cases.append(
        {
            "case": f"{ITEMS[0]} falling into basket",
            "first_step_inside_box": first_inside,
            "counted_while_falling": bool(falling[first_inside]["in_basket"]),
            "in_basket_at_rest": falling[-1]["in_basket"],
            "passed": not counted_while_moving
            and not falling[first_inside]["in_basket"]
            and falling[-1]["in_basket"],
        }
    )
    return {"passed": all(c["passed"] for c in cases), "cases": cases}


def check_all_required(env):
    results = []
    for count in (1, 2, 3):
        env.reset(seed=0)
        for k in range(count):
            place(env.sim_env, ITEMS[k], basket_slot(env.sim_env, k))
        history = run_idle(env, 40)
        first_success = next((i for i, (_, s) in enumerate(history) if s), None)
        results.append(
            {
                "items_placed": count,
                "final_num_in_basket": history[-1][0],
                "first_success_step": first_success,
            }
        )
    passed = (
        results[0]["first_success_step"] is None
        and results[1]["first_success_step"] is None
        and results[2]["first_success_step"] is not None
        and results[2]["first_success_step"] >= HOLD_STEPS - 1
    )
    return {"passed": passed, "hold_steps": HOLD_STEPS, "results": results}


def check_scripted(env, video_path):
    env.reset(seed=0)
    sim_env = env.sim_env
    expert = ScriptedThreeItemsExpert()
    expert.reset(sim_env)
    frames = [env.render()]
    segments = {}
    first_in = {}
    success_step = None
    max_lift = {name: 0.0 for name in ITEMS}
    for t in range(EPISODE_LENGTH):
        seg = expert.segment
        action = expert.act(sim_env)
        _, _, terminated, _, info = env.step(action)
        frames.append(env.render())
        if seg is not None:
            key = f"{seg.item}:{seg.name}"
            segments.setdefault(key, {"start": t})
            segments[key]["end"] = t
            if seg.name == "lift":
                max_lift[seg.item] = max(max_lift[seg.item], float(body_pos(sim_env, seg.item)[2]))
        for name in ITEMS:
            if info[f"{name}_in_basket"] and name not in first_in:
                first_in[name] = t
        if info["is_success"]:
            success_step = t
            break
        if terminated:
            break
    imageio.mimsave(video_path, frames, fps=20)

    # Reachability: every grasp descent arrived at the item, every item was lifted
    # clear of the floor, and the carry segments reached the basket.
    reach = {
        name: {
            "lifted_to_m": round(max_lift[name], 3),
            "lifted": max_lift[name] > CARRY_Z - 0.05,
            "entered_basket_at_step": first_in.get(name),
        }
        for name in ITEMS
    }
    passed = (
        success_step is not None
        and all(r["lifted"] and r["entered_basket_at_step"] is not None for r in reach.values())
        and success_step > max(first_in.values())
    )
    return {
        "passed": passed,
        "success_step": success_step,
        "items": reach,
        "segments": segments,
        "video": str(video_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resets", type=int, default=5)
    args = parser.parse_args()
    out = args.output_dir or Path("outputs") / f"three_items_check_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    env = ThreeItemsLiberoEnv(
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=EPISODE_LENGTH,
    )
    scripted = check_scripted(env, out / "scripted_episode.mp4")
    report = {
        "reset_layout": check_reset_layout(env, args.resets),
        "reachability_and_scripted_full": scripted,
        "per_item": check_per_item(env),
        "all_required": check_all_required(env),
    }
    env.close()
    report["passed"] = all(section["passed"] for section in report.values())
    (out / "report.json").write_text(json.dumps(report, indent=2))

    for name, section in report.items():
        if name != "passed":
            print(f"{'PASS' if section['passed'] else 'FAIL'}  {name}")
    print(f"Report: {out / 'report.json'}")
    print(f"Scripted rollout video: {out / 'scripted_episode.mp4'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
