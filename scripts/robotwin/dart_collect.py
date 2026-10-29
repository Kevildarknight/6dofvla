"""Run RoboTwin's script/collect_data.py with DART-style perturbations of the expert.

A policy trained on clean expert demonstrations never sees the arm off the
expert's path, so in closed loop small errors grow until the grasp misses (see
README, "Evaluate"). DART fixes this by perturbing the expert during collection
and recording how it recovers.

Here, during the planning phase, each unconstrained arm motion is perturbed with
probability DART_PROB (default 0.6):
  1. the planned path runs for a random 20-80% of its length ("prefix"),
  2. the joints are pushed to a random offset, Gaussian with std DART_STD rad
     (default 0.05), clipped to the joint limits ("noise"),
  3. CuRobo replans from there to the original target (the recovery).
Constrained motions (the straight final approach of a grasp) are left alone.
The prefix and noise segments are stored in _traj_data next to the expert's own
paths, tagged with a "dart" key, so the replay phase executes the same motions.

During replay, the frames recorded while a noise segment runs are listed in
dart/episodeN.json as [first, last] frame index pairs. The first frame shows the
state before the push and the last the pushed state. convert_to_lerobot.py drops
every transition that leads into a noise segment, so the dataset teaches the
recovery but never the push.

Run from RoboTwin's root folder with collect_data.py's arguments; collect.sh does
this when DART=1.
"""

import json
import os
import runpy
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path[:0] = ["./script", "./"]  # as if run as script/collect_data.py from RoboTwin's root
from envs._base_task import Base_Task  # noqa: E402

PROB = float(os.environ.get("DART_PROB", 0.6))
STD = float(os.environ.get("DART_STD", 0.05))
CUT = (0.2, 0.8)
MIN_STEPS = 50  # do not cut paths shorter than this many physics steps (0.2 s)
LIMIT_MARGIN = 0.05

init_task_env = Base_Task._init_task_env_
left_move_to_pose = Base_Task.left_move_to_pose


def init_task_env_dart(self, *args, **kwargs):
    init_task_env(self, *args, **kwargs)
    self.dart_rng = np.random.default_rng(1_000_003 + int(kwargs.get("seed", 0)))
    self.dart_frames = []
    if self.save_data:
        write_dart_frames(self)


def write_dart_frames(self):
    path = Path(self.save_dir) / "dart" / f"episode{self.ep_num}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"noise_frames": self.dart_frames, "prob": PROB, "std": STD}))


def run_segment(self, segment):
    self.take_dense_action({"left_arm": segment, "left_gripper": None, "right_arm": None, "right_gripper": None})


def run_noise(self, segment):
    first = self.FRAME_IDX
    run_segment(self, segment)
    if self.save_data:
        self.dart_frames.append([first, self.FRAME_IDX - 1])
        write_dart_frames(self)


def noise_segment(self):
    joints = self.robot.left_arm_joints
    start = np.array(self.robot.get_left_arm_jointState()[: len(joints)])
    limits = np.array([joint.get_limits()[0] for joint in joints])
    target = np.clip(
        start + self.dart_rng.normal(0, STD, start.shape), limits[:, 0] + LIMIT_MARGIN, limits[:, 1] - LIMIT_MARGIN
    )
    _, position, velocity, _, _ = self.robot.left_mplib_planner.TOPP(np.vstack([start, target]), 1 / 250, verbose=True)
    return {"status": "Success", "dart": "noise", "position": position, "velocity": velocity}


def left_move_to_pose_dart(self, pose, constraint_pose=None, *args, **kwargs):
    if not self.plan_success or pose is None:
        return left_move_to_pose(self, pose, constraint_pose, *args, **kwargs)

    if not self.need_plan:
        # Replay: execute the stored DART segments that precede this motion.
        while self.left_cnt < len(self.left_joint_path) and self.left_joint_path[self.left_cnt].get("dart"):
            segment = deepcopy(self.left_joint_path[self.left_cnt])
            self.left_cnt += 1
            if segment["dart"] == "noise":
                run_noise(self, segment)
            else:
                run_segment(self, segment)
        return left_move_to_pose(self, pose, constraint_pose, *args, **kwargs)

    if constraint_pose is not None or self.dart_rng.random() >= PROB:
        return left_move_to_pose(self, pose, constraint_pose, *args, **kwargs)
    planned = left_move_to_pose(self, pose, constraint_pose, *args, **kwargs)
    steps = 0 if planned is None else planned["position"].shape[0]
    if steps < MIN_STEPS:
        return planned
    self.left_joint_path.pop()  # replaced by prefix, noise and the replanned recovery

    cut = int(steps * self.dart_rng.uniform(*CUT))
    prefix = {
        "status": "Success",
        "dart": "prefix",
        "position": planned["position"][:cut],
        "velocity": planned["velocity"][:cut],
    }
    run_segment(self, prefix)
    self.left_joint_path.append(deepcopy(prefix))
    try:
        noise = noise_segment(self)
    except Exception:  # TOPP can fail on a degenerate path; skip the push then
        noise = None
    if noise is not None:
        run_noise(self, noise)
        self.left_joint_path.append(deepcopy(noise))
    return left_move_to_pose(self, pose, constraint_pose, *args, **kwargs)


Base_Task._init_task_env_ = init_task_env_dart
Base_Task.left_move_to_pose = left_move_to_pose_dart
print(f"DART collection: prob {PROB}, std {STD} rad", flush=True)
sys.argv[0] = "script/collect_data.py"
runpy.run_path("script/collect_data.py", run_name="__main__")
