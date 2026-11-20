"""RoboTwin policy module that forwards observations to policy_server.py.

RoboTwin's script/eval_policy.py imports this module by name (policy_name
smolvla_panthera, found through PYTHONPATH) and calls get_model, reset_model and
eval. It runs inside RoboTwin's Python 3.10 env and needs no LeRobot.

Per call of eval: send the current head and wrist images, the 7-D joint state
and the instruction; receive a few actions; execute them one by one. RoboTwin's
take_action expects a 14-D dual-arm qpos; in single-arm mode the right-arm API
is an alias of the left arm, so the 7-D action is sent for both halves.

RoboTwin's per-task step limits (task_config/_eval_step_limit.yml) were set for
its default 16.7 Hz recording; at our 31.25 Hz the same motion takes about 1.9
times as many steps. A step_lim override (eval.sh: STEP_LIM) replaces the limit.

Diagnostics: with SMOLVLA_TRACE_DIR set, every executed action is appended to
<dir>/episodeN.jsonl with the joint state the action was predicted from, the
action, and after the step the joint drive targets, gripper value, end-effector
pose and the positions of the task objects named in SMOLVLA_TRACE_ACTORS
(comma-separated attributes, e.g. "pillbottle,pad").
"""

import json
import os
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

AUTHKEY = b"robotwin-smolvla"


def pack(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image.tobytes(), image.shape


class SmolVLAClient:
    def __init__(self, port, step_lim=None):
        self.connection = Client(("localhost", port), authkey=AUTHKEY)
        self.step_lim = step_lim
        trace_dir = os.environ.get("SMOLVLA_TRACE_DIR")
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.trace_actors = [a for a in os.environ.get("SMOLVLA_TRACE_ACTORS", "").split(",") if a]
        self.episode = -1

    def act(self, observation, task):
        cameras = observation["observation"]
        self.connection.send(
            {
                "head": pack(cameras["head_camera"]["rgb"]),
                "wrist": pack(cameras["wrist_camera"]["rgb"]),
                "state": np.asarray(observation["joint_action"]["vector"], dtype=float).tolist(),
                "task": task,
            }
        )
        return np.asarray(self.connection.recv()["actions"])

    def reset(self):
        self.connection.send({"reset": True})
        self.connection.recv()
        self.episode += 1
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            (self.trace_dir / f"episode{self.episode}.jsonl").write_text("")

    def trace(self, task_env, state, action):
        if not self.trace_dir:
            return
        row = {
            "step": int(task_env.take_action_cnt),
            "request_state": [float(x) for x in state],
            "action": [float(x) for x in action],
            "joint_target": [float(x) for x in task_env.robot.get_left_arm_jointState()],
            "gripper": float(task_env.robot.get_left_gripper_val()),
            "endpose": [float(x) for x in task_env.get_arm_pose("left")],
            "success": bool(task_env.eval_success),
        }
        for name in self.trace_actors:
            actor = getattr(task_env, name, None)
            if actor is not None:
                row[name] = [float(x) for x in actor.get_pose().p]
        with open(self.trace_dir / f"episode{self.episode}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def get_model(usr_args):
    step_lim = usr_args.get("step_lim")
    return SmolVLAClient(int(usr_args.get("port", 6010)), int(step_lim) if step_lim else None)


def eval(TASK_ENV, model, observation):
    if model.step_lim:
        TASK_ENV.step_lim = model.step_lim  # read by eval_policy's loop and take_action
    actions = model.act(observation, TASK_ENV.get_instruction())
    actions[:, 6] = np.clip(actions[:, 6], 0, 1)  # gripper command range in RoboTwin
    state = observation["joint_action"]["vector"]
    for action in actions:
        TASK_ENV.take_action(np.concatenate([action, action]), action_type="qpos")
        model.trace(TASK_ENV, state, action)
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    model.reset()  # clears the server's temporal ensemble, if one is used
