"""RoboTwin policy module for an ACT checkpoint from act_train.sh (single-arm Panthera).

RoboTwin's policy/ACT/deploy_policy.py reads the left, right and head cameras
and a 14-D dual-arm state. This module does the same for one arm: head and
wrist images, resized as act_convert.py resized them, and the 7-D joint vector.
It reuses RoboTwin's ACT class (policy/ACT/act_policy.py), which loads
policy_best.ckpt and dataset_stats.pkl from ckpt_dir and, with temporal_agg,
queries the model every step and averages the overlapping chunks.

RoboTwin's take_action expects a 14-D dual-arm qpos; in single-arm mode the
right-arm API is an alias of the left arm, so the 7-D action is sent twice.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path("policy/ACT").resolve()))  # eval_policy.py runs from RoboTwin's root
from act_policy import ACT  # noqa: E402

CAMERAS = {"cam_high": "head_camera", "cam_wrist": "wrist_camera"}


def get_model(usr_args):
    usr_args["camera_names"] = list(CAMERAS)
    model = ACT(usr_args, Namespace(**usr_args))
    info = json.loads((Path(usr_args["ckpt_dir"]) / "act_dataset.json").read_text())  # copied by act_train.py
    height, width = info["image_size"]
    model.image_size = (width, height)  # cv2 order
    return model


def encode_obs(observation, size):
    cameras = observation["observation"]
    obs = {
        name: np.moveaxis(cv2.resize(cameras[source]["rgb"], size, interpolation=cv2.INTER_AREA), -1, 0) / 255.0
        for name, source in CAMERAS.items()
    }
    obs["qpos"] = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return obs


def eval(TASK_ENV, model, observation):
    for action in model.get_action(encode_obs(observation, model.image_size)):
        action = np.array(action)
        action[6] = np.clip(action[6], 0, 1)  # gripper command range in RoboTwin
        TASK_ENV.take_action(np.concatenate([action, action]), action_type="qpos")


def reset_model(model):
    model.t = 0
    if model.temporal_agg:
        model.all_time_actions.zero_()
