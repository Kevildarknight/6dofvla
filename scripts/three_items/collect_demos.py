"""Record scripted demonstrations of the three-butter task as a LeRobot dataset.

Each episode: reset to the fixed layout, let the scripted expert pick item_1,
item_2 and item_3 in turn and drop each in the basket, and stop when the strict
success check fires. Only successful episodes are saved.

Observations go through the same steps lerobot-eval applies before the policy
(preprocess_observation + LiberoProcessorStep), so the stored images are rotated
180 degrees and the state is [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)],
matching the features of models/smolvla_libero.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import preprocess_observation
from lerobot_env_libero_three_items.config import LiberoThreeItemsEnv
from lerobot_env_libero_three_items.env import ThreeItemsLiberoEnv
from lerobot_env_libero_three_items.scripted import ScriptedThreeItemsExpert
from lerobot_env_libero_three_items.task import EPISODE_LENGTH, LANGUAGE

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
SIZE = 256
FPS = 20

FEATURES = {
    **{
        key: {"dtype": "video", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channels"]}
        for key in IMAGE_KEYS
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_1", "gripper_2"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
}


def to_policy_frame(observation, env_preprocessor):
    """Apply lerobot-eval's observation pipeline to one unbatched env observation."""
    batched = {
        "pixels": {k: v[None] for k, v in observation["pixels"].items()},
        "robot_state": {
            group: {k: np.asarray(v)[None] for k, v in values.items()}
            for group, values in observation["robot_state"].items()
        },
    }
    processed = env_preprocessor(preprocess_observation(batched))
    frame = {
        key: (processed[key][0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy() for key in IMAGE_KEYS
    }
    frame["observation.state"] = processed["observation.state"][0].numpy().astype(np.float32)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/three_items_demos"))
    parser.add_argument("--repo-id", default="local/three_items_demos")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--jitter", type=float, default=0.01, help="waypoint noise in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it.")
        shutil.rmtree(args.root)

    env_preprocessor, _ = LiberoThreeItemsEnv().get_env_processors()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=FEATURES, root=args.root, robot_type="panda", use_videos=True
    )
    env = ThreeItemsLiberoEnv(
        obs_type="pixels_agent_pos",
        observation_height=SIZE,
        observation_width=SIZE,
        episode_length=EPISODE_LENGTH,
    )
    expert = ScriptedThreeItemsExpert(rng=np.random.default_rng(args.seed), jitter=args.jitter)

    saved = attempts = 0
    while saved < args.episodes:
        attempts += 1
        observation, _ = env.reset(seed=args.seed + attempts)
        expert.reset(env.sim_env)
        success = False
        for _ in range(EPISODE_LENGTH):
            action = expert.act(env.sim_env)
            dataset.add_frame({**to_policy_frame(observation, env_preprocessor), "action": action, "task": LANGUAGE})
            observation, _, terminated, _, info = env.step(action)
            if terminated:
                success = info["is_success"]
                break
        if success:
            dataset.save_episode()
            saved += 1
            print(f"episode {saved}/{args.episodes}: {dataset.meta.total_frames} frames total")
        else:
            dataset.clear_episode_buffer()
            print(f"attempt {attempts}: scripted expert failed, episode discarded")

    dataset.finalize()
    env.close()
    print(f"Saved {saved} episodes ({attempts} attempts) to {args.root}")


if __name__ == "__main__":
    main()
