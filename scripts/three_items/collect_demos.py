"""Record scripted demonstrations of the three-butter task as a LeRobot dataset.

Each episode: reset to the next saved layout of the chosen task (by default
task 1, the 40 training layouts, visited in turn), let the scripted expert pick
the three butters one at a time and drop each in the basket, and stop when the
strict success check fires. Only successful episodes are saved.

Variety between episodes: the layout, up to +-1 cm of waypoint noise, a random
pick order, and the instruction (the evaluation sentence for half of the
episodes, otherwise one of LANGUAGE_VARIANTS).

With --noise (DART), the robot executes the expert's action plus noise, while
the dataset stores the expert's clean action for the state the robot is in.
The arm then drifts off the ideal path, and the expert, which steers from the
current position, records how to get back. Demonstrations without noise never
show a correction, so a policy trained on them has nothing to fall back on
once it drifts. Noise is off while the gripper closes or opens, so it does not
just make the robot drop a butter.

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
from lerobot_env_libero_three_items.task import EPISODE_LENGTH, LANGUAGE, LANGUAGE_VARIANTS, SPLITS, TASK_TRAIN

from dataset_features import FEATURES, FPS, IMAGE_KEYS, SIZE


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


class ExecutionNoise:
    """Noise added to executed actions: small per-step jitter plus occasional pushes.

    A push holds one random direction for several steps, which moves the gripper
    a few centimetres; per-step jitter alone averages out within a few steps.
    Values are in action units (1.0 = 5 cm or 0.5 rad of commanded motion).
    """

    def __init__(self, rng, scale):
        self.rng = rng
        self.scale = scale
        self.push = np.zeros(6)
        self.push_steps = 0

    def reset(self):
        self.push[:] = 0
        self.push_steps = 0

    def __call__(self, action, active):
        if not active or self.scale == 0:
            self.reset()
            return action
        if self.push_steps == 0 and self.rng.random() < 0.04:
            direction = self.rng.normal(size=3)
            self.push[:3] = direction / np.linalg.norm(direction) * self.rng.uniform(0.3, 0.6)
            self.push[3:] = self.rng.normal(0, 0.15, size=3)
            self.push_steps = int(self.rng.integers(5, 16))
        jitter = np.concatenate([self.rng.normal(0, 0.1, 3), self.rng.normal(0, 0.05, 3)])
        noisy = action.copy()
        noisy[:6] += self.scale * (jitter + (self.push if self.push_steps else 0))
        self.push_steps = max(self.push_steps - 1, 0)
        return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/three_items_demos"))
    parser.add_argument("--repo-id", default="local/three_items_demos")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--jitter", type=float, default=0.01, help="waypoint noise in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task-id", type=int, default=TASK_TRAIN, choices=sorted(SPLITS.values()), help=f"layouts: {SPLITS}"
    )
    parser.add_argument("--fixed-order", action="store_true", help="always pick item_1, item_2, item_3")
    parser.add_argument(
        "--variant-prob", type=float, default=0.5, help="share of episodes using a rephrased instruction"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, help="scale of execution noise (DART); 0 records clean runs"
    )
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
        task_id=args.task_id,
        obs_type="pixels_agent_pos",
        observation_height=SIZE,
        observation_width=SIZE,
        episode_length=EPISODE_LENGTH,
    )
    rng = np.random.default_rng(args.seed)
    expert = ScriptedThreeItemsExpert(rng=rng, jitter=args.jitter, shuffle_order=not args.fixed_order)
    noise = ExecutionNoise(rng, args.noise)

    saved = attempts = 0
    while saved < args.episodes:
        attempts += 1
        layout = env.init_state_id % len(env._init_states)  # the state this reset restores
        observation, _ = env.reset(seed=args.seed + attempts)
        expert.reset(env.sim_env)
        noise.reset()
        pushed = 0
        task = str(rng.choice(LANGUAGE_VARIANTS)) if rng.random() < args.variant_prob else LANGUAGE
        success = False
        for _ in range(EPISODE_LENGTH):
            segment = expert.segment
            action = expert.act(env.sim_env)
            dataset.add_frame({**to_policy_frame(observation, env_preprocessor), "action": action, "task": task})
            # Gripper closing/opening segments hold still; perturb everything else.
            executed = noise(action, active=segment is not None and not segment.fixed_steps)
            pushed += noise.push_steps > 0
            observation, _, terminated, _, info = env.step(executed)
            if terminated:
                success = info["is_success"]
                break
        if success:
            dataset.save_episode()
            saved += 1
            print(
                f"episode {saved}/{args.episodes}: layout {layout}, order {' '.join(expert.order)}, "
                f"task '{task}', {pushed} pushed steps, {dataset.meta.total_frames} frames total",
                flush=True,
            )
        else:
            dataset.clear_episode_buffer()
            print(f"attempt {attempts}: layout {layout} failed, episode discarded", flush=True)

    dataset.finalize()
    env.close()
    print(f"Saved {saved} episodes ({attempts} attempts) to {args.root}")


if __name__ == "__main__":
    main()
