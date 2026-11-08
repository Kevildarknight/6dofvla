"""Write a dataset card for a converted RoboTwin Panthera dataset and push it to the Hugging Face Hub.

Usage (6dofvla .venv, after `hf auth login`):
  .venv/bin/python scripts/robotwin/push_dataset.py data/<dataset> <user>/<name> [--public] [--card-only]

The card uses LeRobot's own template (the "visualize this dataset" badge and
meta/info.json) with a description of the task, cameras, state and action.
--card-only writes README.md into the dataset folder and pushes nothing. The
repository is private unless --public is given.
"""

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_lerobot_dataset_card

DESCRIPTION = """\
Single-arm demonstrations of the RoboTwin task `move_pillbottle_pad` on a simulated
**Panthera-HT** arm (6 revolute joints and a 1-DoF gripper), collected in
[Panthera-HT_RoboTwin](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin)
(SAPIEN) and converted to LeRobotDataset v3 for fine-tuning SmolVLA.

**Task.** Pick up a pill bottle (one of 5 models, random position) and place it
upright on a blue 8 × 8 cm pad. RoboTwin counts an episode as a success when the
bottle is within 3 cm of the pad centre, upright, and the gripper is open.
Only successful expert episodes are included.

**Collection.** Expert trajectories are planned with CuRobo. Domain randomisation:
the wall and the table each get a random texture in 40 % of episodes
(plain otherwise), distractor objects are placed on the table in 40 % of
episodes, and lighting is random. {episodes} episodes, {frames} frames,
{tasks} distinct language instructions (one per episode).

**Features.**

| Key | Shape | Content |
| --- | --- | --- |
| `observation.images.head` | {head} | Head camera (D435 model), RGB video |
| `observation.images.wrist` | {wrist} | Wrist camera (D405 model), RGB video |
| `observation.state` | 7 | Joint positions joint1–joint6 (rad), gripper (0 closed … 1 open) |
| `action` | 7 | Absolute joint target: the state of the next frame |
| `task` | text | Language instruction |

**Timing.** Frames are saved every 32 ms of simulated time (31.25 Hz) and labelled
{fps} fps. Repeated frames at the joins of planned motions are dropped, so action
chunks advance at a steady rate.

**Gripper range.** The gripper is kept in RoboTwin's 0..1 range; the real Panthera-HT
gripper uses 0..2, so scale at the robot interface.

Conversion code: `scripts/robotwin/convert_to_lerobot.py` in the 6dofvla project.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="a converted dataset folder under data/")
    parser.add_argument("repo_id", help="<user>/<name> on the Hub")
    parser.add_argument("--public", action="store_true", help="create a public repository")
    parser.add_argument("--card-only", action="store_true", help="write README.md locally, push nothing")
    args = parser.parse_args()

    info = json.loads((args.dataset / "meta/info.json").read_text())
    features = info["features"]
    shape = lambda key: "×".join(map(str, features[key]["shape"]))  # noqa: E731
    card_kwargs = {
        "dataset_description": DESCRIPTION.format(
            episodes=info["total_episodes"],
            frames=info["total_frames"],
            tasks=info["total_tasks"],
            fps=info["fps"],
            head=shape("observation.images.head"),
            wrist=shape("observation.images.wrist"),
        ),
        "url": "https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin",
    }
    tags = ["panthera-ht", "robotwin", "simulation", "smolvla"]

    dataset = LeRobotDataset(args.repo_id, root=args.dataset)
    card = create_lerobot_dataset_card(
        tags=tags, dataset_info=dataset.meta.info, license="apache-2.0", repo_id=args.repo_id, **card_kwargs
    )
    card.save(args.dataset / "README.md")
    print(f"Card written to {args.dataset / 'README.md'}")
    if args.card_only:
        return
    dataset.push_to_hub(tags=tags, private=not args.public, **card_kwargs)
    print(f"Pushed: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
