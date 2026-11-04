"""Train RoboTwin's ACT baseline on a dataset written by act_convert.py.

Runs RoboTwin's policy/ACT/imitate_episodes.py unchanged. That script looks the
task up in policy/ACT/SIM_TASK_CONFIGS.json, which RoboTwin's process_data.py
rewrites; instead of editing the vendor file, the entry is added in memory
before imitate_episodes.py starts. The state is 7-D (single arm).

Usage (RoboTwin conda env, from RoboTwin's root folder; act_train.sh does this):
  python act_train.py <dataset dir> <checkpoint dir> [imitate_episodes.py args...]
"""

import json
import runpy
import shutil
import sys
from pathlib import Path

act_dir = Path("policy/ACT").resolve()
sys.path.insert(0, str(act_dir))
import constants  # noqa: E402  (RoboTwin's policy/ACT/constants.py)

dataset, ckpt_dir, *extra = sys.argv[1:]
dataset = Path(dataset).resolve()
info = json.loads((dataset / "act_dataset.json").read_text())
task_name = f"sim-panthera-{dataset.name}"
constants.SIM_TASK_CONFIGS[task_name] = {
    "dataset_dir": str(dataset),
    "num_episodes": info["num_episodes"],
    "episode_len": info["episode_len"],
    "camera_names": info["camera_names"],
}
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
shutil.copy(dataset / "act_dataset.json", ckpt_dir)  # act_panthera.py reads the image size from it

sys.argv = [
    str(act_dir / "imitate_episodes.py"),
    "--task_name", task_name,
    "--ckpt_dir", str(Path(ckpt_dir).resolve()),
    "--policy_class", "ACT",
    "--state_dim", "7",
    *extra,
]
runpy.run_path(sys.argv[0], run_name="__main__")
