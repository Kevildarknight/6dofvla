"""Build a small subset of the original LIBERO training data to mix with the demos.

Fine-tuning only on three-butter demonstrations can make the policy forget the
LIBERO tasks it already solves. Mixing in some of the data the checkpoint was
trained on (HuggingFaceVLA/libero) limits that. This script keeps the basket
tasks, which are closest to the new task:

  single  "pick up the <object> and place it in the basket"   (10 LIBERO Object tasks)
  pair    "put both the <a> and the <b> in the basket"         (3 LIBERO-10 tasks)

It takes --single-per-task and --pair-per-task episodes of each, downloads only
the parquet files that hold them (the full dataset is 35 GB), and rewrites them
with the same video features and fps as the collected demos, so the two
datasets can be merged with lerobot-edit-dataset.

The dataset's own meta/episodes file lists stale file indices, so the
episode-to-file map is built by reading the episode_index column of every
remote file (column reads only, about a minute).
"""

import argparse
import io
import json
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from dataset_features import FEATURES, FPS, IMAGE_KEYS

REPO = "HuggingFaceVLA/libero"
REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"  # main on 2026-10-06
N_FILES = 377
SINGLE = re.compile(r"^pick up the .+ and place it in the basket$")
PAIR = re.compile(r"^put both the .+ and the .+ in the basket$")


def episode_file_map(cache: Path) -> dict[int, dict]:
    """episode_index -> {"task_index": int, "files": [file numbers]}."""
    path = cache / "episode_files.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}

    fs = HfFileSystem()

    def read(i):
        remote = f"datasets/{REPO}@{REVISION}/data/chunk-000/file-{i:03d}.parquet"
        with fs.open(remote, "rb") as f:
            table = pq.read_table(f, columns=["episode_index", "task_index"])
        return i, set(zip(table["episode_index"].to_pylist(), table["task_index"].to_pylist()))

    episodes: dict[int, dict] = {}
    with ThreadPoolExecutor(8) as pool:
        for file_number, pairs in pool.map(read, range(N_FILES)):
            for episode, task_index in pairs:
                entry = episodes.setdefault(episode, {"task_index": task_index, "files": []})
                entry["files"].append(file_number)
    path.write_text(json.dumps(episodes))
    return episodes


def task_names(cache: Path) -> dict[int, str]:
    tasks_file = hf_hub_download(REPO, "meta/tasks.parquet", repo_type="dataset", revision=REVISION, local_dir=cache)
    tasks = pd.read_parquet(tasks_file)
    return {int(row.task_index): name for name, row in tasks.iterrows()}


def select(episodes, names, single_per_task, pair_per_task):
    """Per task, take the episodes stored in the lowest-numbered files (fewest downloads)."""
    by_task = defaultdict(list)
    for episode, entry in episodes.items():
        by_task[entry["task_index"]].append(episode)
    chosen = {}
    for task_index, eps in sorted(by_task.items()):
        name = names[task_index]
        quota = single_per_task if SINGLE.match(name) else pair_per_task if PAIR.match(name) else 0
        eps.sort(key=lambda e: (min(episodes[e]["files"]), e))
        for episode in eps[:quota]:
            chosen[episode] = name
    return dict(sorted(chosen.items()))


def decode(cell) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/libero_basket_subset"))
    parser.add_argument("--repo-id", default="local/libero_basket_subset")
    parser.add_argument("--cache", type=Path, default=Path("data/hf_libero"))
    parser.add_argument("--single-per-task", type=int, default=10)
    parser.add_argument("--pair-per-task", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it.")
        shutil.rmtree(args.root)
    args.cache.mkdir(parents=True, exist_ok=True)

    episodes = episode_file_map(args.cache)
    names = task_names(args.cache)
    chosen = select(episodes, names, args.single_per_task, args.pair_per_task)
    files = sorted({f for e in chosen for f in episodes[e]["files"]})
    print(f"Selected {len(chosen)} episodes from {len(set(chosen.values()))} tasks in {len(files)} files", flush=True)

    local = {}
    for n, file_number in enumerate(files, 1):
        local[file_number] = hf_hub_download(
            REPO,
            f"data/chunk-000/file-{file_number:03d}.parquet",
            repo_type="dataset",
            revision=REVISION,
            local_dir=args.cache,
        )
        print(f"downloaded {n}/{len(files)}", flush=True)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=FEATURES, root=args.root, robot_type="panda", use_videos=True
    )
    columns = [*IMAGE_KEYS, "observation.state", "action", "episode_index", "frame_index"]
    for n, (episode, name) in enumerate(chosen.items(), 1):
        parts = [
            pq.read_table(local[f], columns=columns, filters=[("episode_index", "=", episode)]).to_pandas()
            for f in episodes[episode]["files"]
        ]
        frames = pd.concat(parts).sort_values("frame_index")
        for image, image2, state, action in zip(
            frames[IMAGE_KEYS[0]], frames[IMAGE_KEYS[1]], frames["observation.state"], frames["action"], strict=True
        ):
            dataset.add_frame(
                {
                    IMAGE_KEYS[0]: decode(image),
                    IMAGE_KEYS[1]: decode(image2),
                    "observation.state": np.asarray(state, dtype=np.float32),
                    "action": np.asarray(action, dtype=np.float32),
                    "task": name,
                }
            )
        dataset.save_episode()
        print(f"episode {n}/{len(chosen)}: source {episode}, {len(frames)} frames, '{name}'", flush=True)
    dataset.finalize()

    manifest = {"repo": REPO, "revision": REVISION, "episodes": {str(e): n for e, n in chosen.items()}}
    (args.root / "meta" / "source_episodes.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(chosen)} episodes, {dataset.meta.total_frames} frames to {args.root}")


if __name__ == "__main__":
    main()
