"""Convert RoboTwin single-arm Panthera episodes (HDF5) into the HDF5 layout of RoboTwin's ACT baseline.

ACT here is a check that the task can be learned from our data at all: if
RoboTwin's own ACT succeeds where SmolVLA fails, the problem lies in the SmolVLA
setup, not in the data. To keep the comparison fair, frames are chosen exactly
as convert_to_lerobot.py chooses them (repeats dropped, DART episodes split
around each push), so both policies see the same (state, action) pairs.

Output, one file per clean stretch: <out>/episode_<i>.hdf5 with
  action                          [T, 7]  joint_action/vector at the next kept frame
  observations/qpos               [T, 7]  joint_action/vector at frame t
  observations/images/cam_high    [T, H, W, 3] uint8 RGB, head camera
  observations/images/cam_wrist   [T, H, W, 3] uint8 RGB, wrist camera
and act_dataset.json (episode count, longest episode, camera names, image size).

RoboTwin's process_data.py resizes both cameras to 480x640. The head camera is
only 240x320, and ACT folds the cameras side by side, so both must share one
size; the default 240x320 keeps the head camera native and fits a 4 GiB GPU.
act_panthera.py resizes the live images the same way (cv2.INTER_AREA).

Runs in this repository's .venv (it reuses convert_to_lerobot.py's helpers).
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from convert_to_lerobot import clean_segments, decode, kept_frames, noise_frames, passed_episodes

CAMERAS = {"cam_high": "head_camera", "cam_wrist": "wrist_camera"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="RoboTwin folder data/<task_name>/<task_config>")
    parser.add_argument("--out", type=Path, default=None, help="default: data/act_<task_config>")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--min-segment", type=int, default=10, help="shortest clean stretch kept from a DART episode")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.out or Path("data") / f"act_{args.src.name}"
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    episodes = passed_episodes(args.src)
    if not episodes:
        raise SystemExit(f"No passed episodes in {args.src}/collection_manifest.json")
    size = (args.width, args.height)
    count, longest, frames = 0, 0, 0
    for episode in episodes:
        with h5py.File(args.src / "data" / f"episode{episode}.hdf5", "r") as f:
            vector = f["joint_action/vector"][()].astype(np.float32)
            if vector.shape[1] != 7:
                raise SystemExit(f"episode {episode}: joint vector has {vector.shape[1]} values, expected 7")
            noisy = noise_frames(args.src, episode)
            segments = clean_segments(kept_frames(vector, True), noisy)
            if noisy:
                segments = [s for s in segments if len(s) >= args.min_segment]
            rgb = {cam: f[f"observation/{source}/rgb"] for cam, source in CAMERAS.items()}
            for segment in segments:
                t, t_next = np.array(segment).T
                with h5py.File(out / f"episode_{count}.hdf5", "w") as g:
                    g["action"] = vector[t_next]
                    g["observations/qpos"] = vector[t]
                    for cam, data in rgb.items():
                        images = np.stack([cv2.resize(decode(data[i]), size, interpolation=cv2.INTER_AREA) for i in t])
                        g.create_dataset(
                            f"observations/images/{cam}",
                            data=images,
                            chunks=(1, *images.shape[1:]),
                            compression="lzf",
                        )
                count += 1
                longest = max(longest, len(t))
                frames += len(t)
        print(f"episode {episode}: {len(segments)} ACT episode(s)", flush=True)

    (out / "act_dataset.json").write_text(
        json.dumps(
            {
                "num_episodes": count,
                "episode_len": longest,
                "camera_names": list(CAMERAS),
                "image_size": [args.height, args.width],
                "source": str(args.src.resolve()),
            },
            indent=2,
        )
    )
    print(f"Saved {len(episodes)} RoboTwin episodes as {count} ACT episodes ({frames} frames) in {out}")


if __name__ == "__main__":
    main()
