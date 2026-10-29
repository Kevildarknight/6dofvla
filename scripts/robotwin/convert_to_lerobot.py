"""Convert RoboTwin single-arm Panthera episodes (HDF5) into a LeRobot dataset for SmolVLA.

Input is one RoboTwin output folder, data/<task_name>/<task_config>/, holding
data/episodeN.hdf5, instructions/episodeN.json and collection_manifest.json.
Only episodes the manifest marks as passed are converted.

Per frame:
  observation.state         joint_action/vector[t]: joint1..joint6 (rad), gripper (0 closed .. 1 open)
  action                    joint_action/vector[t+1]: the next absolute joint target
  observation.images.head   head_camera rgb (D435 view)
  observation.images.wrist  wrist_camera rgb (D405 view)
  task                      one of the episode's "seen" instructions

Action = next state is the convention RoboTwin's own converters use
("action_alignment": "next_state"); the last frame has no next state and is dropped.

RoboTwin saves a frame every save_freq physics steps, but restarts the count at
each planned motion and also saves at both ends of it, so the last frame of one
motion and the first of the next show the same instant. With
--drop-duplicates (on by default), a frame whose joint vector equals the
previous kept one is skipped, which removes those repeats and the still frames
of fixed delays. Without it, about one frame in six is a repeat.

Episodes collected with dart_collect.py (DART=1 collect.sh) list their
perturbation frames in dart/episodeN.json. A transition whose next frame lies
inside a perturbation is dropped, since its action would teach the push itself.
The episode is split there, so each clean stretch becomes its own LeRobot
episode and action chunks never span a push; stretches shorter than
--min-segment frames are left out. The stretch after a push starts at the
pushed state and shows the expert's recovery, which is what DART is for.

The gripper stays in RoboTwin's 0..1 range. The real arm uses 0..2 (see
CLAUDE.md); scale at the robot interface, not here, so sim and real data share
one convention.

The --fps label is metadata: SmolVLA's action chunks are indexed by frame, so
it sets video playback speed and timestamp spacing, not what the policy learns.
With save_freq 8 the frames are 32 ms apart (31.25 Hz), labelled 30.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERAS = {"observation.images.head": "head_camera", "observation.images.wrist": "wrist_camera"}
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def decode(jpeg_bytes):
    # RoboTwin passes the renderer's RGB array to cv2.imencode, which treats it
    # as BGR; cv2.imdecode undoes that, so the result is already RGB. Swapping
    # channels here would turn red objects blue.
    return cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)


def passed_episodes(src):
    manifest = json.loads((src / "collection_manifest.json").read_text())
    return sorted(int(k) for k, v in manifest["episodes"].items() if v["hdf5"]["passed"])


def features_for(first_hdf5):
    with h5py.File(first_hdf5, "r") as f:
        shapes = {key: decode(f[f"observation/{cam}/rgb"][0]).shape for key, cam in CAMERAS.items()}
        dim = f["joint_action/vector"].shape[1]
    if dim != len(JOINT_NAMES):
        raise SystemExit(f"{first_hdf5}: joint vector has {dim} values, expected 7 (single-arm Panthera)")
    return {
        **{
            key: {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]}
            for key, shape in shapes.items()
        },
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
    }


def kept_frames(vector, drop_duplicates):
    if not drop_duplicates:
        return list(range(len(vector)))
    keep = [0]
    for t in range(1, len(vector)):
        if np.abs(vector[t] - vector[keep[-1]]).max() > 1e-6:
            keep.append(t)
    return keep


def noise_frames(src, episode):
    path = src / "dart" / f"episode{episode}.json"
    if not path.exists():
        return set()
    spans = json.loads(path.read_text())["noise_frames"]
    # The first frame of a span is the state before the push; the rest are reached by it.
    return {t for first, last in spans for t in range(first + 1, last + 1)}


def clean_segments(keep, noisy):
    """Split consecutive (t, t_next) pairs wherever a pair would lead into a noisy frame."""
    segments, current = [], []
    for t, t_next in zip(keep[:-1], keep[1:]):
        if any(i in noisy for i in range(t + 1, t_next + 1)):
            if current:
                segments.append(current)
            current = []
        else:
            current.append((t, t_next))
    if current:
        segments.append(current)
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="RoboTwin folder data/<task_name>/<task_config>")
    parser.add_argument("--root", type=Path, default=None, help="default: data/robotwin_<task_config>")
    parser.add_argument("--repo-id", default=None, help="default: local/robotwin_<task_config>")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--keep-duplicates", dest="drop_duplicates", action="store_false")
    parser.add_argument("--seed", type=int, default=0, help="picks among several seen instructions")
    parser.add_argument("--min-segment", type=int, default=10, help="shortest clean stretch kept from a DART episode")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    name = f"robotwin_{args.src.name}"
    root = args.root or Path("data") / name
    repo_id = args.repo_id or f"local/{name}"
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it.")
        shutil.rmtree(root)

    episodes = passed_episodes(args.src)
    if not episodes:
        raise SystemExit(f"No passed episodes in {args.src}/collection_manifest.json")
    features = features_for(args.src / "data" / f"episode{episodes[0]}.hdf5")
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=args.fps, features=features, root=root, robot_type="panthera", use_videos=True
    )
    rng = np.random.default_rng(args.seed)

    dropped_total = 0
    for episode in episodes:
        instructions = json.loads((args.src / "instructions" / f"episode{episode}.json").read_text())
        task = str(rng.choice(instructions["seen"]))
        with h5py.File(args.src / "data" / f"episode{episode}.hdf5", "r") as f:
            vector = f["joint_action/vector"][:].astype(np.float32)
            keep = kept_frames(vector, args.drop_duplicates)
            images = {key: f[f"observation/{cam}/rgb"] for key, cam in CAMERAS.items()}
            noisy = noise_frames(args.src, episode)
            segments = clean_segments(keep, noisy)
            if noisy:
                segments = [segment for segment in segments if len(segment) >= args.min_segment]
            for segment in segments:
                for t, t_next in segment:
                    frame = {key: decode(images[key][t]) for key in CAMERAS}
                    dataset.add_frame(
                        {**frame, "observation.state": vector[t], "action": vector[t_next], "task": task}
                    )
                dataset.save_episode()
        dropped_total += len(vector) - len(keep)
        kept = sum(len(segment) for segment in segments)
        dart_note = f", {len(segments)} DART segments" if noisy else ""
        print(
            f"episode {episode}: {kept} frames ({len(vector) - len(keep)} repeats dropped{dart_note}), "
            f"task '{task}'",
            flush=True,
        )

    dataset.finalize()
    print(
        f"Saved {len(episodes)} RoboTwin episodes as {dataset.meta.total_episodes} LeRobot episodes, "
        f"{dataset.meta.total_frames} frames "
        f"({dropped_total} repeats dropped) to {root}"
    )


if __name__ == "__main__":
    main()
