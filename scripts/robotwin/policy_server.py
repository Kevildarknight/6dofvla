"""Serve a SmolVLA checkpoint to RoboTwin's evaluator over a local socket.

RoboTwin runs in its own Python 3.10 env, while this repository's LeRobot 0.6.1
needs Python 3.12, so the policy and the simulator run in separate processes.
smolvla_panthera.py (loaded by RoboTwin's script/eval_policy.py) connects here.

Each request is a dict of plain Python values, so neither side has to unpickle
the other's numpy version:
  {"head": (bytes, shape), "wrist": (bytes, shape), "state": [7 floats], "task": str}
Images are uint8 RGB, height x width x 3, as RoboTwin renders them. The reply is
{"actions": [[7 floats], ...]}: the first --action-steps actions of the
predicted chunk, unnormalised, in the dataset's units (joint radians, gripper
0..1). The checkpoint's preprocessor renames head and wrist to camera1 and
camera2, as in training.
"""

import argparse
from multiprocessing.connection import Listener

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

AUTHKEY = b"robotwin-smolvla"


def to_image(packed):
    data, shape = packed
    image = np.frombuffer(data, np.uint8).reshape(shape)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a pretrained_model folder")
    parser.add_argument("--port", type=int, default=6010)
    parser.add_argument("--action-steps", type=int, default=10, help="actions returned per request")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.config.device = args.device
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    if args.action_steps > policy.config.chunk_size:
        raise SystemExit(f"--action-steps must be at most the chunk size, {policy.config.chunk_size}")

    with Listener(("localhost", args.port), authkey=AUTHKEY) as listener:
        print(f"SmolVLA server ready on port {args.port}", flush=True)
        while True:
            with listener.accept() as connection:
                print("evaluator connected", flush=True)
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        print("evaluator disconnected", flush=True)
                        break
                    frame = {
                        "observation.images.head": to_image(request["head"]),
                        "observation.images.wrist": to_image(request["wrist"]),
                        "observation.state": torch.tensor(request["state"], dtype=torch.float32),
                        "task": request["task"],
                    }
                    with torch.inference_mode():
                        chunk = policy.predict_action_chunk(preprocessor(frame))
                        actions = postprocessor(chunk)[0, : args.action_steps]
                    connection.send({"actions": actions.cpu().numpy().astype(float).tolist()})


if __name__ == "__main__":
    main()
