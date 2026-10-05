"""Record scripted demos (images, state, action) to outputs/triple_basket_demos/ep_XXXX.npz.

Usage: .venv/bin/python scripts/triple_basket/collect_demos.py [n_episodes]
Only successful episodes are kept. Convert to a LeRobotDataset for fine-tuning (see README).
"""
import sys

import numpy as np
from common import REPO, fixed_reset, make_env, tb
from scripted_expert import run_episode

n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
out = REPO / "outputs" / "triple_basket_demos"
out.mkdir(parents=True, exist_ok=True)
env = make_env()
kept = 0
for ep in range(n):
    fixed_reset(env)
    rec = {"agentview": [], "wrist": [], "state": [], "action": []}

    def on_step(obs, a):
        rec["agentview"].append(obs["agentview_image"][::-1])
        rec["wrist"].append(obs["robot0_eye_in_hand_image"][::-1])
        rec["state"].append(np.concatenate([obs["robot0_eef_pos"], obs["robot0_gripper_qpos"]]))
        rec["action"].append(a)

    success = run_episode(env, on_step)
    print(f"episode {ep}: success={success} steps={len(rec['action'])}")
    if success:
        np.savez_compressed(out / f"ep_{kept:04d}.npz", task="put all three items in the basket",
                            **{k: np.asarray(v) for k, v in rec.items()})
        kept += 1
print(f"kept {kept}/{n} demos in {out}")
