"""Render a scripted rollout of every difficulty-level task (needs the GPU).

Writes one video per task and an overview image (five frames per task) to
outputs/levels_render_check/, and prints the step at which each task succeeded.
Use it to check that the objects and containers are visible to the cameras;
make_level_states.py checks the physics without rendering.
"""

from pathlib import Path

import imageio
import numpy as np
from lerobot_env_libero_three_items.config import LiberoLevelsEnv
from lerobot_env_libero_three_items.env import ThreeItemsLiberoEnv
from lerobot_env_libero_three_items.levels import TASKS
from lerobot_env_libero_three_items.scripted import ScriptedThreeItemsExpert

OUT = Path("outputs/levels_render_check")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = LiberoLevelsEnv()
    suite = cfg.make_suite()
    strips = []
    for task_id, task in enumerate(TASKS):
        env = ThreeItemsLiberoEnv(
            task_suite=suite,
            task_suite_name=cfg.task,
            task_id=task_id,
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_length=900,
        )
        env.reset(seed=0)
        expert = ScriptedThreeItemsExpert()
        expert.reset(env.sim_env)
        frames = [env.render()]
        result = "no success within 900 steps"
        for t in range(900):
            _, _, terminated, _, info = env.step(expert.act(env.sim_env))
            frames.append(env.render())
            if terminated:
                result = f"success at step {t}" if info["is_success"] else f"ended at step {t}"
                break
        env.close()
        imageio.mimsave(OUT / f"{task_id}_{task.name}.mp4", frames, fps=20)
        idx = np.linspace(0, len(frames) - 1, 5).astype(int)
        strips.append(np.concatenate([frames[i][::2, ::2] for i in idx], 1))
        print(f"task {task_id} {task.name}: {result}", flush=True)
    imageio.imwrite(OUT / "overview.png", np.concatenate(strips, 0))
    print(f"Videos and overview: {OUT}")


if __name__ == "__main__":
    main()
