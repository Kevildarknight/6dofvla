# Instructions for Codex agents

This repository demonstrates a fine-tuned SmolVLA 0.45B policy controlling a Franka robot in the LIBERO Object simulator. The primary exercise is task 9: pick up the orange juice carton and place it in the basket. Run the **full episode** and verify the evaluator's success flag; loading the model or predicting one action is not a completed demo.

## Reproduce the student exercise

1. Read README.md and check that the machine has Linux, Python 3.12, Git, a CUDA-capable NVIDIA GPU, a working driver, and free disk space.
2. From the repository root, run bash scripts/install.sh. This creates .venv, clones LeRobot v0.6.1, installs pinned dependencies, and downloads the fine-tuned checkpoint and LIBERO assets. Use the project virtual environment for all Python commands.
3. Run bash scripts/run_orange_juice.sh. The script chooses a fresh output directory unless one is supplied.
4. Read the eval_info.json file in the selected output directory. Confirm that the entry for task 9 has metrics.successes[0] == true. Open the videos/libero_object_9/eval_episode_0.mp4 file in that directory to observe the robot's raw trajectory. Report the actual result of the current run, even if it differs from the included successful example.

The published video is [demos/main_orange_juice.mp4](demos/main_orange_juice.mp4). It is a slowed and captioned presentation of a successful run, not a substitute for checking the evaluator's JSON output. A useful student prompt is: “Follow AGENTS.md, run the LIBERO Object orange juice episode, tell me whether the evaluator marked success, and show me where to find the raw video.”

## Repository boundaries

Keep .venv/, .libero/, lerobot/, models/, outputs/, downloaded assets, local papers, and all generated videos out of Git. The **only tracked MP4** is demos/main_orange_juice.mp4. Do not add other video files to the repository. Do not edit the vendor checkout in lerobot/ for this exercise. If a run fails, inspect its logs and JSON before changing the pinned setup or claiming success.

The pinned upstream LeRobot commit is 7e241bd630a3719a56157a497ce5d08f244784f1; the policy revision is 6721902bc4d61e50a3bfdb11dfb4cb626f05d102. Evaluation uses LIBERO Object, task ID 9, seed 1000, batch size 1, one episode, and EGL rendering. The included result is one successful episode, not a benchmark success rate.
