# SmolVLA in LIBERO: Orange Juice Demo

This repository reproduces a **closed-loop robot rollout**, not a model-loading smoke test. A fine-tuned SmolVLA 0.45B policy observes the simulated Franka robot and predicts actions until it picks up the orange juice carton and places it in the basket. LIBERO reports success for the recorded episode.

**[Watch the main orange juice demo (MP4)](demos/main_orange_juice.mp4)**

The video is an 8.1-second presentation cut of the successful rollout. It is slowed down, resized, and captioned; the robot trajectory and the evaluator's success flag were not changed. This is the **only video included in this repository**. The evaluator generates a fresh raw video when you run the demo locally.

| Run | Value |
| --- | --- |
| Benchmark | LIBERO Object, task ID 9 |
| Instruction | “pick up the orange juice and place it in the basket” |
| Policy | [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero), pinned to revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 |
| Evaluation | One full episode, seed 1000, batch size 1; successes: [true] |
| Tested hardware | Ubuntu 24.04.4, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM |

## Install

You need Linux, Python 3.12 with venv, Git, an NVIDIA driver and CUDA-capable GPU, and enough disk space for PyTorch, LeRobot, the checkpoint, and LIBERO assets. The validated machine used driver 580.173.02, PyTorch 2.7.1+cu126, MuJoCo 3.8.1, and robosuite 1.4.0. A separate CUDA Toolkit installation was not needed for the tested PyTorch wheel.

~~~bash
git clone https://github.com/Huytai1o2/smolVLA.git
cd smolVLA
bash scripts/install.sh
~~~

The installer creates .venv, clones [LeRobot](https://github.com/huggingface/lerobot) at tag v0.6.1 (commit 7e241bd630a3719a56157a497ce5d08f244784f1), installs its SmolVLA and LIBERO dependencies, and downloads the **fine-tuned** policy with its processor and normalization statistics. It also downloads the LIBERO object assets. These large files are local only and are ignored by Git. No training dataset is required to replay this evaluation.

## Run the full episode

~~~bash
bash scripts/run_orange_juice.sh
~~~

The script runs lerobot-eval with EGL rendering, LIBERO Object task 9, one episode, and seed 1000. It prints the evaluator's success flag and the paths to the result and raw rollout video. To choose an output directory, pass it as the first argument:

~~~bash
bash scripts/run_orange_juice.sh outputs/my_orange_juice_run
~~~

For manual inspection, open the eval_info.json file inside the selected output directory and check per_task[0].metrics.successes[0]. The raw video is under the videos/libero_object_9/eval_episode_0.mp4 file in that directory. Simulation outcomes may differ across GPUs, drivers, or dependency versions.

## What was tested

We ran one complete episode for each of the ten LIBERO Object tasks at seed 1000. Tasks 0, 1, 2, 3, 5, 6, 8, and 9 succeeded; tasks 4 and 7 failed. **Eight successes out of ten tasks is a demo-selection result, not a statistically meaningful benchmark success rate.** We selected task 9 because the orange carton and pick-and-place motion are easy to follow in a short presentation. To try another task, change the task ID argument in scripts/run_orange_juice.sh.

The policy and benchmark come from [SmolVLA](https://arxiv.org/abs/2506.01844), [LeRobot's LIBERO documentation](https://huggingface.co/docs/lerobot/libero), and the [published fine-tuned checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero). This project does not include the original model weights or vendor source code in Git.

Students using Codex can open the cloned repository and ask it to follow [AGENTS.md](AGENTS.md) to install and run the orange juice episode, then inspect the evaluator result and video.
