# Project memory for Claude

This file replaces the chat history lost when the WSL install was unregistered (October 2026). It records what the repository does, what was lost, and the plan to move from the simulated Franka to the Panthera-HT arm. Read README.md and AGENTS.md for the original exercise.

## What the repository does

SmolVLA 0.45B, fine-tuned and evaluated in LIBERO (MuJoCo, robosuite) with a simulated **Franka Panda** arm.

| Part | Where | State |
| --- | --- | --- |
| Orange juice demo (LIBERO Object task 9) | scripts/install.sh, scripts/run_orange_juice.sh | Works with the published checkpoint |
| Three butters into the basket | plugins/lerobot_env_libero_three_items, scripts/three_items/ | Scene, success check, 40 train and 10 held-out layouts, scripted demos with DART noise, LIBERO data mix, train and eval scripts |
| Single-butter debug task (task 3) | branch three-items-task | Not merged |
| Difficulty levels: 10 tasks with generated cubes, cylinders, prisms, a wide tray and a narrow bin | branch sim-levels, plugins/.../LEVELS.md, scripts/levels/ | **Latest work (17 Oct), not merged into main.** Check out this branch to continue |
| `--append` for collect_demos.py | branches dxng, quan | Not merged |
| branch claude/eloquent-hawking-sjrf0j | scripts/triple_basket/ | First draft of the three-item task, superseded |

Pinned versions: LeRobot v0.6.1 (commit 7e241bd), Python 3.12, PyTorch 2.7.1, transformers 5.5.4, hf-libero 0.1.4, MuJoCo 3.8.1. Policy HuggingFaceVLA/smolvla_libero at revision 6721902. LIBERO interface: two 256×256 cameras (rotated 180° by the env), 8-D state, 7-D action (end-effector delta and gripper), 20 Hz.

## Lost with the WSL install, and what was saved

Everything ignored by .gitignore was only on the WSL disk. Three folders were copied to the Windows drive "New Volume" (D:) under smolvla_backup/ before WSL was removed:

| Backup folder | Restore to | Content |
| --- | --- | --- |
| 040000 | outputs/three_fixed_train_20k/checkpoints/040000 | Checkpoint 40,000 of the three_fixed_train_20k run |
| three_fixed_demos_dart | data/three_fixed_demos_dart | DART demos on the fixed layout |
| three_random_dart200 | data/three_random_dart200 | 200 DART demos on random layouts |

Lost, to be rebuilt if needed:

- data/three_items_demos (100 demos, about 1.7 h to recollect);
- data/libero_basket_subset and the mixed datasets (about 30 min to rebuild with prepare_libero_subset.sh and build_mix.sh);
- other runs under outputs/ and their logs;
- .venv, lerobot/, models/, .libero/ (recreated by scripts/install.sh).

All three were restored on the Ubuntu machine on 22 Oct 2026. From pretrained_model/train_config.json, checkpoint 040000 was trained on **data/three_fixed_demos_dart only** (50 episodes, 30,721 frames, fixed layout, no LIBERO mix), batch size 4, seed 1000: a 20k run resumed to 40,000 steps. Evaluate it with TASK_ID=0. data/three_random_dart200 has 200 episodes and 123,022 frames on random layouts and has not been trained on in a surviving run.

Checkpoints were trained with `push_to_hub=false`, so none are on the Hugging Face Hub. From now on, push important checkpoints to the Hub or copy them off the machine.

## Machine

Dell G15 5511 laptop with an Intel i5-12450H and an RTX 3050 Ti Laptop GPU (4 GiB). It now dual-boots Windows and **Ubuntu 24.04 LTS**; WSL is no longer used. At 4 GiB, batch size 4 is the practical training limit.

Ubuntu 24.04 is the system README.md was tested on, and it ships Python 3.12, which scripts/install.sh requires. Only the venv and build packages are missing on a fresh install:

```bash
sudo apt install -y python3.12-venv python3.12-dev git
```

## Moving to Panthera-HT

The arm is now the [Panthera-HT](https://github.com/HighTorque-Robotics/Panthera-HT_Main), a 6-DoF arm with HighTorque joint motors on CAN and a 1-DoF gripper. It replaces the SO-ARM 101. The main repository is only an index; the code is in these repositories (checked 21 Oct 2026):

| Repository | Use here | Environment |
| --- | --- | --- |
| [Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK) | Motor control, kinematics (Pinocchio), URDF, robot_param/Follower.yaml | Prebuilt `hightorque_robot` 1.2.0 wheels for Python 3.9–3.12 on x86-64, built on Ubuntu 22; if the wheel fails on 24.04, build from source as its README describes |
| [Panthera-HT_lerobot](https://github.com/HighTorque-Robotics/Panthera-HT_lerobot) | `panthera_follower`, `panthera_leader`, `panthera_dual_follower` robots for lerobot-record and policy rollout | Python 3.10, its own vendored LeRobot **0.4.3**; expects Panthera-HT_SDK as a sibling directory |
| [Panthera-HT_RoboTwin](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin) | Simulation (SAPIEN, Vulkan), CuRobo planning, expert data collection; single-arm mode for 32 tasks (`<task>_panthera_single`) | Python 3.10.20, PyTorch 2.7.1 + CUDA 12.8, CuRobo 0.7.8 (needs nvcc 12.8); a separate Python 3.11 env for data conversion |
| [Panthera-HT-ROS2](https://github.com/HighTorque-Robotics/Panthera-HT-ROS2) | MoveIt, ros2_control, Gazebo Sim, direct SDK control node | Branches humble (Ubuntu 22.04), **jazzy (Ubuntu 24.04, use this one)** and Lyrical. Clone with `-b jazzy`; the default branch is humble |

### Panthera interface

| Item | Value |
| --- | --- |
| Joints | joint1–joint6 plus gripper |
| Joint limits (rad) | lower [-2.4, -0.1, -0.1, -1.6, -1.7, -2.5], upper [2.4, 3.2, 4.0, 1.6, 1.7, 2.5] |
| Gripper range | 0.0 to 2.0 |
| LeRobot state and action | 7 absolute joint positions, keys `joint1.pos` … `joint6.pos`, `gripper.pos` |
| Real cameras | RealSense D405 on the wrist, D435/D435i top or side view, 640×480 at 30 fps; serial numbers are hard-coded in panthera_follower.py and must be changed for our cameras |
| RoboTwin single arm | 7-D action (6 joints and gripper), head and wrist cameras, HDF5 output |
| RoboTwin converters | ACT, Pi0, Pi0.5, GO1, RDT, DP, DP3 and others. **No SmolVLA converter**; the Pi0/Pi0.5 LeRobot writer is the closest starting point. Its examples are dual-arm (14-D); single-arm 7-D data has to be checked separately |

### What changes compared with the LIBERO work

- **State and action:** absolute joint positions (7-D) instead of end-effector deltas (7-D) with an 8-D state. The normalisation statistics and action expert must be retrained. Start from lerobot/smolvla_base rather than the LIBERO checkpoint, or test both.
- **Rate:** 30 fps in RoboTwin and on the real arm, versus 20 Hz in LIBERO.
- **Simulator:** SAPIEN in RoboTwin instead of MuJoCo in LIBERO. The LIBERO plugin, scripted controller and success check cannot be reused directly. Their ideas carry over: fixed/train/held-out layouts, checks before training, DART noise, mixing in original data, and difficulty levels.
- **LeRobot versions:** this repository uses LeRobot 0.6.1 with Python 3.12, while Panthera-HT_lerobot vendors LeRobot 0.4.3 with Python 3.10. The SDK wheel exists for Python 3.12, so porting the panthera_follower robot to LeRobot 0.6.1 as a plugin (like the three-items env plugin) is likely easier than downgrading. Check dataset format compatibility between the two versions before mixing data.

### Plan

1. Install the Ubuntu 24.04 base: NVIDIA driver, Vulkan, python3.12-venv, then scripts/install.sh and scripts/three_items/install.sh to confirm the LIBERO pipeline still works.
2. Install Panthera-HT_RoboTwin in its own conda env and collect one single-arm task (for example move_pillbottle_pad_panthera_single). Check the video and HDF5.
3. Write an HDF5 → LeRobot dataset converter for SmolVLA (single arm, 7-D, 30 fps, wrist and head cameras, task instruction).
4. Fine-tune SmolVLA on the converted data and evaluate in RoboTwin with policy/eval.py or a SmolVLA adapter.
5. On the real arm: port panthera_follower to LeRobot 0.6.1, set the camera serial numbers, record teleoperated demos with panthera_leader, and fine-tune on real data, optionally mixed with simulation data.

## Rules

- Keep .venv/, lerobot/, models/, data/, outputs/ and generated videos out of Git (.gitignore is a whitelist; add new tracked files to it).
- Do not edit vendor checkouts (lerobot/, RoboTwin, CuRobo); add plugins or wrappers instead.
- Report the actual evaluator result (eval_info.json, raw video) before claiming success.
