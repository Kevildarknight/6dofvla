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

## Lost with the WSL install

Everything ignored by .gitignore was only on the WSL disk and is gone unless it was copied to D: first:

- data/three_items_demos (100 demos, about 1.7 h to recollect) and the DART variant;
- data/libero_basket_subset and data/three_items_mix (about 30 min to rebuild);
- outputs/, including the three_fixed_train_20k run and its checkpoint 040000 (most of a day on the RTX 3050 Ti);
- .venv, lerobot/, models/, .libero/ (recreated by scripts/install.sh).

Checkpoints were trained with `push_to_hub=false`, so none are on the Hugging Face Hub. From now on, push important checkpoints to the Hub or copy them off the machine.

## Machine

Laptop with an Intel i5-12450H and an RTX 3050 Ti Laptop GPU (4 GiB). It now dual-boots Windows and **Ubuntu 22.04**; WSL is no longer used. At 4 GiB, batch size 4 is the practical training limit.

Ubuntu 22.04 ships Python 3.10, but scripts/install.sh requires python3.12:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

## Moving to Panthera-HT

The arm is now the [Panthera-HT](https://github.com/HighTorque-Robotics/Panthera-HT_Main), a 6-DoF arm with HighTorque joint motors on CAN and a 1-DoF gripper. It replaces the SO-ARM 101. The main repository is only an index; the code is in these repositories (checked 21 Oct 2026):

| Repository | Use here | Environment |
| --- | --- | --- |
| [Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK) | Motor control, kinematics (Pinocchio), URDF, robot_param/Follower.yaml | Prebuilt `hightorque_robot` 1.2.0 wheels for Python 3.9–3.12 on x86-64, built on Ubuntu 22 |
| [Panthera-HT_lerobot](https://github.com/HighTorque-Robotics/Panthera-HT_lerobot) | `panthera_follower`, `panthera_leader`, `panthera_dual_follower` robots for lerobot-record and policy rollout | Python 3.10, its own vendored LeRobot **0.4.3**; expects Panthera-HT_SDK as a sibling directory |
| [Panthera-HT_RoboTwin](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin) | Simulation (SAPIEN, Vulkan), CuRobo planning, expert data collection; single-arm mode for 32 tasks (`<task>_panthera_single`) | Python 3.10.20, PyTorch 2.7.1 + CUDA 12.8, CuRobo 0.7.8 (needs nvcc 12.8); a separate Python 3.11 env for data conversion |
| [Panthera-HT-ROS2](https://github.com/HighTorque-Robotics/Panthera-HT-ROS2) | MoveIt, ros2_control, Gazebo Sim | ROS 2 Humble on Ubuntu 22.04 |

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

1. Install the Ubuntu 22.04 base: NVIDIA driver, Vulkan, python3.12, then scripts/install.sh and scripts/three_items/install.sh to confirm the LIBERO pipeline still works.
2. Install Panthera-HT_RoboTwin in its own conda env and collect one single-arm task (for example move_pillbottle_pad_panthera_single). Check the video and HDF5.
3. Write an HDF5 → LeRobot dataset converter for SmolVLA (single arm, 7-D, 30 fps, wrist and head cameras, task instruction).
4. Fine-tune SmolVLA on the converted data and evaluate in RoboTwin with policy/eval.py or a SmolVLA adapter.
5. On the real arm: port panthera_follower to LeRobot 0.6.1, set the camera serial numbers, record teleoperated demos with panthera_leader, and fine-tune on real data, optionally mixed with simulation data.

## Rules

- Keep .venv/, lerobot/, models/, data/, outputs/ and generated videos out of Git (.gitignore is a whitelist; add new tracked files to it).
- Do not edit vendor checkouts (lerobot/, RoboTwin, CuRobo); add plugins or wrappers instead.
- Report the actual evaluator result (eval_info.json, raw video) before claiming success.
