# RoboTwin data for SmolVLA on Panthera-HT

These scripts collect simulated demonstrations of the Panthera-HT arm in [Panthera-HT_RoboTwin](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin) and convert them into a LeRobot dataset that SmolVLA can be fine-tuned on. Two repositories and two Python environments are involved:

| Step | Runs in | Environment |
| --- | --- | --- |
| Collect demonstrations | Panthera-HT_RoboTwin | conda env `RoboTwin` (Python 3.10, SAPIEN, CuRobo) |
| Convert to LeRobot | this repository | `.venv` (Python 3.12, LeRobot 0.6.1), created by scripts/install.sh |

Tested on Ubuntu 24.04 with an RTX 3050 Ti Laptop GPU (4 GiB).

## 1. Clone

```bash
sudo apt install -y git git-lfs unzip python3.12-venv python3.12-dev
git lfs install

git clone -b robotwin-data https://github.com/Kevildarknight/6dofvla.git ~/6dofvla
git clone https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin.git ~/panthera/Panthera-HT_RoboTwin
```

**Install git-lfs before cloning Panthera-HT_RoboTwin.** The Panthera URDF and meshes are stored in Git LFS. Without it they are 130-byte pointer files and SAPIEN cannot load the robot. To check:

```bash
head -c 60 ~/panthera/Panthera-HT_RoboTwin/RoboTwin/assets/embodiments/panthera-6dof/panthera_6dof.urdf
# must print <?xml ...; if it prints "version https://git-lfs...", run: git lfs pull
```

collect.sh looks for RoboTwin at `~/panthera/Panthera-HT_RoboTwin/RoboTwin`. If you clone it elsewhere, set `ROBOTWIN_DIR` to its inner `RoboTwin/` folder.

## 2. Set up the `RoboTwin` environment

Follow sections 1.1, 1.2 and 2 of the [Panthera-HT_RoboTwin README](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin#1-环境配置), from the inner `RoboTwin/` folder. In short:

1. Create a conda env `RoboTwin` with Python 3.10.20, then install FFmpeg 7 and PyTorch 2.7.1 + CUDA 12.8.
2. Make sure `nvcc --version` reports 12.8, then install the pinned pip packages and CuRobo v0.7.8 (editable, in `envs/curobo`).
3. Apply the two `sed` patches to SAPIEN and MPlib.
4. Download the generic assets with `bash script/_download_assets.sh`, which fetches about 16 GB.

Do not run the upstream `script/_install.sh`. The README says it installs versions that differ from the tested ones.

**If the asset download prints nothing for a long time,** check whether IPv6 works on your network. If IPv6 is broken, every Python connection to Hugging Face waits about 40 s before falling back to IPv4. Either make the system prefer IPv4:

```bash
echo 'precedence ::ffff:0:0/96  100' | sudo tee -a /etc/gai.conf
```

or download the three zips with `wget -4 -c https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/<name>.zip` into `assets/`, then unzip them and run `python script/update_embodiment_config_path.py`.

Check the environment:

```bash
conda activate RoboTwin
python -c "import torch, sapien, curobo; print(torch.__version__, torch.cuda.is_available(), sapien.__version__)"
```

## 3. Set up this repository

```bash
cd ~/6dofvla
bash scripts/install.sh
```

This creates `.venv` with LeRobot 0.6.1, which the converter uses. The conda env is not needed for conversion.

## 4. Collect demonstrations

```bash
cd ~/6dofvla
bash scripts/robotwin/collect.sh                      # 50 episodes of move_pillbottle_pad, single arm
bash scripts/robotwin/collect.sh move_pillbottle_pad move_pillbottle_pad_panthera_single_30hz 5   # 5 episodes
```

The script copies the task config from this folder into RoboTwin's `task_config/` and runs RoboTwin's collector in the `RoboTwin` env. The copy is ignored by RoboTwin's Git and is overwritten on every run, so edit the config here. Episodes whose planning fails are skipped, so the saved seeds have gaps; this is expected.

Output: `$ROBOTWIN_DIR/data/move_pillbottle_pad/move_pillbottle_pad_panthera_single_30hz/`, with `data/episodeN.hdf5`, videos under `video/` and `collection_manifest.json`. Watch a few head-camera videos to check that the bottle really ends up on the pad.

## 5. Convert to a LeRobot dataset

```bash
cd ~/6dofvla
bash scripts/robotwin/convert.sh \
  ~/panthera/Panthera-HT_RoboTwin/RoboTwin/data/move_pillbottle_pad/move_pillbottle_pad_panthera_single_30hz
```

Give the source as an absolute path, because convert.sh runs from the repository root. The dataset is written to `data/robotwin_<config name>/`; pass `--overwrite` to rebuild it and `--help` for the other options.

| Feature | Content |
| --- | --- |
| `observation.state` | joint1–joint6 (rad), gripper (0 closed, 1 open), at frame t |
| `action` | the same 7 values at the next kept frame (absolute joint target) |
| `observation.images.head` | head camera, 240×320 |
| `observation.images.wrist` | wrist camera, 360×640 |
| `task` | one of the episode's "seen" instructions |

## Things to know

- **Frame rate.** RoboTwin's planner fixes the physics step at 1/250 s, so `save_freq` must be a whole number of steps. `save_freq: 8` saves a frame every 32 ms (31.25 Hz); the dataset is labelled 30 fps. SmolVLA indexes action chunks by frame, so the label does not change what the policy learns.
- **Repeated frames.** RoboTwin saves a frame at both ends of every planned motion, so about one frame in six repeats the one before. The converter drops frames whose joint vector equals the previous one; `--keep-duplicates` keeps them.
- **Gripper scale.** The simulator's gripper is 0..1; the real arm uses 0..2. Data stays 0..1, and scaling belongs in the real-robot interface.
- **Do not commit to Panthera-HT_RoboTwin.** After `git lfs pull` and the asset script, it shows modified config and mesh files; leave them. New code goes in this repository.
