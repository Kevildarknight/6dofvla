# Custom LIBERO task: three butters into the basket

Instruction given to the policy: “pick up the three butters one by one and place them all in the basket”.

This is a new LIBERO task for this repository, packaged as a LeRobot environment plugin. It registers `--env.type=libero_three_items` for lerobot-eval and lerobot-train without editing the vendor checkout in lerobot/.

## 1. Task specification

| Item | Choice |
| --- | --- |
| Scene | LIBERO Object floor arena, the same scene as the orange juice demo |
| Objects | Three identical butters with separate IDs: `item_1`, `item_2`, `item_3` |
| Container | `basket_1`, the LIBERO Object basket (inner box about 12 × 12 × 14 cm) |
| Fixed layout (world x, y in metres) | item_1 (−0.10, −0.18), item_2 (0.06, −0.10), item_3 (−0.15, 0.03), basket (0.00, 0.25) |
| Order | item_1, then item_2, then item_3 (scripted demonstrations follow this order) |
| Episode limit | 900 control steps at 20 Hz (45 s) |

Butter was chosen because three butters fit side by side in the basket. A butter lies flat with its 4 cm side along the axis the gripper closes on, so a straight top-down grasp works. The items are at least 0.17 m apart, so the open gripper never touches a neighbouring item.

## 2. Scene and initial state

| File | Role |
| --- | --- |
| [three_butters_basket.bddl](lerobot_env_libero_three_items/assets/three_butters_basket.bddl) | Declares the arena, the basket, the three item IDs, their placement regions, and LIBERO's goal |
| [three_butters_basket.pruned_init](lerobot_env_libero_three_items/assets/three_butters_basket.pruned_init) | One saved MuJoCo state; every reset restores it |
| [task.py](lerobot_env_libero_three_items/task.py) | Layout, item IDs, instruction, and a one-task suite in the shape LeRobot's LiberoEnv reads |
| [config.py](lerobot_env_libero_three_items/config.py) | Registers `libero_three_items` as a LeRobot env type |

LIBERO's placement sampler moves objects by 1–2 cm between resets, even inside small regions. To make the layout fixed, [make_init_state.py](../../scripts/three_items/make_init_state.py) places each body at `LAYOUT_XY`, lets the scene settle, and saves the state. After changing the layout, regenerate it with `bash scripts/three_items/make_init_state.sh`.

In LIBERO's BDDL parser, the three IDs must be declared on one line (`item_1 item_2 item_3 - butter`). With one line per item, only the last item is created.

## 3. Success criterion

[success.py](lerobot_env_libero_three_items/success.py) replaces LIBERO's goal check. At each control step, an item counts as in the basket only when all of these are true:

- **Inside:** its centre is inside the basket's `contain_region` box, measured in the basket frame, with 5 mm trimmed from each side so an item lying on the rim does not count.
- **Released:** it is not touching the gripper.
- **Settled:** it moved less than 2 cm/s since the previous step.

```
success = item_1 in basket AND item_2 in basket AND item_3 in basket,
          for 10 consecutive control steps (0.5 s)
```

[env.py](lerobot_env_libero_three_items/env.py) sets this result as `info["is_success"]` and ends the episode on it. lerobot-eval then writes it to eval_info.json. LIBERO's own `done` flag is ignored because it is only the BDDL goal, which becomes true while the last butter is still falling. The env also reports `num_items_in_basket` and `item_k_in_basket` in `info`.

## 4. Scene validation (before any training)

```bash
bash scripts/three_items/install.sh       # once, after scripts/install.sh
bash scripts/three_items/check_scene.sh   # writes outputs/three_items_check_*/report.json
```

[check_scene.py](../../scripts/three_items/check_scene.py) uses the scripted controller in [scripted.py](lerobot_env_libero_three_items/scripted.py) and teleports objects. It does not use a learned policy. Result on the tested machine:

| Check | What is verified | Result |
| --- | --- | --- |
| reset_layout | 5 resets: every body within 0.03 mm of its layout position; identical robot joints | PASS |
| reachability_and_scripted_full | Scripted run lifts each butter to 0.21 m; they enter the basket at steps 156, 339, 516; success at step 525, after the third butter | PASS |
| per_item | One butter in the basket flags only that butter; a butter beside the basket or still falling is not counted | PASS |
| all_required | One or two butters in the basket: no success. All three: success after 10 steps | PASS |

The scripted rollout video is saved next to the report as scripted_episode.mp4.

## 5. Demonstrations and fine-tuning

```bash
bash scripts/three_items/collect_demos.sh --episodes 50   # data/three_items_demos (LeRobot dataset)
bash scripts/three_items/train.sh                         # outputs/three_items_train_*/
```

Each demonstration repeats the same cycle for item_1, item_2 and item_3: move above the butter, descend, close, lift, carry over the basket, open, then rise. Waypoints get up to ±1 cm of random noise per episode, and only episodes that pass the success check are saved. Observations go through the same preprocessing as lerobot-eval, so the stored images (rotated 180°) and the 8-D state match what the checkpoint receives during evaluation.

train.sh fine-tunes models/smolvla_libero with the checkpoint's defaults: the vision-language backbone is frozen, and the action expert and state projection (97 M parameters) are trained. The settings can be changed with the BATCH_SIZE, STEPS and SAVE_FREQ environment variables.

Measured on the RTX 3050 Ti Laptop GPU (4 GiB) in a 20-step smoke test at batch size 4:

| | Observation |
| --- | --- |
| PyTorch memory reported by lerobot-train | 2.74 GB |
| Total GPU memory used (nvidia-smi, including other processes) | 3,862 MiB of 4,096 MiB |
| Time per step | 1.3–4 s |
| Demonstration collection | about 2.4 min per episode, including video encoding |

The default 20,000 steps would take most of a day on this GPU. Use a larger GPU, or run fewer steps (for example `STEPS=5000`) and check the result. At 4 GiB, batch size 4 is the practical limit.

## 6. Evaluation

```bash
bash scripts/three_items/eval.sh outputs/three_items_train_<run>/checkpoints/last/pretrained_model 10
```

The script runs lerobot-eval with `--env.type=libero_three_items` and prints `successes[i]` for each episode. Check both outputs:

- eval_info.json, at `per_task[0].metrics.successes` for task group `libero_three_items`, task 0.
- The raw videos in videos/libero_three_items_0/eval_episode_*.mp4. Confirm that all three butters are actually in the basket at the end.

Every episode starts from the same layout, so the episodes differ only in the policy's sampling noise. They do not measure generalisation to new layouts.

### Baseline before fine-tuning

`bash scripts/three_items/eval.sh models/smolvla_libero 1` with the original LIBERO checkpoint ran all 900 steps (706 s) and wrote `successes: [False]`. In the raw video, the arm moves toward and over the basket but never grasps a butter; all three stay on the floor. This confirms that the evaluator calls the task and records the success flag. It also shows that the checkpoint does not perform this three-pick sequence without new demonstrations.
