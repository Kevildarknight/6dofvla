# Custom LIBERO task: three butters into the basket

Instruction given to the policy: “pick up the three butters one by one and place them all in the basket”.

This is a new LIBERO task for this repository, packaged as a LeRobot environment plugin. It registers `--env.type=libero_three_items` for lerobot-eval and lerobot-train without editing the vendor checkout in lerobot/.

## 1. Task specification

| Item | Choice |
| --- | --- |
| Scene | LIBERO Object floor arena, the same scene as the orange juice demo |
| Objects | Three identical butters with separate IDs: `item_1`, `item_2`, `item_3` |
| Container | `basket_1`, the LIBERO Object basket (inner box about 12 × 12 × 14 cm) |
| Layouts | Task 0: one fixed layout. Task 1: 40 random training layouts. Task 2: 10 random held-out layouts (see section 2) |
| Fixed layout, task 0 (world x, y in metres) | item_1 (−0.10, −0.18), item_2 (0.06, −0.10), item_3 (−0.15, 0.03), basket (0.00, 0.25) |
| Order | item_1, then item_2, then item_3 (scripted demonstrations follow this order) |
| Episode limit | 900 control steps at 20 Hz (45 s) |

Butter was chosen because three butters fit side by side in the basket. In the fixed layout, a butter lies flat with its 4 cm side along the axis the gripper closes on, so a straight top-down grasp works. In random layouts the butters are turned by up to 45°, and the scripted controller turns the wrist to match. Butters are always at least 13 cm apart, so the open gripper never touches a neighbouring butter.

## 2. Scene and initial state

| File | Role |
| --- | --- |
| [three_butters_basket.bddl](lerobot_env_libero_three_items/assets/three_butters_basket.bddl) | Declares the arena, the basket, the three item IDs, their placement regions, and LIBERO's goal |
| three_butters_basket{,_train,_heldout}.pruned_init (in [assets/](lerobot_env_libero_three_items/assets/)) | Saved MuJoCo states for tasks 0, 1 and 2 |
| three_butters_basket{,_train,_heldout}.layouts.json | The layout behind each saved state, in the same order |
| [layouts.py](lerobot_env_libero_three_items/layouts.py) | Sampling ranges, spacing rules, and the code that places objects |
| [task.py](lerobot_env_libero_three_items/task.py) | Layout, item IDs, instruction, and a one-task suite in the shape LeRobot's LiberoEnv reads |
| [config.py](lerobot_env_libero_three_items/config.py) | Registers `libero_three_items` as a LeRobot env type |

LIBERO's placement sampler moves objects by 1–2 cm between resets, even inside small regions. So the layouts are not sampled at reset time. They are saved ahead of time as MuJoCo states, and reset restores one of them. All three task IDs share the scene and the instruction; they differ only in their saved states:

| Task ID | Split | Layouts | Use |
| --- | --- | --- | --- |
| 0 | fixed | 1, from `LAYOUT_XY` | The original demo layout and the scene check in section 4 |
| 1 | train | 40, seed 0 | Demonstrations |
| 2 | heldout | 10, seed 1 | Evaluation only; never used for demonstrations |

Within a task, episode *i* starts from layout *i* and wraps around after the last one. Random layouts are sampled within these limits:

| Quantity | Range or rule |
| --- | --- |
| Butter position | x from −0.20 to 0.10 m, y from −0.22 to 0.08 m |
| Basket position | x from −0.08 to 0.08 m, y from 0.18 to 0.28 m |
| Butter yaw | −45° to +45° from the fixed layout's heading |
| Spacing | Butters at least 13 cm apart, and at least 18 cm from the basket centre |

[make_init_states.py](../../scripts/three_items/make_init_states.py) generates the states. For each layout, it teleports the objects, lets them settle, and rejects the layout if anything moved more than 5 mm. It then runs the scripted controller from that state and keeps the layout only if all three butters end up in the basket. After changing the ranges or counts, regenerate with `bash scripts/three_items/make_init_states.sh`.

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

The scripted rollout video is saved next to the report as scripted_episode.mp4. These checks use task 0.

For the random layouts (tasks 1 and 2):

```bash
bash scripts/three_items/check_layouts.sh   # writes outputs/three_items_layouts_*/report.json
```

[check_layouts.py](../../scripts/three_items/check_layouts.py) loads every saved layout through the same env class lerobot-eval uses. It checks four things:

- reset restores each layout's positions and butter yaws;
- every layout follows the spacing rules;
- the scripted controller succeeds from the first few layouts of each split;
- no held-out layout is a near copy of a training layout.

Result on the tested machine:

| Check | What happened | Result |
| --- | --- | --- |
| Generation | 40 training and 10 held-out layouts kept; one candidate rejected in each split because the scripted run failed | PASS |
| train_restore, heldout_restore | All 50 layouts restored within 0.03 mm and 0.01° | PASS |
| train_scripted, heldout_scripted | Scripted controller succeeded from layouts 0–2 of each split (steps 516–545) | PASS |
| separate | The closest held-out/training pair still differs by 7.6 cm for at least one object | PASS |

Generating all 51 layouts took about an hour, mostly for the verification rollouts.

## 5. Demonstrations, data mix, and fine-tuning

```bash
bash scripts/three_items/collect_demos.sh            # 100 demos -> data/three_items_demos
bash scripts/three_items/prepare_libero_subset.sh    # 160 LIBERO episodes -> data/libero_basket_subset
bash scripts/three_items/build_mix.sh                # both -> data/three_items_mix
bash scripts/three_items/train.sh                    # fine-tune on the mix -> outputs/three_items_train_*/
```

### Demonstrations

[collect_demos.py](../../scripts/three_items/collect_demos.py) records demonstrations from task 1: episode *i* uses training layout *i* (mod 40), so 100 episodes cover every training layout two or three times. Each demonstration picks the three butters one at a time: move above the butter, turn the wrist to the butter's heading, descend, close, lift, carry over the basket, open, then rise. Only episodes that pass the success check are saved.

Episodes vary in four ways:

| Variation | Detail |
| --- | --- |
| Layout | 40 training layouts (section 2) |
| Waypoints | Up to ±1 cm of random noise per episode |
| Pick order | Random permutation of item_1, item_2, item_3 (`--fixed-order` turns this off) |
| Instruction | The evaluation sentence for half of the episodes; otherwise one of four rephrasings in `LANGUAGE_VARIANTS` (`--variant-prob`) |

Observations go through the same preprocessing as lerobot-eval, so the stored images (rotated 180°) and the 8-D state match what the checkpoint receives during evaluation.

### Mixing in original LIBERO data

Training only on the new task can make the policy forget the LIBERO tasks it already solves. [prepare_libero_subset.py](../../scripts/three_items/prepare_libero_subset.py) takes part of HuggingFaceVLA/libero, the data the checkpoint was trained on, at a pinned revision. It keeps only the basket tasks, which are closest to the new one:

| Source tasks | Episodes taken |
| --- | --- |
| 10 LIBERO Object tasks, "pick up the X and place it in the basket" | 10 per task |
| 3 LIBERO-10 tasks, "put both the X and the Y in the basket" | 20 per task |

It downloads only the parquet files holding those episodes (the full dataset is 35 GB) and rewrites them with the demos' video format. The dataset's own meta/episodes file lists stale file numbers, so the script finds each episode by reading the episode_index column of every remote file. A check on one converted frame found state and action identical to the source, and images within 2/255 on average after video compression.

Both datasets are labelled 10 fps, the label HuggingFaceVLA/libero uses. In both, one frame is one 20 Hz control step; SmolVLA's action chunks are indexed by frame, so the label does not change what is learned. build_mix.sh merges the two with lerobot-edit-dataset and prints the share of frames from each.

Result on the tested machine:

| Dataset | Episodes | Frames | Share of frames | Size | Time |
| --- | --- | --- | --- | --- | --- |
| data/three_items_demos | 100 (101 attempts; one random-order run on layout 10 failed and was discarded) | 52,498 | 63% | 473 MB | about 1.7 h |
| data/libero_basket_subset | 160 from 13 tasks, read from 71 source files | 30,919 | 37% | 223 MB, plus a 6 GB download cache in data/hf_libero | about 30 min |
| data/three_items_mix | 260 | 83,417 | 100% | 695 MB | under 1 min |

In the demos, each of the six pick orders appears 14–19 times, and the evaluation sentence is used in 46 of the 100 episodes. The data/hf_libero cache is not needed after the subset is built and can be deleted.

### Fine-tuning

train.sh fine-tunes models/smolvla_libero on data/three_items_mix (set `DATASET=three_items_demos` to use the demos alone) with the checkpoint's defaults: the vision-language backbone is frozen, and the action expert and state projection (97 M parameters) are trained. The settings can be changed with the BATCH_SIZE, STEPS and SAVE_FREQ environment variables.

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
policy=outputs/three_items_train_<run>/checkpoints/last/pretrained_model
TASK_ID=1 bash scripts/three_items/eval.sh "$policy" 10   # training layouts: has it learned the task?
TASK_ID=2 bash scripts/three_items/eval.sh "$policy" 10   # held-out layouts: does it generalise?
```

The script runs lerobot-eval with `--env.type=libero_three_items --env.task_ids=[TASK_ID]` and prints `successes[i]` for each episode. TASK_ID defaults to 0, the fixed layout. Check both outputs:

- eval_info.json, at `per_task[0].metrics.successes` for task group `libero_three_items` and the chosen task ID.
- The raw videos in videos/libero_three_items_<TASK_ID>/eval_episode_*.mp4. Confirm that all three butters are actually in the basket at the end.

Episode *i* starts from layout *i* of the chosen task. With task 2 and 10 episodes, each held-out layout is tried once. With task 0, every episode repeats the same layout, so those episodes differ only in the policy's sampling noise.

### Baseline before fine-tuning

`bash scripts/three_items/eval.sh models/smolvla_libero 1` with the original LIBERO checkpoint ran all 900 steps (706 s) and wrote `successes: [False]`. In the raw video, the arm moves toward and over the basket but never grasps a butter; all three stay on the floor. This confirms that the evaluator calls the task and records the success flag. It also shows that the checkpoint does not perform this three-pick sequence without new demonstrations.
