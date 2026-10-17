# Difficulty levels

`--env.type=libero_levels` is a set of pick-and-place tasks of increasing difficulty, built from generated primitive objects and containers. They use the same LIBERO floor scene, robot, cameras, actions, and success check as the three-butter task. The goal is always to put every object in the container.

## Levels

| Level | Objects | What gets harder |
| --- | --- | --- |
| one_object | red cube | One pick |
| three_same | 3 red cubes | Several identical picks in a row |
| three_shapes | red cube, red cylinder, red triangular prism | Shape: square, round, triangle |
| three_sizes | red cylinders: small, medium, large | Size, same shape and colour |
| colors_sizes | small red, medium green, large blue cylinder | Colour and size |

| Container | Inside (L × W × H) | Character |
| --- | --- | --- |
| tray_wide | 20 × 14 × 4 cm | Wide and shallow: easy to hit, objects can roll over the low rim |
| bin_narrow | 10 × 10 × 12 cm | Narrow and tall: small target, the arm must lift higher |

Each level is paired with each container, giving 10 tasks. Task IDs follow `levels.TASKS`:

| ID | Task | Instruction |
| --- | --- | --- |
| 0 | one_object_tray_wide | put the red cube in the tray |
| 1 | one_object_bin_narrow | put the red cube in the bin |
| 2 | three_same_tray_wide | put the three red cubes in the tray |
| 3 | three_same_bin_narrow | put the three red cubes in the bin |
| 4 | three_shapes_tray_wide | put the cube, the cylinder and the triangle in the tray |
| 5 | three_shapes_bin_narrow | put the cube, the cylinder and the triangle in the bin |
| 6 | three_sizes_tray_wide | put the small, the medium and the large cylinder in the tray |
| 7 | three_sizes_bin_narrow | put the small, the medium and the large cylinder in the bin |
| 8 | colors_sizes_tray_wide | put the red, the green and the blue cylinder in the tray |
| 9 | colors_sizes_bin_narrow | put the red, the green and the blue cylinder in the bin |

## Objects

| Shape | Size | Colours |
| --- | --- | --- |
| Cube | 4 cm | red, green, blue, yellow |
| Cylinder | 4 cm tall; diameter 3 cm (small), 4 cm (medium), 5.6 cm (large) | red, green, blue, yellow |
| Triangular prism | 4 cm tall, 4.6 cm sides | red, green, blue, yellow |

Every object fits in the open Panda gripper (about 8 cm) from above. [shapes.py](lerobot_env_libero_three_items/shapes.py) writes the MuJoCo XML files (and an OBJ mesh for the prism) into `assets/shapes/` and registers each one as a LIBERO object category, such as `cylinder_large_blue` or `tray_wide`. To add an object or a container, add it to `shapes.py`; to add a level, add it to `LEVELS` in [levels.py](lerobot_env_libero_three_items/levels.py), which generates the BDDL scene files in `assets/levels/`.

## Scripted controller

The controller in [scripted.py](lerobot_env_libero_three_items/scripted.py) reads the scene to adapt to each object and container:

- **Grasp height:** 3 mm above the object's centre.
- **Carry and release heights:** from the container's rim, so the object clears the 12 cm bin walls.
- **Wrist angle:** turns to the nearest equivalent heading of each shape. A cube repeats every 90°, a prism every 120°, a cylinder needs no turn.
- **Drop points:** spread along the tray, or into the four quadrants of the bin.

The three-butter tasks keep their original fixed values. After this change, the scripted runs there still succeed at the same steps as before (525 for the fixed three-butter layout, 165 for one butter).

## Checking the scenes

```bash
bash scripts/levels/make_level_states.sh   # CPU only; writes assets/levels/*.pruned_init and check.json
```

For each task, the script places the objects, lets them settle, saves the initial state, and runs the scripted controller with the strict success check. The scene runs without cameras, so it does not need the GPU and can run during training. Result on the tested machine:

| Task | Scripted success at step | Settle drift |
| --- | --- | --- |
| one_object, tray / bin | 181 / 167 | 0 mm |
| three_same, tray / bin | 584 / 526 | 0 mm |
| three_shapes, tray / bin | 586 / 528 | 3.2 mm (prism) |
| three_sizes, tray / bin | 585 / 530 | 0 mm |
| colors_sizes, tray / bin | 585 / 530 | 0 mm |

All three objects ended inside the container in every task.

```bash
bash scripts/levels/render_check.sh   # GPU; videos and overview.png in outputs/levels_render_check/
```

This renders the same scripted runs through the cameras the policy uses. It is how a missing-visuals bug was found: the generated geoms are collision geoms, which the cameras do not draw, so robosuite is asked to add a drawn copy of each (`duplicate_collision_geoms=True` in shapes.py). With that fix, all ten tasks render with the objects and containers visible and succeed at the same steps as without cameras.

## Demonstrations and evaluation

```bash
bash scripts/levels/collect_demos.sh 4 --episodes 50 --noise 1.0   # task 4 -> data/levels_three_shapes_tray_wide
DATASET=levels_three_shapes_tray_wide STEPS=10000 bash scripts/three_items/train.sh
bash scripts/levels/eval.sh 4 outputs/<run>/checkpoints/last/pretrained_model 5
```

Each task has one fixed layout. Random layouts like those of the three-butter task (layouts.py) are not yet generated for the levels.
