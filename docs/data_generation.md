# Data collection

This page documents how the released datasets were collected, so you can
regenerate them, scale them up, or collect your own data (including real-robot
data) in the same format. Everything below runs in the `glam_eval` conda
environment (see the [installation section](../README.md#installation)), except
the SAM 3 mask extraction which needs its own environment.

The pipeline has three parts:

1. **Scripted expert demonstrations in simulation** — Frank (the dual-arm
   Kinova robot, target set) and the floating UMI gripper (auxiliary set).
2. **Object-mask extraction** with SAM 3 — required for the mask-based GLAM-O
   world model (the default; skip this if you train with `--raw_image`).
3. **Real-robot demonstrations** — the target source for real-world tasks that plugs into the pipeline exactly like the sim target set.

## 1. Scripted expert demonstration collection in simulation

`data_gen/generate_frank_data.py` and `data_gen/generate_umi_data.py` roll out
scripted experts in MuJoCo. Task 1 (stack two) uses a single arm/gripper; task
2 (stack three) runs both arms/grippers in parallel.

The simulated UMI gripper (`sim_eval/bimanual_suite/mjc/umi_gripper.xml`) is
built on the [UMI gripper model from MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/umi_gripper)
(MIT License, Columbia Artificial Intelligence and Robotics Lab).

```bash
conda activate glam_eval

# Frank (target set)
python data_gen/generate_frank_data.py --task 1 --num 500 --seed 1 --save_video
python data_gen/generate_frank_data.py --task 2 --num 500 --seed 1 --save_video

# UMI (auxiliary set)
python data_gen/generate_umi_data.py --task 1 --num 500 --seed 2001 --save_video
python data_gen/generate_umi_data.py --task 2 --num 500 --seed 2001 --save_video
```

`--save_video` is required for GLAM training data: it writes the per-frame
RGB images (`front_frames/`, `overhead_frames/`) alongside the trajectory.
Episodes are written to the README data layout (override with `--out_root`):

```
data/stack_two/frank/task_01/{success,failure}/0001/
├── frank_blue_grasp.npz     # trajectory (joints, actions, EE poses, cube poses)
├── front_frames/            # RGB frames (0000.png, ...)
├── overhead_frames/
├── front.mp4
└── overhead.mp4
```

The scripted expert is not perfect — episodes that fail the task's success check (cube-stack geometry at the end of the episode) land in `failure/` and are excluded from training (you can delete the `failure/` folder before training). Only `success/` episodes are used downstream.

## 2. Object masks with SAM 3 (for GLAM-O)

The mask-based GLAM-O world model (the repository default) consumes binary
object masks (`*_frames_mask/`). `data_gen/extract_masks_sam3.py` produces
them with text-prompted [SAM 3](https://github.com/facebookresearch/sam3)
video segmentation, tracking the task-relevant cubes through each episode:

```bash
# task 1 (stack two): the blue cube is the manipulated object
CUDA_VISIBLE_DEVICES=0 python data_gen/extract_masks_sam3.py \
    --data_root data/stack_two --text_prompt "blue cube" \
    --camera_views front overhead --embodiments umi frank --use_video_tracking

# task 2 (stack three): blue and purple cubes are manipulated
CUDA_VISIBLE_DEVICES=0 python data_gen/extract_masks_sam3.py \
    --data_root data/stack_three --text_prompt "blue cube and purple cube" \
    --camera_views front overhead --embodiments umi frank --use_video_tracking \
    --multi_object
```

Single-object prompts keep only the largest connected component per frame
(suppressing stray detections). When the prompt names multiple objects, pass
`--multi_object` so every sizable component is kept — without it, one of the
two cubes would be dropped from every mask.

This step needs its own environment with pip install -U ultralytics (which provides the SAM 3 predictors) and a SAM 3 checkpoint from the
[SAM 3 repository](https://github.com/facebookresearch/sam3), then pass the checkpoint via `--model_path` if it is not in the working directory. Masks are written next to the frames as `{view}_frames_mask/` PNGs. If you train GLAM with `--raw_image` instead, this
step can be skipped entirely.

## 3. Real-robot demonstrations

For the real-world tasks in the paper we collect demonstrations on the real
Frank robot and feed them into the same pipeline through the `frank_real`
interface (`--frank_real_roots` in steps 1–3 of the README). The episode
format deliberately mirrors the sim layout, and the modalities GLAM consumes are minimal. Integrating GLAM's architecture and pipeline into your own robot setup is straightforward:

```
<root>/frank_real/task_<id>/success/0001/
├── frank_blue_grasp.npz
├── front_frames/            # RGB frames from an external front camera
├── overhead_frames/         # RGB frames from an external overhead camera
├── front_frames_mask/       # from step 2 (SAM 3 works on real images too,
└── overhead_frames_mask/    #   with --embodiments frank_real)
```

The npz stores time-synchronized arrays at the control rate (20 Hz in our
setup), with `T` steps matching the number of saved frames. For a single-arm
task (dual-arm doubles the widths, ordered `[left, right]`):

| Key | Shape | Content |
|---|---|---|
| `actions` | (T, 8) | commanded joint positions `[arm(7), gripper(1)]`, gripper in [0, 1] |
| `joint_pos` | (T, 8) | measured joint positions `[arm(7), gripper(1)]` |
| `frank_ee_state_overhead` / `_front` | (T, 7) | measured EE pose `[pos(3), quat wxyz(4)]` in each camera frame |

That is all GLAM needs from a real episode: RGB frames from two calibrated
external cameras (only camera extrinsics are required, to express the EE pose in camera frames), the robot's joint/gripper states and commands, and the EE pose from forward kinematics. Once labelled with action latents in step 2 of the README, real episodes supervise the downstream policy exactly like the sim target set (`--frank_real_max_episodes` in step 3).
