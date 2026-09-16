"""Replay GLAM world-model action predictions (UMI episodes) in MuJoCo.

Takes the val_predictions_umi.pkl produced by glam/generate_val_predictions.py.
UMI episodes are human demonstrations with no robot actions, so there is no GT
replay — the point is to check whether the actions GLAM infers from human data
alone accomplish the task on the robot. Cube initial poses are read from the
UMI episode npz; the robot starts from a home pose.

The output video stacks the original UMI demonstration frames (top) with the
simulated replay (bottom), time-aligned per prediction step via the
frame_indices stored in the pkl.

Tasks: 1 = stack two (right arm, 8-d actions), 2 = stack three (dual arm, 16-d
actions in ENV layout).
"""
import argparse
import pickle
import warnings
import re
from pathlib import Path

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import mujoco
import numpy as np
from tqdm import tqdm

from bimanual_suite.mjc.cubes import CubeBaseEnvironment, GRIPPER_RANGE, CALIBRATION_ACTIONS, TASK_PTU_TILT

# Reuse the shared env classes and replay helpers from the Frank replay script
from eval_predictions_in_sim import (
    TwoCubesEnvironment, ThreeCubesEnvironment,
    set_cubes_poses, get_cube_pos, evaluate_state,
    set_robot_init_from_init_qpos16, set_right_init_from_action8,
    warmup_hold_current_pose, joint_pos_dataset_to_env, set_robot_init_from_env_action16,
    build_full_action_hold_left_joint, _append_frame,
    _collect_pred_files_from_folder, _flatten_chunk_actions, _detect_mode,
)

warnings.filterwarnings("ignore")


def extract_episode_id_from_key(key: str) -> str | None:
    match = re.search(r'(\d{4})', key)
    if match: return match.group(1)
    return None


def _load_init_poses_from_npz(npz_path: Path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        b_init = data["blue_pose_world_init"] if "blue_pose_world_init" in data else (data["blue_pose_world"][0] if "blue_pose_world" in data else None)
        o_init = data["orange_pose_world_init"] if "orange_pose_world_init" in data else None
        p_init = data["purple_pose_world_init"] if "purple_pose_world_init" in data else None
        return b_init, o_init, p_init
    except Exception as e:
        print(f"  [Error] Failed to load npz {npz_path}: {e}")
    return None, None, None


def find_umi_episode_dir(data_item: dict, episode_key: str, args) -> Path | None:
    """Locate the UMI episode folder: first under --umi_data_root/<ep_id>/, then
    the path recorded in the pkl itself."""
    if args.umi_data_root:
        ep_id = extract_episode_id_from_key(episode_key)
        if ep_id:
            local_dir = Path(args.umi_data_root) / ep_id
            if (local_dir / "umi_blue_grasp.npz").exists():
                return local_dir

    umi_path_str = data_item.get('umi_file_path')
    if umi_path_str:
        p = Path(umi_path_str)
        if p.exists():
            return p.parent

    return None


def get_init_poses(episode_dir: Path | None):
    if episode_dir is None:
        return None, None, None
    return _load_init_poses_from_npz(episode_dir / "umi_blue_grasp.npz")


def load_umi_demo_frames(episode_dir: Path | None, frame_indices, view: str,
                         target_h: int, target_w: int):
    """Load one source-demo frame per prediction step, resized to the sim frame size.

    frame_indices maps prediction step t -> original episode frame number (stored
    in the pkl by generate_val_predictions.py). Returns None if the frames folder
    is missing; missing individual frames reuse the previous one.
    """
    if episode_dir is None or frame_indices is None:
        return None
    frames_dir = episode_dir / f"{view}_frames"
    if not frames_dir.exists():
        return None

    frames = []
    last = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for idx in np.asarray(frame_indices, dtype=np.int64):
        img = None
        for name in (f"{idx:04d}.png", f"{idx:05d}.png", f"{idx}.png"):
            p = frames_dir / name
            if p.exists():
                img = cv2.imread(str(p))
                break
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            last = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(last.copy())
    return frames


def run_sim_joint_dual(env, init_qpos, blue_init, orange_init, purple_init,
                       actions16_env, repeat, warmup_steps, render_each_action,
                       check_success, task):
    """Dual-arm joint replay from a home pose (UMI has no robot init state)."""
    try: env.reset()
    except TypeError: env.reset(seed=0)

    set_cubes_poses(env, blue_init, orange_init, purple_init)
    cube_init_z = get_cube_pos(env, "blue_cube")[2]

    final_success = False; final_reason = "Not Started"; final_dist = 999.9

    # init_qpos is the dual-arm home pose in DATASET layout — convert to ENV layout.
    if init_qpos is not None and np.asarray(init_qpos).shape[-1] == 16:
        set_robot_init_from_env_action16(env, joint_pos_dataset_to_env(init_qpos))

    if actions16_env is not None and len(actions16_env) > 0:
        set_robot_init_from_env_action16(env, actions16_env[0])

    warmup_hold_current_pose(env, steps=warmup_steps)

    frames = []
    repeat = max(int(repeat), 1)
    for t in range(len(actions16_env)):
        full = np.asarray(actions16_env[t], dtype=np.float64).reshape(-1)
        for _ in range(repeat):
            env.step(full)
            if not render_each_action:
                _append_frame(env, frames, camera="front")
        if render_each_action:
            _append_frame(env, frames, camera="front")

    if check_success:
        final_success, final_dist, _, final_reason = evaluate_state(env, cube_init_z, task)
    return frames, final_success, final_dist, final_reason


def run_sim_joint(env, init_qpos, blue_init, orange_init, actions8, repeat, warmup_steps,
                  render_each_action, check_success, task):
    """Single-arm joint replay from a home pose."""
    try: env.reset()
    except TypeError: env.reset(seed=0)

    set_cubes_poses(env, blue_init, orange_init)
    cube_init_z = get_cube_pos(env, "blue_cube")[2]

    final_success = False; final_reason = "Not Started"; final_dist = 999.9

    if init_qpos is not None and np.asarray(init_qpos).shape[-1] == 16:
        set_robot_init_from_init_qpos16(env, init_qpos)

    set_right_init_from_action8(env, actions8[0])
    warmup_hold_current_pose(env, steps=warmup_steps)

    frames = []
    repeat = max(int(repeat), 1)
    for t in range(len(actions8)):
        full = build_full_action_hold_left_joint(env, actions8[t])
        for _ in range(repeat):
            env.step(full)
            if not render_each_action:
                _append_frame(env, frames, camera="front")
        if render_each_action:
            _append_frame(env, frames, camera="front")

    if check_success:
        final_success, final_dist, _, final_reason = evaluate_state(env, cube_init_z, task)
    return frames, final_success, final_dist, final_reason


def _run_one_pred_file(args, pred_file: str):
    pkl_path = Path(pred_file)
    print(f"\nLoading predictions from: {pkl_path}")

    if args.out_dir is None:
        out_dir = pkl_path.parent / "sim_results_umi"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {out_dir}")

    with open(pred_file, "rb") as f:
        data_db = pickle.load(f)

    if args.task == 2:
        env = ThreeCubesEnvironment(seed=0, render_height=args.render_h, render_width=args.render_w, ptu_tilt=TASK_PTU_TILT[args.task])
    else:
        env = TwoCubesEnvironment(seed=0, render_height=args.render_h, render_width=args.render_w, ptu_tilt=TASK_PTU_TILT[args.task])

    repeat_list = [int(x) for x in args.repeat_list.split(",") if x.strip()] or [4]

    log_file = None
    if args.check_success:
        log_path = out_dir / "evaluation_logs.txt"
        log_file = open(log_path, "a")
        log_file.write(f"\n=== Evaluation Started: {pkl_path.name} (Task {args.task}) ===\n")

    n_succ = 0; n_total = 0

    for key, item in tqdm(data_db.items(), desc=f"Simulating ({pkl_path.name})"):
        pred_action = _flatten_chunk_actions(np.asarray(item["pred_action"]), args.replan_every)
        pred_mode = _detect_mode(pred_action)

        init_qpos = np.asarray(item.get("init_qpos")) if item.get("init_qpos") is not None else None

        episode_dir = find_umi_episode_dir(item, key, args)
        blue_init, orange_init, purple_init = get_init_poses(episode_dir)

        # Source-demo frames, one per prediction step (time-aligned via the pkl's frame_indices)
        demo_frames = load_umi_demo_frames(
            episode_dir, item.get("frame_indices"), args.umi_view, args.render_h, args.render_w
        )
        if demo_frames is not None and len(demo_frames) != len(pred_action):
            print(f"  [Warn] {key}: demo frames ({len(demo_frames)}) != actions ({len(pred_action)}); skipping demo overlay.")
            demo_frames = None

        for rep in repeat_list:
            if pred_mode == "joint_dual":
                frames_pr, succ_pr, dist_pr, reason_pr = run_sim_joint_dual(
                    env, init_qpos, blue_init, orange_init, purple_init, pred_action,
                    rep, args.warmup_steps, args.render_each_action, args.check_success, args.task
                )
            else:
                frames_pr, succ_pr, dist_pr, reason_pr = run_sim_joint(
                    env, init_qpos, blue_init, orange_init, pred_action,
                    rep, args.warmup_steps, args.render_each_action, args.check_success, args.task
                )

            suffix = ""
            if args.check_success:
                n_total += 1
                if succ_pr: n_succ += 1
                suffix = "_succ" if succ_pr else "_fail"
                if log_file:
                    pr_s = "SUCCESS" if succ_pr else f"FAIL({reason_pr})"
                    log_file.write(f"[{key}] Rep{rep} | Pred: {pr_s} (d={dist_pr:.3f})\n")
                    log_file.flush()

            if len(frames_pr) > 0:
                if demo_frames is not None:
                    # Sim replay renders `frames_per_step` frames per action
                    # (1 with --render_each_action, else `rep`); repeat each demo
                    # frame accordingly so the two streams stay step-aligned.
                    frames_per_step = max(1, len(frames_pr) // len(pred_action))
                    stitched = []
                    for i, f_sim in enumerate(frames_pr):
                        step = min(i // frames_per_step, len(demo_frames) - 1)
                        stitched.append(np.vstack([
                            demo_frames[step],
                            f_sim.astype(np.uint8),
                        ]))
                    out_frames = stitched
                else:
                    out_frames = [f.astype(np.uint8) for f in frames_pr]

                save_p = out_dir / f"{key}_rep{rep}{suffix}.mp4"
                imageio.mimsave(save_p, out_frames, fps=args.video_fps)

    if args.check_success and n_total > 0:
        summary = f"Success Rate: {n_succ}/{n_total} ({100.0*n_succ/n_total:.1f}%)"
        print(summary)
        if log_file: log_file.write(summary + "\n")

    if log_file: log_file.close()
    try: env.close()
    except Exception: pass


def main(args):
    if args.folder is not None:
        pred_files = _collect_pred_files_from_folder(args.folder, "val_predictions_umi.pkl")
        if len(pred_files) == 0:
            print(f"[Warn] No val_predictions_umi.pkl found under: {args.folder}")
            return
    elif args.pred_files is not None:
        pred_files = args.pred_files
    else:
        pred_files = [args.pred_file]

    for pf in pred_files:
        _run_one_pred_file(args, pf)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred_file", type=str, help="Path to single pkl file")
    g.add_argument("--pred_files", type=str, nargs="+", help="Paths to multiple pkl files (processed in order)")
    g.add_argument("--folder", type=str, help="Folder to search subfolders for val_predictions_umi.pkl")

    p.add_argument("--umi_data_root", type=str, default=None, help="UMI success dir for the initial cube poses and demo frames (e.g. <data_root>/umi/task_01/success/)")
    p.add_argument("--umi_view", type=str, default="front", choices=["front", "overhead"], help="Which demo camera to stack above the sim replay")
    p.add_argument("--out_dir", type=str, default=None, help="Default: sim_results_umi/ next to the pkl")
    p.add_argument("--task", type=int, default=1, choices=[1, 2], help="1 = stack two, 2 = stack three")
    p.add_argument("--video_fps", type=int, default=10)
    p.add_argument("--repeat_list", type=str, default="4", help="Comma-separated env-steps per action to try")
    p.add_argument("--render_each_action", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--render_h", type=int, default=480)
    p.add_argument("--render_w", type=int, default=640)
    p.add_argument("--check_success", action="store_true", help="Enable success checking and log results")
    p.add_argument("--replan_every", type=int, default=1, help="Execute N steps from each chunk before querying the next prediction.")

    args = p.parse_args()
    main(args)
