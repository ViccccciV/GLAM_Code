"""Replay GLAM world-model action predictions (Frank episodes) in MuJoCo.

Takes the val_predictions_frank.pkl produced by glam/generate_val_predictions.py,
replays the GT and predicted joint actions side by side from the episode's real
initial state, saves stitched comparison videos (GT top, prediction bottom),
per-dimension action curves, and optionally checks task success.

Tasks: 1 = stack two (right arm, 8-d actions), 2 = stack three (dual arm, 16-d
actions in ENV layout).
"""
import argparse
import pickle
import warnings
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from tqdm import tqdm

from bimanual_suite.mjc.cubes import CubeBaseEnvironment, GRIPPER_RANGE, CALIBRATION_ACTIONS, TASK_PTU_TILT

warnings.filterwarnings("ignore")


# ----------------------------
# Environments
# ----------------------------
class TwoCubesEnvironment(CubeBaseEnvironment):
    xml_file = "two_cubes.xml"
    blue_qpos = None
    orange_qpos = None

    def get_cube_poses(self):
        cubes = {"blue": self.blue_qpos, "orange": self.orange_qpos}
        cube_pos = {}
        cube_quat = {}
        for k, cube_idx in cubes.items():
            if cube_idx is not None:
                cube_pos[f"{k}_pos"] = self.data.qpos[cube_idx: cube_idx + 3].copy()
                cube_quat[f"{k}_quat"] = self.data.qpos[cube_idx + 3: cube_idx + 7].copy()
        return {**cube_pos, **cube_quat}

    def cube_init(self, seed: int):
        self.blue_qpos = self.name_to_qpos_adr("blue_cube")
        self.orange_qpos = self.name_to_qpos_adr("orange_cube")
        self.data.qpos[self.blue_qpos: self.blue_qpos + 3] = [0.4, -0.05, 0.79]
        self.data.qpos[self.orange_qpos: self.orange_qpos + 3] = [0.5, -0.05, 0.79]

    def reset(self, seed: int | None = None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.calibration_qpos[:]
        mujoco.mj_forward(self.model, self.data)
        if seed is not None:
            self.cube_init(seed)
        for _ in range(10):
            self.step(CALIBRATION_ACTIONS)
        return self.observation


class ThreeCubesEnvironment(CubeBaseEnvironment):
    xml_file = "three_cubes.xml"
    blue_qpos = None
    orange_qpos = None
    purple_qpos = None

    def get_cube_poses(self):
        cubes = {"blue": self.blue_qpos, "orange": self.orange_qpos, "purple": self.purple_qpos}
        cube_pos = {}
        cube_quat = {}
        for k, cube_idx in cubes.items():
            if cube_idx is not None:
                cube_pos[f"{k}_pos"] = self.data.qpos[cube_idx: cube_idx + 3].copy()
                cube_quat[f"{k}_quat"] = self.data.qpos[cube_idx + 3: cube_idx + 7].copy()
        return {**cube_pos, **cube_quat}

    def cube_init(self, seed: int):
        self.blue_qpos = self.name_to_qpos_adr("blue_cube")
        self.orange_qpos = self.name_to_qpos_adr("orange_cube")
        self.purple_qpos = self.name_to_qpos_adr("purple_cube")
        self.data.qpos[self.blue_qpos: self.blue_qpos + 3] = [0.4, -0.05, 0.79]
        self.data.qpos[self.orange_qpos: self.orange_qpos + 3] = [0.5, -0.05, 0.79]
        self.data.qpos[self.purple_qpos: self.purple_qpos + 3] = [0.6, -0.05, 0.79]

    def reset(self, seed: int | None = None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.calibration_qpos[:]
        mujoco.mj_forward(self.model, self.data)
        if seed is not None:
            self.cube_init(seed)
        for _ in range(10):
            self.step(CALIBRATION_ACTIONS)
        return self.observation


# ----------------------------
# Cube init helpers (from NPZ)
# ----------------------------
def _load_init_poses(path: Path):
    try:
        data = np.load(path, allow_pickle=True)
        b_init = data["blue_pose_world_init"] if "blue_pose_world_init" in data else (data["blue_pose_world"][0] if "blue_pose_world" in data else None)
        o_init = data["orange_pose_world_init"] if "orange_pose_world_init" in data else None
        p_init = data["purple_pose_world_init"] if "purple_pose_world_init" in data else None
        return b_init, o_init, p_init
    except Exception as e:
        print(f"  [Error] Found file but failed to load: {e}")
        return None


def get_init_poses_from_npz(data_root: Path | None, episode_key: str, item: dict | None = None):
    """Initial cube poses for one episode. Tries the episode npz path recorded in
    the pkl first, then falls back to searching under --data_root."""
    if item is not None:
        src = item.get("umi_file_path")
        if src and Path(src).exists():
            res = _load_init_poses(Path(src))
            if res is not None:
                return res

    if data_root is None:
        print(f"  [Warn] {episode_key}: episode npz not found (no recorded path, no --data_root); cubes stay at env defaults!")
        return None, None, None

    parts = episode_key.split('_')
    ep_id = parts[-1]

    candidates = [
        data_root / "task_01" / "success" / ep_id / "frank_blue_grasp.npz",
        data_root / "task_02" / "success" / ep_id / "frank_blue_grasp.npz",
        data_root / "success" / ep_id / "frank_blue_grasp.npz",
        data_root / ep_id / "frank_blue_grasp.npz",
    ]
    candidates += [p.parent.parent / "failure" / ep_id / "frank_blue_grasp.npz" for p in candidates]

    for path in candidates:
        if path.exists():
            res = _load_init_poses(path)
            if res is not None:
                return res

    print(f"  [Warn] {episode_key}: episode npz not found under {data_root}; cubes stay at env defaults!")
    return None, None, None


def set_cubes_poses(env, blue_pose, orange_pose, purple_pose=None):
    """Sets the initial poses of cubes from extracted data. purple_pose is optional."""
    for pose, attr, body in [(blue_pose, "blue_qpos", "blue_cube"),
                             (orange_pose, "orange_qpos", "orange_cube"),
                             (purple_pose, "purple_qpos", "purple_cube")]:
        if pose is None or np.linalg.norm(np.asarray(pose)[:3]) < 1e-5:
            continue
        qidx = getattr(env, attr, None)
        if qidx is None:
            try:
                qidx = int(env.name_to_qpos_adr(body))
            except Exception:
                continue
        qidx = int(qidx)
        env.data.qpos[qidx:qidx + 3] = pose[:3]
        env.data.qpos[qidx + 3:qidx + 7] = pose[3:7]

    mujoco.mj_forward(env.model, env.data)


def get_cube_pos(env, name="blue_cube"):
    poses = env.get_cube_poses()
    key = name.replace("_cube", "_pos")
    if key in poses:
        return poses[key]
    try:
        qidx = env.name_to_qpos_adr(name)
        return env.data.qpos[qidx:qidx+3].copy()
    except Exception:
        return np.array([0.0, 0.0, 0.0])


# ----------------------------
# Success check
# ----------------------------
def evaluate_state(env, cube_init_z, task):
    """Task 1: blue stacked on orange. Task 2: blue on orange AND purple on blue."""
    curr_blue_pos = get_cube_pos(env, "blue_cube")

    if task == 2:
        curr_orange_pos = get_cube_pos(env, "orange_cube")
        curr_purple_pos = get_cube_pos(env, "purple_cube")
        bm_xy = float(np.linalg.norm(curr_blue_pos[:2] - curr_orange_pos[:2]))
        bm_z  = float(curr_blue_pos[2] - curr_orange_pos[2])
        mt_xy = float(np.linalg.norm(curr_purple_pos[:2] - curr_blue_pos[:2]))
        mt_z  = float(curr_purple_pos[2] - curr_blue_pos[2])
        stacked_ok = ((bm_xy < 0.025) and (abs(bm_z - 0.055) < 0.025)
                      and (mt_xy < 0.025) and (abs(mt_z - 0.055) < 0.025))
        reason = "Success" if stacked_ok else f"BO xy={bm_xy:.3f} z={bm_z:.3f} | PB xy={mt_xy:.3f} z={mt_z:.3f}"
        return stacked_ok, max(bm_xy, mt_xy), curr_blue_pos[2], reason

    curr_orange_pos = get_cube_pos(env, "orange_cube")
    xy_dist = float(np.linalg.norm(curr_blue_pos[:2] - curr_orange_pos[:2]))
    z_dist = float(abs(curr_blue_pos[2] - curr_orange_pos[2]))

    stacked_ok = (xy_dist < 0.02) and (abs(z_dist - 0.055) < 0.02)
    reason = "Success" if stacked_ok else f"XY Dist: {xy_dist:.3f}, Z Dist: {z_dist:.3f}"
    return stacked_ok, xy_dist, curr_blue_pos[2], reason


# ----------------------------
# Robot init helpers
# ----------------------------
def set_robot_init_from_init_qpos16(env, init_qpos16: np.ndarray):
    if init_qpos16 is None: return
    q = np.asarray(init_qpos16, dtype=np.float64).reshape(-1)
    if q.shape[0] != 16: return

    left_j = q[:7]
    right_j = q[7:14]
    left_g = float(q[14])
    right_g = float(q[15])

    env.data.qpos[env.left_joint_state_to_pos] = left_j
    env.data.qpos[env.right_joint_state_to_pos] = right_j
    env.data.qpos[env.left_gripper_idx] = np.clip(left_g, 0.0, 1.0) * GRIPPER_RANGE
    env.data.qpos[env.right_gripper_idx] = np.clip(right_g, 0.0, 1.0) * GRIPPER_RANGE
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    env.data.ctrl[env.joint_action_map[:7]] = left_j
    env.data.ctrl[env.joint_action_map[7:14]] = right_j
    env.data.qfrc_applied[env.dof_ids] = env.data.qfrc_bias[env.dof_ids]


def set_right_init_from_action8(env, a8: np.ndarray):
    a8 = np.asarray(a8, dtype=np.float64).reshape(-1)
    if a8.shape[0] != 8: return
    right_j = a8[:7]
    right_g = float(np.clip(a8[7], 0.0, 1.0))

    env.data.qpos[env.right_joint_state_to_pos] = right_j
    env.data.qpos[env.right_gripper_idx] = right_g * GRIPPER_RANGE
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    env.data.ctrl[env.joint_action_map[7:14]] = right_j
    env.data.qfrc_applied[env.dof_ids] = env.data.qfrc_bias[env.dof_ids]


def warmup_hold_current_pose(env, steps: int = 10):
    if steps <= 0: return
    obs = env.observation
    full = np.zeros((16,), dtype=np.float64)
    full[:7] = obs["left_pos"][:7]
    full[7:14] = obs["right_pos"][:7]
    full[14] = float(obs["left_pos"][7])
    full[15] = float(obs["right_pos"][7])
    for _ in range(int(steps)):
        env.step(full)


# ----------------------------
# Dual-arm helpers
# ----------------------------
# IMPORTANT: there are TWO 16-dim layouts in the dual-arm Frank dataset:
#   - data["actions"]   : ENV layout  [l_arm(7), r_arm(7), l_grip(1), r_grip(1)]
#                         (this is what env.step() expects)
#   - data["joint_pos"] : DATASET layout [l_arm(7), l_grip(1), r_arm(7), r_grip(1)]
#                         (mujoco qpos slice — each arm followed by its own gripper)
#
# generate_val_predictions.py copies data["actions"] verbatim into the pkl, so
# the predicted/GT 16-d actions reaching this script are already ENV layout.
# data["joint_pos"][0] is also passed through (as init_qpos) and is DATASET layout.
def joint_pos_dataset_to_env(jp16: np.ndarray) -> np.ndarray:
    """Dataset layout [l_arm, l_grip, r_arm, r_grip] -> env layout [l_arm, r_arm, l_grip, r_grip]."""
    a = np.asarray(jp16, dtype=np.float64).reshape(-1)
    out = np.zeros(16, dtype=np.float64)
    out[:7]   = a[:7]    # l_arm
    out[7:14] = a[8:15]  # r_arm
    out[14]   = a[7]     # l_grip
    out[15]   = a[15]    # r_grip
    return out


def set_robot_init_from_env_action16(env, a16_env: np.ndarray):
    """Set both arms from a 16-d action in ENV layout [l_arm(7), r_arm(7), l_grip, r_grip]."""
    a = np.asarray(a16_env, dtype=np.float64).reshape(-1)
    if a.shape[0] != 16: return
    l_j = a[:7]
    r_j = a[7:14]
    l_g = float(np.clip(a[14], 0.0, 1.0))
    r_g = float(np.clip(a[15], 0.0, 1.0))

    env.data.qpos[env.left_joint_state_to_pos]  = l_j
    env.data.qpos[env.right_joint_state_to_pos] = r_j
    env.data.qpos[env.left_gripper_idx]  = l_g * GRIPPER_RANGE
    env.data.qpos[env.right_gripper_idx] = r_g * GRIPPER_RANGE
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.data.ctrl[env.joint_action_map[:7]]   = l_j
    env.data.ctrl[env.joint_action_map[7:14]] = r_j
    env.data.qfrc_applied[env.dof_ids] = env.data.qfrc_bias[env.dof_ids]


def _append_frame(env, frames, camera="front"):
    env.renderer.update_scene(env.data, camera=camera)
    frames.append(env.renderer.render())


def run_sim_and_get_frames_joint_dual(env, init_qpos, blue_init, orange_init, purple_init,
                                      actions16_env, repeat, warmup_steps, render_each_action,
                                      check_success, task):
    """Dual-arm joint replay. actions16_env is shape (T, 16) in ENV layout."""
    try: env.reset()
    except TypeError: env.reset(seed=0)

    set_cubes_poses(env, blue_init, orange_init, purple_init)
    cube_init_z = get_cube_pos(env, "blue_cube")[2]

    final_success = False; final_reason = "Not Started"; final_dist = 999.9

    # init_qpos comes from data["joint_pos"][0] which is DATASET layout — convert.
    if init_qpos is not None and np.asarray(init_qpos).shape[-1] == 16:
        init_env = joint_pos_dataset_to_env(init_qpos)
        set_robot_init_from_env_action16(env, init_env)

    # actions16_env is already in ENV layout (data["actions"] verbatim).
    if actions16_env is not None and len(actions16_env) > 0:
        set_robot_init_from_env_action16(env, actions16_env[0])

    warmup_hold_current_pose(env, steps=warmup_steps)

    frames = []
    T = len(actions16_env)
    repeat = max(int(repeat), 1)

    for t in range(T):
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


# ----------------------------
# Single-arm joint replay
# ----------------------------
def build_full_action_hold_left_joint(env, right_a8: np.ndarray) -> np.ndarray:
    obs = env.observation
    full = np.zeros((16,), dtype=np.float64)
    full[:7] = obs["left_pos"][:7]
    full[7:14] = right_a8[:7]
    full[14] = float(obs["left_pos"][7])
    full[15] = float(right_a8[7])
    return full


def run_sim_and_get_frames_joint(env, init_qpos, blue_init, orange_init, actions8, repeat, warmup_steps, render_each_action, check_success, task):
    try: env.reset()
    except TypeError: env.reset(seed=0)

    set_cubes_poses(env, blue_init, orange_init)
    cube_init_z = get_cube_pos(env, "blue_cube")[2]

    final_success = False
    final_reason = "Not Started"
    final_dist = 999.9

    if init_qpos is not None:
        set_robot_init_from_init_qpos16(env, init_qpos)

    if actions8 is not None and len(actions8) > 0:
        set_right_init_from_action8(env, actions8[0])

    warmup_hold_current_pose(env, steps=warmup_steps)

    frames = []
    T = len(actions8)
    repeat = max(int(repeat), 1)

    for t in range(T):
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


# ----------------------------
# Plotting
# ----------------------------
def save_curves(out_dir: Path, key: str, gt: np.ndarray, pred: np.ndarray):
    names = [f"J{i}" for i in range(7)] + ["Grip"]
    dim = 8
    fig, axs = plt.subplots(dim, 1, figsize=(8, 2 * dim))
    for d in range(dim):
        axs[d].plot(gt[:, d], "k-", label="GT")
        axs[d].plot(pred[:, d], "r--", label="Pred")
        axs[d].set_title(names[d])
        if d == 0:
            axs[d].legend()
        axs[d].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{key}_curves_joint.png")
    plt.close()


# ----------------------------
# Multi-file support & chunk unrolling
# ----------------------------
def _collect_pred_files_from_folder(folder: str, target_name: str) -> list[str]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"--folder not found: {root}")
    return sorted([str(p) for p in root.rglob(target_name) if p.is_file()])


def _flatten_chunk_actions(action_array: np.ndarray, replan_every: int) -> np.ndarray:
    """
    Unrolls a (T, K, Dim) chunked trajectory array into a flat (T, Dim) array
    by executing `replan_every` steps from each chunk before querying the next prediction.
    """
    if action_array.ndim == 3:
        T, K, Dim = action_array.shape
        flat = np.zeros((T, Dim), dtype=action_array.dtype)
        t = 0
        while t < T:
            # Take up to `replan_every` steps from the chunk predicted at time `t`,
            # without exceeding the chunk size `K` or the remaining length `T - t`.
            steps = min(replan_every, K, T - t)
            flat[t:t+steps] = action_array[t, :steps, :]
            t += steps
        return flat
    return action_array


def _detect_mode(arr: np.ndarray) -> str:
    if arr.ndim == 2 and arr.shape[1] == 16: return "joint_dual"
    if arr.ndim == 2 and arr.shape[1] == 8: return "joint"
    raise ValueError(
        f"Unsupported action width {arr.shape[-1]} — this replay script only supports "
        f"joint actions (8-d single-arm or 16-d dual-arm). Train the action decoder "
        f"with --target_action_type joint."
    )


def _run_one_pred_file(args, pred_file: str):
    pkl_path = Path(pred_file)
    print(f"Loading predictions from: {pkl_path}")

    if args.out_dir is None:
        # Default: put results next to the pkl (val_predictions_frank.pkl -> sim_results_frank/)
        folder_name = pkl_path.stem.replace("val_predictions_", "sim_results_")
        out_dir = pkl_path.parent / folder_name
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
    data_root = Path(args.data_root) if args.data_root else None

    log_file = None
    if args.check_success:
        log_path = out_dir / "evaluation_logs.txt"
        log_file = open(log_path, "a")
        log_file.write(f"\n=== Evaluation Started: {pkl_path.name} (Task {args.task}) ===\n")

    for key, item in tqdm(data_db.items(), desc=f"Simulating ({pkl_path.name})"):
        pred_action = _flatten_chunk_actions(np.asarray(item["pred_action"]), args.replan_every)
        gt_action = _flatten_chunk_actions(np.asarray(item["gt_action"]), args.replan_every)

        pred_mode = _detect_mode(pred_action)
        gt_mode = _detect_mode(gt_action)

        init_qpos = np.asarray(item.get("init_qpos")) if item.get("init_qpos") is not None else None

        if not args.no_curves and gt_mode == pred_mode == "joint":
            save_curves(out_dir, key, gt_action, pred_action)

        blue_init, orange_init, purple_init = get_init_poses_from_npz(data_root, key, item)

        for rep in repeat_list:
            # --- GT replay ---
            if gt_mode == "joint_dual":
                frames_gt, succ_gt, dist_gt, reason_gt = run_sim_and_get_frames_joint_dual(
                    env, init_qpos, blue_init, orange_init, purple_init, gt_action,
                    rep, args.warmup_steps, args.render_each_action, args.check_success, args.task
                )
            else:
                frames_gt, succ_gt, dist_gt, reason_gt = run_sim_and_get_frames_joint(
                    env, init_qpos, blue_init, orange_init, gt_action,
                    rep, args.warmup_steps, args.render_each_action, args.check_success, args.task
                )

            # --- Prediction replay ---
            if pred_mode == "joint_dual":
                frames_pr, succ_pr, dist_pr, reason_pr = run_sim_and_get_frames_joint_dual(
                    env, init_qpos, blue_init, orange_init, purple_init, pred_action,
                    rep, args.warmup_steps, args.render_each_action, args.check_success, args.task
                )
            else:
                frames_pr, succ_pr, dist_pr, reason_pr = run_sim_and_get_frames_joint(
                    env, init_qpos, blue_init, orange_init, pred_action,
                    rep, args.warmup_steps, args.render_each_action, args.check_success, args.task
                )

            # --- Stitch & save: GT on top, prediction on bottom ---
            min_len = min(len(frames_gt), len(frames_pr))
            if min_len == 0:
                continue

            stitched = []
            for i in range(min_len):
                stitched.append(np.vstack([
                    frames_gt[i].astype(np.uint8),
                    frames_pr[i].astype(np.uint8),
                ]))

            suffix = ""
            if args.check_success:
                suffix = "_succ" if succ_pr else "_fail"
                if log_file:
                    gt_s = "SUCCESS" if succ_gt else f"FAIL({reason_gt})"
                    pr_s = "SUCCESS" if succ_pr else f"FAIL({reason_pr})"
                    log_file.write(f"[{key}] GT: {gt_s} (d={dist_gt:.3f}) | Pred: {pr_s} (d={dist_pr:.3f})\n")
                    log_file.flush()

            save_p = out_dir / f"{key}_rep{rep}{suffix}.mp4"
            imageio.mimsave(save_p, stitched, fps=args.video_fps)

    if log_file: log_file.close()
    try: env.close()
    except Exception: pass


def main(args):
    if args.folder is not None:
        pred_files = _collect_pred_files_from_folder(args.folder, "val_predictions_frank.pkl")
        if len(pred_files) == 0:
            print(f"[Warn] No val_predictions_frank.pkl found under: {args.folder}")
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
    g.add_argument("--folder", type=str, help="Folder to search subfolders for val_predictions_frank.pkl")
    p.add_argument("--data_root", type=str, default=None, help="Fallback root of the Frank data folder for initial cube poses (the episode path recorded in the pkl is tried first)")
    p.add_argument("--out_dir", type=str, default=None, help="Default: sim_results_frank/ next to the pkl")
    p.add_argument("--task", type=int, default=1, choices=[1, 2], help="1 = stack two, 2 = stack three")
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--repeat_list", type=str, default="4", help="Comma-separated env-steps per action to try")
    p.add_argument("--render_each_action", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--render_h", type=int, default=480)
    p.add_argument("--render_w", type=int, default=640)
    p.add_argument("--no_curves", action="store_true")
    p.add_argument("--check_success", action="store_true", help="Enable success checking and log results")
    p.add_argument("--replan_every", type=int, default=1, help="Execute N steps from each chunk before querying the next prediction.")

    args = p.parse_args()
    main(args)
