#!/usr/bin/env python3
"""Scripted-expert data generation for the floating UMI gripper simulation.

Tasks:
  1 (stack two):   one UMI gripper grasps the blue cube and stacks it onto the
                   orange cube.
  2 (stack three): TWO UMI grippers. The right gripper stacks blue onto orange
                   while the left gripper — running its own phase machine IN
                   PARALLEL — grasps purple and stacks it on blue.

The gripper base is pinned to the pose a Frank end-effector would occupy, so
UMI and Frank demonstrations share the same camera setup and workspace. The 6
slide/hinge joints are driven in task space via world-frame Jacobians (no IK
solver needed). Each episode saves a `umi_blue_grasp.npz` trajectory plus
(with `--save_video`) per-frame RGB images and mp4 videos for both cameras:

    <out_root>/umi/task_0<task>/{success,failure}/<episode>/
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import mujoco
import numpy as np
from scipy import spatial

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim_eval"))

from bimanual_suite.mjc.cubes import TASK_PTU_TILT
from bimanual_suite.mjc.umi_grasp_env import (
    UmiSingleGraspEnv,
    _quat_to_rotmat_scalar_first,
    compute_z_align_rot,
    compute_yaw_align_rot,
    compute_slide_jacobian,
    apply_world_delta_to_qcmd,
    apply_world_rot_to_ori_cmd,
    increase_umi_damping,
    R_S_TO_U0,
)

MJC_DIR = pathlib.Path(__file__).resolve().parent.parent / "sim_eval" / "bimanual_suite" / "mjc"

CTRL_HZ = 20
MAX_STEPS = 400

FINGER_OPEN = 0.0
FINGER_CLOSED = 0.04

# Data roots used by the README layout: task 1 = stack two, task 2 = stack three.
TASK_DATA_DIRNAME = {1: "stack_two", 2: "stack_three"}

# Main (right-gripper) phases.
ALIGN = 0
CLOSE = 1
LIFT = 2
DONE = 3
OPEN = 4
RETRACT = 5

# Task 2 LEFT-gripper parallel phases (mirror of the Frank generator).
LEFT_IDLE = 100
LEFT_APPROACH = 101
LEFT_PREGRASP = 102
LEFT_CLOSE = 103
LEFT_CARRY = 104
LEFT_OPEN = 105
LEFT_RETRACT = 106
LEFT_DONE = 107

phase_name = {
    ALIGN: "ALIGN",
    CLOSE: "CLOSE",
    LIFT: "LIFT",
    DONE: "DONE",
    OPEN: "OPEN",
    RETRACT: "RETRACT",
    LEFT_IDLE: "LEFT_IDLE",
    LEFT_APPROACH: "LEFT_APPROACH",
    LEFT_PREGRASP: "LEFT_PREGRASP",
    LEFT_CLOSE: "LEFT_CLOSE",
    LEFT_CARRY: "LEFT_CARRY",
    LEFT_OPEN: "LEFT_OPEN",
    LEFT_RETRACT: "LEFT_RETRACT",
    LEFT_DONE: "LEFT_DONE",
}

# World pose of the Frank right EE used to place the single UMI gripper base
# (so the gripper starts where the robot's hand would be).
FRANK_EE_INIT_POS = np.array([0.30700085646892056, -0.15428018174352967, 0.9760246583264004])
FRANK_EE_INIT_QUAT = np.array([0.9999301639467234, -0.005303140619198958, -0.0008811738968235569, -0.010524612183622125])


# ---------------------------- Task 1 environment ----------------------------

class UmiTwoCubesEnv(UmiSingleGraspEnv):
    """Task 1 environment (two_cube_umi.xml): adds the orange cube."""

    def __init__(self, xml_path: str, ctrl_hz: int = 20, render_height: int = 480,
                 render_width: int = 640, ptu_tilt: float = TASK_PTU_TILT[1]):
        super().__init__(xml_path, ctrl_hz, render_height, render_width, ptu_tilt=ptu_tilt)

        def body_qpos_adr(name: str) -> int:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                jid = self.model.body_jntadr[bid]
                return self.model.jnt_qposadr[jid]
            except Exception:
                return -1

        self.orange_qpos = body_qpos_adr("orange_cube")

    def get_orange_pose_world(self):
        if getattr(self, "orange_qpos", -1) < 0:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        qidx = self.orange_qpos
        pos = self.data.qpos[qidx: qidx + 3].copy()
        quat = self.data.qpos[qidx + 3: qidx + 7].copy()
        return pos, quat


def reinit_cubes_task1(
    env,
    rng: np.random.Generator,
    orange_center=(0.40, 0.13),
    blue_center=(0.26, 0.0),
    perturb: float = 0.05,
):
    """Task 1: orange and blue each perturbed around their own center."""
    blue_qidx = env.blue_qpos
    orange_qidx = getattr(env, "orange_qpos", -1)

    blue_z = float(env.data.qpos[blue_qidx + 2])
    orange_z = float(env.data.qpos[orange_qidx + 2]) if orange_qidx >= 0 else blue_z

    x_o = float(rng.uniform(orange_center[0] - perturb, orange_center[0] + perturb))
    y_o = float(rng.uniform(orange_center[1] - perturb, orange_center[1] + perturb))
    ori_o = 2.0 * np.pi * rng.uniform()
    quat_o = np.array([np.cos(ori_o / 2.0), 0.0, 0.0, np.sin(ori_o / 2.0)], dtype=np.float64)

    x_b = float(rng.uniform(blue_center[0] - perturb, blue_center[0] + perturb))
    y_b = float(rng.uniform(blue_center[1] - perturb, blue_center[1] + perturb))
    ori_b = 2.0 * np.pi * rng.uniform()
    quat_b = np.array([np.cos(ori_b / 2.0), 0.0, 0.0, np.sin(ori_b / 2.0)], dtype=np.float64)

    env.data.qpos[blue_qidx: blue_qidx + 3] = np.array([x_b, y_b, blue_z], dtype=np.float64)
    env.data.qpos[blue_qidx + 3: blue_qidx + 7] = quat_b

    if orange_qidx >= 0:
        env.data.qpos[orange_qidx: orange_qidx + 3] = np.array([x_o, y_o, orange_z], dtype=np.float64)
        env.data.qpos[orange_qidx + 3: orange_qidx + 7] = quat_o

    mujoco.mj_forward(env.model, env.data)


# ---------------------------- Shared helpers ----------------------------

def get_next_index(root: pathlib.Path) -> int:
    idxs = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(idxs) + 1 if idxs else 1


def find_umi_base_bodyid(env: UmiSingleGraspEnv):
    try:
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_base_site")
        try:
            body_id = int(env.model.site[sid].bodyid)
        except Exception:
            body_id = int(env.model.site_bodyid[sid])
        return body_id
    except Exception:
        try:
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint_x")
            return int(env.model.jnt_bodyid[jid])
        except Exception:
            return None


def compute_world_delta_rot(cur_quat_wxyz, target_quat_wxyz):
    """World-frame rotation vector that takes cur to target."""
    R_cur = spatial.transform.Rotation.from_quat(cur_quat_wxyz, scalar_first=True).as_matrix()
    R_tgt = spatial.transform.Rotation.from_quat(target_quat_wxyz, scalar_first=True).as_matrix()
    R_delta = R_tgt @ R_cur.T
    return spatial.transform.Rotation.from_matrix(R_delta).as_rotvec()


def compute_target_base_pose(frank_ee_pos, frank_ee_quat):
    """
    Compute the gripper-base body pose that places the effective umi_ee site
    at the given Frank EE world pose. The holder/site offsets below match the
    validated single-UMI calibration (base body quat ~ [1, 1, 0, 0]/sqrt(2),
    site at [0, 0.17, 0] with a 90-degree x rotation).
    """
    R_frank = _quat_to_rotmat_scalar_first(frank_ee_quat)
    T_frank = np.eye(4)
    T_frank[:3, :3] = R_frank
    T_frank[:3, 3] = frank_ee_pos

    q_holder_local = np.array([1.0, 1.0, 0.0, 0.0])
    q_holder_local /= np.linalg.norm(q_holder_local)
    R_holder = _quat_to_rotmat_scalar_first(q_holder_local)
    T_base_holder = np.eye(4)
    T_base_holder[:3, :3] = R_holder

    site_local_pos = np.array([0.0, 0.17, 0.0])
    site_local_quat = np.array([0.70710678, 0.70710678, 0.0, 0.0])
    R_site_local = _quat_to_rotmat_scalar_first(site_local_quat)
    T_holder_site = np.eye(4)
    T_holder_site[:3, :3] = R_site_local
    T_holder_site[:3, 3] = site_local_pos

    T_base_site_total = T_base_holder @ T_holder_site
    T_base_target = T_frank @ np.linalg.inv(T_base_site_total)

    target_pos = T_base_target[:3, 3]
    target_quat_wxyz = spatial.transform.Rotation.from_matrix(T_base_target[:3, :3]).as_quat(scalar_first=True)
    return target_pos, target_quat_wxyz


def save_episode_videos(trial_dir, action_log, overhead_frames, front_frames, ep):
    """Write mp4 videos and per-frame PNGs, aligned with the action log."""
    min_frames_len = min(len(action_log), len(overhead_frames), len(front_frames))

    front_vid = trial_dir / "front.mp4"
    overhead_vid = trial_dir / "overhead.mp4"

    writer_front = imageio.get_writer(front_vid, fps=CTRL_HZ, codec="mpeg4", quality=8)
    writer_over = imageio.get_writer(overhead_vid, fps=CTRL_HZ, codec="mpeg4", quality=8)

    front_frames_dir = trial_dir / "front_frames"
    overhead_frames_dir = trial_dir / "overhead_frames"
    front_frames_dir.mkdir(parents=True, exist_ok=True)
    overhead_frames_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min_frames_len):
        f_front = front_frames[idx].astype(np.uint8)
        f_over = overhead_frames[idx].astype(np.uint8)

        writer_front.append_data(f_front)
        writer_over.append_data(f_over)

        imageio.imwrite(front_frames_dir / f"{idx:04d}.png", f_front)
        imageio.imwrite(overhead_frames_dir / f"{idx:04d}.png", f_over)

    writer_front.close()
    writer_over.close()
    print(f"[ep {ep}] Saved front & overhead videos and images ({min_frames_len} frames).")


# =============================================================================
# Task 1 — single UMI gripper, stack blue onto orange.
# =============================================================================

def run_task1(args, rng):
    xml_path = str(MJC_DIR / "two_cube_umi.xml")
    env = UmiTwoCubesEnv(xml_path, ctrl_hz=CTRL_HZ, render_height=480, render_width=640,
                         ptu_tilt=TASK_PTU_TILT[1])

    out_root = pathlib.Path(args.out_root) if args.out_root else \
        pathlib.Path("data") / TASK_DATA_DIRNAME[1]
    task_root = out_root / "umi" / "task_01"
    success_root = task_root / "success"
    failure_root = task_root / "failure"
    success_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)

    success_idx = get_next_index(success_root)
    failure_idx = get_next_index(failure_root)
    print(f"Saving task 1 UMI episodes under: {task_root}")

    increase_umi_damping(env, factor=1.0)

    # Pin the gripper base so the umi_ee site matches the Frank right-EE pose.
    target_base_pos, target_base_quat = compute_target_base_pose(FRANK_EE_INIT_POS, FRANK_EE_INIT_QUAT)

    use_viewer = args.viewer
    success_count = 0
    failure_count = 0
    total_start = time.time()

    if use_viewer:
        from mujoco import viewer
        context = viewer.launch_passive(model=env.model, data=env.data,
                                        show_left_ui=False, show_right_ui=False)
    else:
        from contextlib import nullcontext
        context = nullcontext()

    with context as viewer_ctx:
        for ep in range(args.num):
            print(f"\n========== Rollout {ep + 1}/{args.num} (task 1 UMI) ==========")

            env.randomize_umi_pose(
                rng, pos_center=target_base_pos, pos_range=0.04, max_rot_deg=2.0,
                quat_center=target_base_quat,
            )

            obs = env.reset()

            ep_rng = np.random.default_rng(args.seed + ep)
            reinit_cubes_task1(env, ep_rng, orange_center=(0.40, 0.13),
                               blue_center=(0.26, 0.0), perturb=0.05)

            obs = env.get_observation()
            blue_pos0, blue_quat0 = env.get_blue_pose_world()
            orange_pos0, orange_quat0 = env.get_orange_pose_world()

            umi_pos_init_world, umi_quat_init_world = obs["umi_tcp_pos"].copy(), obs["umi_tcp_quat"].copy()
            umi_pos_init_over, umi_quat_init_over = obs["umi_pos_overhead"].copy(), obs["umi_quat_overhead"].copy()
            umi_pos_init_front, umi_quat_init_front = obs["umi_pos_front"].copy(), obs["umi_quat_front"].copy()

            base_body_id = find_umi_base_bodyid(env)
            if base_body_id is not None:
                try:
                    umi_base_pos_init = env.model.body_pos[base_body_id].copy()
                    umi_base_quat_init = env.model.body_quat[base_body_id].copy()
                except Exception:
                    umi_base_pos_init, umi_base_quat_init = None, None
            else:
                umi_base_pos_init, umi_base_quat_init = None, None

            try:
                umi_qpos_init = np.array([env.data.qpos[env.q_gx], env.data.qpos[env.q_gy], env.data.qpos[env.q_gz],
                                          env.data.qpos[env.q_rx], env.data.qpos[env.q_ry], env.data.qpos[env.q_rz]], dtype=np.float64)
            except Exception:
                umi_qpos_init = None

            q_cmd = np.array([env.data.qpos[env.q_gx], env.data.qpos[env.q_gy], env.data.qpos[env.q_gz],
                              env.data.qpos[env.q_rx], env.data.qpos[env.q_ry], env.data.qpos[env.q_rz]], dtype=np.float64)

            ori_cmd = q_cmd[3:].copy()
            finger_cmd = FINGER_OPEN

            tcp_pos = obs["umi_tcp_pos"]
            target_xy = np.array([blue_pos0[0], blue_pos0[1], tcp_pos[2]])

            grasp_height = blue_pos0[2] + 0.01
            target_grasp = np.array([blue_pos0[0], blue_pos0[1], grasp_height])
            safe_height = grasp_height + 0.1

            lift_waypoints = None
            lift_waypoint_idx = 0
            stack_yaw_remaining = 0.0
            stack_hold_count = 0

            phase = ALIGN

            overhead_frames, front_frames, action_log = [], [], []
            blue_world_pose_log, blue_over_pose_log, blue_front_pose_log = [], [], []
            umi_world_pose_log, umi_over_pose_log, umi_front_pose_log = [], [], []

            ep_start = time.time()
            step_count = 0
            pause_counter = 0

            _, J_inv = compute_slide_jacobian(env)

            while step_count < MAX_STEPS and phase != DONE:
                tcp_pos = env.get_umi_tcp_pos()

                if phase == ALIGN:
                    quat_tcp = env.get_umi_tcp_quat()
                    R_ws = _quat_to_rotmat_scalar_first(quat_tcp)

                    R_wU0 = R_ws @ R_S_TO_U0
                    u0_x_world = R_wU0[:, 0]
                    u0_z_world = R_wU0[:, 2]

                    z_target_world = np.array([0.0, 0.0, -1.0])
                    delta_rot_z, z_ok = compute_z_align_rot(u0_z_world, z_target_world, max_step=np.deg2rad(0.5))

                    if z_ok:
                        _, blue_quat_cur = env.get_blue_pose_world()
                        R_w_cube = _quat_to_rotmat_scalar_first(blue_quat_cur)
                        cube_x_world = R_w_cube[:, 0]
                        delta_rot_yaw, yaw_ok = compute_yaw_align_rot(u0_x_world, cube_x_world, max_step=np.deg2rad(0.5))
                    else:
                        delta_rot_yaw, yaw_ok = np.zeros(3), False

                    delta_rot_world = delta_rot_z + delta_rot_yaw
                    ori_cmd = apply_world_rot_to_ori_cmd(delta_rot_world, ori_cmd, env, max_norm=np.deg2rad(0.5))

                    delta_xy_world = np.array([target_xy[0] - tcp_pos[0], target_xy[1] - tcp_pos[1], 0.0])
                    q_cmd = apply_world_delta_to_qcmd(delta_xy_world, q_cmd, J_inv, max_step=0.003)
                    xy_err = np.linalg.norm(tcp_pos[:2] - target_xy[:2])
                    xy_ok = xy_err < 0.007

                    delta_z = safe_height - tcp_pos[2]
                    delta_world_z = np.array([0.0, 0.0, delta_z])
                    q_cmd = apply_world_delta_to_qcmd(delta_world_z, q_cmd, J_inv, max_step=0.003)
                    z_safe_ok = abs(delta_z) < 0.003

                    if z_ok and yaw_ok and xy_ok and z_safe_ok:
                        phase = CLOSE

                elif phase == CLOSE:
                    z_err = target_grasp[2] - tcp_pos[2]
                    z_tol = 0.003

                    if abs(z_err) > z_tol:
                        delta_world = np.array([0.0, 0.0, z_err])
                        q_cmd = apply_world_delta_to_qcmd(delta_world, q_cmd, J_inv, max_step=0.003)
                    else:
                        finger_cmd = min(FINGER_CLOSED, finger_cmd + 0.002)

                    if finger_cmd >= FINGER_CLOSED - 1e-4:
                        grasp_tcp = tcp_pos.copy()

                        # Stack waypoints (mirror of the Frank generator):
                        #   WP1 lift 10 cm, WP2 above orange (grasp offset),
                        #   WP3 rotate in place (yaw applied during WP3),
                        #   WP4 descend onto orange.
                        orange_pos, orange_quat = env.get_orange_pose_world()
                        blue_pos_at_grasp, blue_quat_at_grasp = env.get_blue_pose_world()

                        held_rot = spatial.transform.Rotation.from_quat(blue_quat_at_grasp, scalar_first=True)
                        target_rot = spatial.transform.Rotation.from_quat(orange_quat, scalar_first=True)
                        delta_yaw = target_rot.as_euler('zyx')[0] - held_rot.as_euler('zyx')[0]
                        stack_yaw_remaining = (delta_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

                        offset = grasp_tcp - blue_pos_at_grasp
                        cos_y = np.cos(stack_yaw_remaining)
                        sin_y = np.sin(stack_yaw_remaining)
                        offset_rot = np.array([
                            offset[0] * cos_y - offset[1] * sin_y,
                            offset[0] * sin_y + offset[1] * cos_y,
                            offset[2],
                        ], dtype=np.float64)

                        wp1 = grasp_tcp.copy() + np.array([0.0, 0.0, 0.1], dtype=np.float64)
                        target_blue_above = orange_pos + np.array([0.0, 0.0, 0.15], dtype=np.float64)

                        wp2 = target_blue_above + offset
                        wp3 = target_blue_above + offset_rot

                        target_blue_drop = orange_pos + np.array([0.0, 0.0, 0.055 + 0.01], dtype=np.float64)
                        wp4 = target_blue_drop + offset_rot

                        lift_waypoints = [wp1, wp2, wp3, wp4]
                        lift_waypoint_idx = 0
                        phase = LIFT

                elif phase == LIFT:
                    if lift_waypoints is not None and lift_waypoint_idx < len(lift_waypoints):
                        target_wp = lift_waypoints[lift_waypoint_idx]
                        delta_world = target_wp - tcp_pos
                        q_cmd = apply_world_delta_to_qcmd(delta_world, q_cmd, J_inv, max_step=0.01)
                        finger_cmd = FINGER_CLOSED

                        # WP3 is the in-place rotation: apply the remaining yaw in
                        # small world-frame steps while holding position.
                        if lift_waypoint_idx == 2 and abs(stack_yaw_remaining) > 1e-4:
                            step_yaw = np.clip(stack_yaw_remaining, -np.deg2rad(2.0), np.deg2rad(2.0))
                            delta_rot_world = np.array([0.0, 0.0, step_yaw])
                            ori_cmd = apply_world_rot_to_ori_cmd(delta_rot_world, ori_cmd, env, max_norm=np.deg2rad(0.7))
                            stack_yaw_remaining -= step_yaw

                        pos_err = np.linalg.norm(tcp_pos - target_wp)
                        yaw_done = (abs(stack_yaw_remaining) < 1e-3) if lift_waypoint_idx == 2 else True

                        if pos_err < 0.002 and yaw_done:
                            if lift_waypoint_idx == 2 and pause_counter < 10:
                                pause_counter += 1
                            else:
                                stack_hold_count += 1
                        else:
                            stack_hold_count = 0

                        if stack_hold_count >= 5:
                            if lift_waypoint_idx == 2:
                                pause_counter = 0
                            lift_waypoint_idx += 1
                            stack_hold_count = 0
                    else:
                        phase = OPEN

                elif phase == OPEN:
                    finger_cmd = max(FINGER_OPEN, finger_cmd - 0.002)
                    if finger_cmd <= FINGER_OPEN + 1e-4:
                        finger_cmd = FINGER_OPEN
                        wp0 = tcp_pos.copy()
                        wp1 = wp0 + np.array([0.0, 0.0, 0.1], dtype=np.float64)
                        lift_waypoints = [wp0, wp1]
                        lift_waypoint_idx = 0
                        phase = RETRACT

                elif phase == RETRACT:
                    if lift_waypoints is not None and lift_waypoint_idx < len(lift_waypoints):
                        target_wp = lift_waypoints[lift_waypoint_idx]
                        delta_world = target_wp - tcp_pos
                        q_cmd = apply_world_delta_to_qcmd(delta_world, q_cmd, J_inv, max_step=0.008)
                        finger_cmd = FINGER_OPEN
                        if np.linalg.norm(tcp_pos - target_wp) < 0.007:
                            lift_waypoint_idx += 1
                    else:
                        if pause_counter < 20:
                            pause_counter += 1
                        else:
                            phase = DONE

                else:
                    phase = DONE

                action = np.concatenate([q_cmd[:3], ori_cmd, np.array([finger_cmd])])

                if args.save_video:
                    overhead_frames.append(obs["overhead"])
                    front_frames.append(obs["front"])

                action_log.append(action.copy())

                blue_world_pose_log.append(np.concatenate([obs["blue_pos_world"], obs["blue_quat_world"]], axis=-1))
                blue_over_pose_log.append(np.concatenate([obs["blue_pos_overhead"], obs["blue_quat_overhead"]], axis=-1))
                blue_front_pose_log.append(np.concatenate([obs["blue_pos_front"], obs["blue_quat_front"]], axis=-1))

                umi_world_pose_log.append(np.concatenate([obs["umi_tcp_pos"], obs["umi_tcp_quat"]], axis=-1))
                umi_over_pose_log.append(np.concatenate([obs["umi_pos_overhead"], obs["umi_quat_overhead"]], axis=-1))
                umi_front_pose_log.append(np.concatenate([obs["umi_pos_front"], obs["umi_quat_front"]], axis=-1))

                obs = env.step(action)
                step_count += 1
                if use_viewer:
                    viewer_ctx.sync()

            ep_time = time.time() - ep_start
            print(f"[ep {ep}] Finished after {step_count} steps, final phase = {phase_name[phase]}, episode time = {ep_time:.2f}s")

            blue_pos_final, blue_quat_final = env.get_blue_pose_world()

            waypoints_executed = False
            if lift_waypoints is not None:
                waypoints_executed = (lift_waypoint_idx >= len(lift_waypoints))

            orange_pos_final, _ = env.get_orange_pose_world()
            xy_dist = float(np.linalg.norm(blue_pos_final[:2] - orange_pos_final[:2]))
            z_dist = float(abs(blue_pos_final[2] - orange_pos_final[2]))
            stacked_ok = (xy_dist < 0.02) and (abs(z_dist - 0.055) < 0.02)
            success = (phase == DONE) and waypoints_executed and stacked_ok
            debug_msg = f"task1(stack two): stacked_ok={stacked_ok}, xy_dist={xy_dist:.4f}, z_dist={z_dist:.4f}"

            if success:
                base_dir, trial_idx = success_root, success_idx
                success_idx += 1
                success_count += 1
                print(f"[ep {ep}] Marked as SUCCESS (trial {trial_idx:04d})")
            else:
                base_dir, trial_idx = failure_root, failure_idx
                failure_idx += 1
                failure_count += 1
                print(f"[ep {ep}] Marked as FAILURE (trial {trial_idx:04d})")
            print("    DEBUG ->", debug_msg)

            trial_dir = base_dir / f"{trial_idx:04d}"
            trial_dir.mkdir(parents=True, exist_ok=False)

            npz_path = trial_dir / "umi_blue_grasp.npz"
            np.savez_compressed(
                npz_path,
                actions=np.stack(action_log, axis=0),
                blue_pose_world=np.stack(blue_world_pose_log, axis=0),
                blue_pose_overhead=np.stack(blue_over_pose_log, axis=0),
                blue_pose_front=np.stack(blue_front_pose_log, axis=0),
                umi_pose_world=np.stack(umi_world_pose_log, axis=0),
                umi_pose_overhead=np.stack(umi_over_pose_log, axis=0),
                umi_pose_front=np.stack(umi_front_pose_log, axis=0),
                blue_pose_world_init=np.concatenate([blue_pos0, blue_quat0], axis=-1),
                orange_pose_world_init=np.concatenate([orange_pos0, orange_quat0], axis=-1),
                blue_pose_world_final=np.concatenate([blue_pos_final, blue_quat_final], axis=-1),
                umi_pose_world_init=np.concatenate([umi_pos_init_world, umi_quat_init_world], axis=-1),
                umi_pose_overhead_init=np.concatenate([umi_pos_init_over, umi_quat_init_over], axis=-1),
                umi_pose_front_init=np.concatenate([umi_pos_init_front, umi_quat_init_front], axis=-1),
                umi_base_pose_init_pos=umi_base_pos_init,
                umi_base_pose_init_quat=umi_base_quat_init,
                umi_qpos_init=umi_qpos_init,
                success=np.array(success, dtype=bool),
                phase_final=np.array(phase, dtype=np.int32),
                task=np.array(1, dtype=np.int32),
            )
            print("[ep", ep, "] Saved dataset to", npz_path)

            if args.save_video and len(overhead_frames) > 0:
                save_episode_videos(trial_dir, action_log, overhead_frames, front_frames, ep)

    total_time = time.time() - total_start
    print("\n========== Collection summary ==========")
    print(f"Total episodes: {args.num}")
    print(f"Success: {success_count}, Failure: {failure_count}")
    print(f"Total wall-clock time: {total_time:.2f}s ({total_time / max(args.num, 1):.2f}s per episode)")


# =============================================================================
# Task 2 — DUAL UMI grippers, parallel 3-cube stacking.
#
# Schedule (matching the Frank generator):
#   - Right gripper: ALIGN -> CLOSE -> LIFT (4 wp blue -> orange) -> OPEN ->
#     RETRACT (4 wp pulling -x/-y/+z).
#   - Left gripper: starts when right finishes LIFT wp 0; runs APPROACH ->
#     PREGRASP -> CLOSE on purple, holds; then CARRY -> OPEN -> RETRACT
#     (purple -> blue) once the right gripper has fully completed RETRACT.
# =============================================================================


class UmiThreeCubesEnv:
    """
    Self-contained dual-UMI-gripper env for task 2. Does NOT inherit from
    UmiSingleGraspEnv (whose actuator/joint names assume one gripper); provides
    per-gripper actuator/joint qpos addresses indexed by 'left' / 'right'.
    """

    def __init__(self, xml_path: str, ctrl_hz: int = 20,
                 render_height: int = 480, render_width: int = 640,
                 ptu_tilt: float = TASK_PTU_TILT[2]):
        self.ptu_tilt = float(ptu_tilt)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.ctrl_hz = ctrl_hz
        self.dt = self.model.opt.timestep
        self.sim_hz = int(1.0 / self.dt)
        self.substeps = int(self.sim_hz // self.ctrl_hz)

        # Per-gripper actuator ids
        self.act = {"left": {}, "right": {}}
        self.q = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for axis in ("x", "y", "z", "rx", "ry", "rz"):
                self.act[side][axis] = self.model.actuator(f"{side}_gripper_joint_{axis}_act").id
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper_joint_{axis}")
                self.q[side][axis] = self.model.jnt_qposadr[jid]
            self.act[side]["fingers"] = self.model.actuator(f"{side}_fingers_actuator").id

        # Site ids
        self.site = {
            "left": self.model.site("left_umi_ee").id,
            "right": self.model.site("right_umi_ee").id,
        }

        # Cube freejoint qpos addresses
        def body_qpos_adr(name):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            return self.model.jnt_qposadr[jid]

        self.blue_qpos = body_qpos_adr("blue_cube")
        self.orange_qpos = body_qpos_adr("orange_cube")
        self.purple_qpos = body_qpos_adr("purple_cube")

        # Gripper root body ids (bases pinned at runtime to the target poses).
        self.gripper_root_bid = {
            "left": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_root"),
            "right": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_root"),
        }

        # Cameras
        self.cam_overhead = self.model.camera("overhead_cam").id
        self.cam_front = self.model.camera("front").id

        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self._configure_overhead_camera_pose()
        mujoco.mj_forward(self.model, self.data)

    def _configure_overhead_camera_pose(self):
        """Mirror of UmiSingleGraspEnv._configure_overhead_camera_pose."""
        LIFT_HEIGHT = 0.40

        def _safe_set_joint_qpos(name, val):
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.data.qpos[self.model.jnt_qposadr[jid]] = val
            except Exception:
                pass

        def _safe_set_actuator_ctrl(name, val):
            try:
                self.data.ctrl[self.model.actuator(name).id] = val
            except Exception:
                pass

        _safe_set_joint_qpos("ewellix_lift_top_joint", LIFT_HEIGHT)
        _safe_set_joint_qpos("ptu_pan", 0.0)
        _safe_set_joint_qpos("ptu_tilt", self.ptu_tilt)
        _safe_set_actuator_ctrl("lift", LIFT_HEIGHT)
        _safe_set_actuator_ctrl("ptu_pan", 0.0)
        _safe_set_actuator_ctrl("ptu_tilt", self.ptu_tilt)

    def step_ctrl(self):
        """Step physics for one control timestep."""
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def randomize_umi_pose(self, side, rng, pos_center, quat_center,
                           pos_range=0.0, max_rot_deg=0.0):
        """
        Mirror of UmiSingleGraspEnv.randomize_umi_pose for ONE named gripper:
        pins {side}_gripper_root body pos/quat (with optional perturbation),
        zeros that side's 6 slide/hinge joints + qvel + ctrl, then mj_forward.
        """
        assert side in ("left", "right")
        pos_center = np.asarray(pos_center, dtype=np.float64)
        quat_center = np.asarray(quat_center, dtype=np.float64)

        if pos_range > 0.0:
            pos = pos_center + rng.uniform(-pos_range, pos_range, size=3)
        else:
            pos = pos_center.copy()

        if max_rot_deg > 0.0:
            axis = rng.standard_normal(3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            angle = rng.uniform(-np.deg2rad(max_rot_deg), np.deg2rad(max_rot_deg))
            rot_perturb = spatial.transform.Rotation.from_rotvec(axis * angle)
            rot_center = spatial.transform.Rotation.from_quat(quat_center, scalar_first=True)
            quat = (rot_perturb * rot_center).as_quat(scalar_first=True)
        else:
            quat = quat_center.copy()

        bid = self.gripper_root_bid[side]
        self.model.body_pos[bid] = pos
        self.model.body_quat[bid] = quat

        for ax in ("x", "y", "z", "rx", "ry", "rz"):
            self.data.qpos[self.q[side][ax]] = 0.0
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper_joint_{ax}")
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
            self.data.ctrl[self.act[side][ax]] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def get_tcp_pos(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.site[side]].copy()

    def get_tcp_quat(self, side: str) -> np.ndarray:
        R_ws = self.data.site_xmat[self.site[side]].reshape(3, 3)
        return spatial.transform.Rotation.from_matrix(R_ws).as_quat(scalar_first=True)

    def get_cube_pose(self, name: str):
        """Return (pos(3,), quat_wxyz(4,)) for a named cube."""
        if name == "blue":
            adr = self.blue_qpos
        elif name == "orange":
            adr = self.orange_qpos
        elif name == "purple":
            adr = self.purple_qpos
        else:
            raise ValueError(f"unknown cube: {name}")
        return self.data.qpos[adr:adr + 3].copy(), self.data.qpos[adr + 3:adr + 7].copy()

    def render_camera(self, cam_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render()

    def get_observation(self):
        """Rendered overhead/front images and per-side TCP pose info."""
        obs = {
            "overhead": self.render_camera("overhead_cam"),
            "front": self.render_camera("front"),
            "left_tcp_pos": self.get_tcp_pos("left"),
            "left_tcp_quat": self.get_tcp_quat("left"),
            "right_tcp_pos": self.get_tcp_pos("right"),
            "right_tcp_quat": self.get_tcp_quat("right"),
        }
        return obs

    def get_site_pose_in_camera(self, site_id: int, cam_name: str):
        cam_id = self.model.camera(cam_name).id
        R_wc = self.data.cam_xmat[cam_id].reshape(3, 3)
        t_wc = self.data.cam_xpos[cam_id].copy()
        site_pos = self.data.site_xpos[site_id].copy()
        R_ws = self.data.site_xmat[site_id].reshape(3, 3)
        pos_cam = R_wc.T @ (site_pos - t_wc)
        R_cs = R_wc.T @ R_ws
        quat_cam = spatial.transform.Rotation.from_matrix(R_cs).as_quat(scalar_first=True)
        return pos_cam, quat_cam


def reinit_cubes_task2(env: UmiThreeCubesEnv, rng: np.random.Generator,
                       blue_center=np.array([0.26, -0.16]),
                       orange_center=np.array([0.40, 0.00]),
                       purple_center=np.array([0.26, 0.16]),
                       perturb: float = 0.05, settle_steps: int = 30):
    """Mirror of the Frank generator's task-2 cube randomization."""
    for adr, center in (
        (env.blue_qpos, blue_center),
        (env.orange_qpos, orange_center),
        (env.purple_qpos, purple_center),
    ):
        x = float(rng.uniform(center[0] - perturb, center[0] + perturb))
        y = float(rng.uniform(center[1] - perturb, center[1] + perturb))
        ori = float(2.0 * np.pi * rng.uniform())
        quat = np.array([math.cos(ori / 2.0), 0.0, 0.0, math.sin(ori / 2.0)], dtype=np.float64)
        env.data.qpos[adr:adr + 3] = [x, y, env.data.qpos[adr + 2]]
        env.data.qpos[adr + 3:adr + 7] = quat
    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(settle_steps)):
        env.step_ctrl()


def compute_slide_jacobian_for(env: UmiThreeCubesEnv, side: str):
    """3x3 slide Jacobian (world-frame columns) for one named gripper."""
    mujoco.mj_forward(env.model, env.data)
    cols = []
    for axis in ("x", "y", "z"):
        jid = env.model.joint(f"{side}_gripper_joint_{axis}").id
        bodyid = env.model.jnt_bodyid[jid]
        R_body = env.data.xmat[bodyid].reshape(3, 3)
        axis_local = env.model.jnt_axis[jid].copy()
        cols.append(R_body @ axis_local)
    J = np.column_stack(cols)
    return J, np.linalg.inv(J)


def compute_rot_jacobian_for(env: UmiThreeCubesEnv, side: str):
    """3x3 hinge Jacobian (world-frame columns) for one named gripper."""
    mujoco.mj_forward(env.model, env.data)
    cols = []
    for axis in ("rx", "ry", "rz"):
        jid = env.model.joint(f"{side}_gripper_joint_{axis}").id
        bodyid = env.model.jnt_bodyid[jid]
        R_body = env.data.xmat[bodyid].reshape(3, 3)
        axis_local = env.model.jnt_axis[jid].copy()
        cols.append(R_body @ axis_local)
    J = np.column_stack(cols)
    return J, np.linalg.inv(J)


def apply_world_delta_to_qcmd_local(delta_world, q_cmd_local, J_inv_local, max_step=0.005):
    """Same math as umi_grasp_env.apply_world_delta_to_qcmd, on a local q_cmd."""
    dist = float(np.linalg.norm(delta_world))
    if dist > max_step and dist > 1e-9:
        delta_world = delta_world * (max_step / dist)
    dq = J_inv_local @ delta_world
    q_cmd_local[0:3] += dq
    q_cmd_local[0:3] = np.clip(q_cmd_local[0:3], -1.2, 1.2)
    return q_cmd_local


def apply_world_rot_to_ori_cmd_local(delta_rot_world, ori_cmd_local, env, side, max_norm=np.deg2rad(5.0)):
    """Same math as umi_grasp_env.apply_world_rot_to_ori_cmd, with the
    side-specific rotation Jacobian."""
    angle = float(np.linalg.norm(delta_rot_world))
    if angle > max_norm and angle > 1e-9:
        delta_rot_world = delta_rot_world * (max_norm / angle)
    _, J_rot_inv = compute_rot_jacobian_for(env, side)
    ori_cmd_local += J_rot_inv @ delta_rot_world
    return ori_cmd_local


def calculate_cube_grasp_pose_umi(current_quat_wxyz: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Given current EE quat (top-down grasp posture) and target cube quat, return
    the target EE quat aligned to grab the cube: pick the closest 90-degree-
    symmetric yaw of the cube relative to the EE x-axis.
    """
    R_cur = spatial.transform.Rotation.from_quat(current_quat_wxyz, scalar_first=True).as_matrix()
    R_cube = spatial.transform.Rotation.from_quat(target_quat_wxyz, scalar_first=True).as_matrix()
    ee_x = R_cur[:, 0]
    cube_x = R_cube[:, 0]
    ee_xy = np.array([ee_x[0], ee_x[1]])
    cube_xy = np.array([cube_x[0], cube_x[1]])
    if np.linalg.norm(ee_xy) < 1e-9 or np.linalg.norm(cube_xy) < 1e-9:
        return current_quat_wxyz.copy()
    a_ee = math.atan2(ee_xy[1], ee_xy[0])
    a_c = math.atan2(cube_xy[1], cube_xy[0])
    delta = a_c - a_ee
    delta = (delta + math.pi / 4) % (math.pi / 2) - math.pi / 4
    R_yaw = spatial.transform.Rotation.from_euler('z', delta).as_matrix()
    R_target = R_yaw @ R_cur
    return spatial.transform.Rotation.from_matrix(R_target).as_quat(scalar_first=True)


def calculate_stacking_quat_umi(grasp_quat_wxyz, held_cube_quat_wxyz, target_cube_quat_wxyz):
    """EE quat for stacking: rotate the EE so the held cube's yaw aligns with
    the target cube's yaw (folded into [-45deg, 45deg])."""
    held_rot = spatial.transform.Rotation.from_quat(held_cube_quat_wxyz, scalar_first=True)
    target_rot = spatial.transform.Rotation.from_quat(target_cube_quat_wxyz, scalar_first=True)
    delta_yaw = target_rot.as_euler('zyx')[0] - held_rot.as_euler('zyx')[0]
    delta_yaw = (delta_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    R_yaw = spatial.transform.Rotation.from_euler('z', delta_yaw).as_matrix()
    grasp_R = spatial.transform.Rotation.from_quat(grasp_quat_wxyz, scalar_first=True).as_matrix()
    target_R = R_yaw @ grasp_R
    return spatial.transform.Rotation.from_matrix(target_R).as_quat(scalar_first=True)


def build_stack_waypoints_umi(grasp_tcp_pos, grasp_tcp_quat,
                              held_cube_pos, held_cube_quat,
                              target_cube_pos, target_cube_quat):
    """
    Mirror of the Frank generator's build_stack_waypoints. Returns 4
    (pos, quat) waypoints expressed in the umi_ee site frame:
      WP1: lift +10 cm
      WP2: above target cube +15 cm (grasp orientation)
      WP3: aligned orientation, still above
      WP4: descend onto target (target_z + 0.055 + 0.01)
    """
    pos0 = grasp_tcp_pos.copy()
    quat0 = grasp_tcp_quat.copy()
    target_quat = calculate_stacking_quat_umi(quat0, held_cube_quat, target_cube_quat)

    offset = pos0 - held_cube_pos
    held_rot = spatial.transform.Rotation.from_quat(held_cube_quat, scalar_first=True)
    target_rot = spatial.transform.Rotation.from_quat(target_cube_quat, scalar_first=True)
    delta_yaw = target_rot.as_euler('zyx')[0] - held_rot.as_euler('zyx')[0]
    delta_yaw = (delta_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    cos_y = np.cos(delta_yaw)
    sin_y = np.sin(delta_yaw)
    offset_rot = np.array([
        offset[0] * cos_y - offset[1] * sin_y,
        offset[0] * sin_y + offset[1] * cos_y,
        offset[2],
    ], dtype=np.float64)

    wp1 = (pos0 + np.array([0.0, 0.0, 0.10]), quat0)
    target_above = target_cube_pos + np.array([0.0, 0.0, 0.15])
    wp2 = (target_above + offset, quat0)
    wp3 = (target_above + offset_rot, target_quat)
    target_drop = target_cube_pos + np.array([0.0, 0.0, 0.055 + 0.01])
    wp4 = (target_drop + offset_rot, target_quat)
    return [wp1, wp2, wp3, wp4]


def run_task2(args, rng):
    """Self-contained main loop for the dual-gripper 3-cube stacking task."""
    xml_path = str(MJC_DIR / "three_cubes_umi.xml")
    env = UmiThreeCubesEnv(xml_path, ctrl_hz=CTRL_HZ, render_height=480, render_width=640,
                           ptu_tilt=TASK_PTU_TILT[2])

    # Both grippers reuse the validated single-UMI init quat (jaws down); the
    # left one is additionally rotated 180 degrees around world z. Only the
    # base positions differ (mirroring the Frank arms' EE init poses).
    RIGHT_EE_INIT_QUAT = FRANK_EE_INIT_QUAT.copy()
    _R_z180_wxyz = np.array([0.0, 0.0, 0.0, 1.0])
    LEFT_EE_INIT_QUAT = (
        spatial.transform.Rotation.from_quat(_R_z180_wxyz, scalar_first=True)
        * spatial.transform.Rotation.from_quat(FRANK_EE_INIT_QUAT, scalar_first=True)
    ).as_quat(scalar_first=True)
    LEFT_EE_INIT_POS = np.array([0.3103776920270693, 0.15002473884003256, 0.9747285351239286], dtype=np.float64)
    RIGHT_EE_INIT_POS = np.array([0.30000437194762, -0.15002323996262293, 0.941309973680565], dtype=np.float64)

    LEFT_TARGET_BASE_POS, LEFT_TARGET_BASE_QUAT = compute_target_base_pose(LEFT_EE_INIT_POS, LEFT_EE_INIT_QUAT)
    RIGHT_TARGET_BASE_POS, RIGHT_TARGET_BASE_QUAT = compute_target_base_pose(RIGHT_EE_INIT_POS, RIGHT_EE_INIT_QUAT)

    out_root = pathlib.Path(args.out_root) if args.out_root else \
        pathlib.Path("data") / TASK_DATA_DIRNAME[2]
    task_root = out_root / "umi" / "task_02"
    success_root = task_root / "success"
    failure_root = task_root / "failure"
    success_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)
    success_idx = get_next_index(success_root)
    failure_idx = get_next_index(failure_root)
    print(f"Saving task 2 UMI episodes under: {task_root}")

    success_count, failure_count = 0, 0
    use_viewer = args.viewer

    if use_viewer:
        from mujoco import viewer
        viewer_ctx_mgr = viewer.launch_passive(env.model, env.data)
    else:
        viewer_ctx_mgr = None

    try:
        for ep in range(int(args.num)):
            print(f"\n========== Rollout {ep + 1}/{args.num} (task 2 UMI dual-gripper) ==========")
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))

            # Reset to home: all gripper joints at 0, fingers open.
            mujoco.mj_resetData(env.model, env.data)
            env._configure_overhead_camera_pose()
            for side in ("left", "right"):
                for axis in ("x", "y", "z", "rx", "ry", "rz"):
                    env.data.qpos[env.q[side][axis]] = 0.0
                env.data.ctrl[env.act[side]["fingers"]] = FINGER_OPEN
                for axis in ("x", "y", "z", "rx", "ry", "rz"):
                    env.data.ctrl[env.act[side][axis]] = 0.0
            mujoco.mj_forward(env.model, env.data)

            # Pin each gripper's base body at the target init pose, with the
            # same randomization as the single-gripper task.
            env.randomize_umi_pose("left", ep_rng,
                                   pos_center=LEFT_TARGET_BASE_POS,
                                   quat_center=LEFT_TARGET_BASE_QUAT,
                                   pos_range=0.04, max_rot_deg=2.0)
            env.randomize_umi_pose("right", ep_rng,
                                   pos_center=RIGHT_TARGET_BASE_POS,
                                   quat_center=RIGHT_TARGET_BASE_QUAT,
                                   pos_range=0.04, max_rot_deg=2.0)

            reinit_cubes_task2(env, ep_rng, perturb=0.05, settle_steps=30)

            blue_pos0, blue_quat0 = env.get_cube_pose("blue")
            orange_pos0, orange_quat0 = env.get_cube_pose("orange")
            purple_pos0, purple_quat0 = env.get_cube_pose("purple")

            # Per-gripper q_cmd (gx, gy, gz, rx, ry, rz).
            q_cmd = {side: np.zeros(6, dtype=np.float64) for side in ("left", "right")}
            for side in ("left", "right"):
                q_cmd[side][:] = [
                    env.data.qpos[env.q[side]["x"]], env.data.qpos[env.q[side]["y"]], env.data.qpos[env.q[side]["z"]],
                    env.data.qpos[env.q[side]["rx"]], env.data.qpos[env.q[side]["ry"]], env.data.qpos[env.q[side]["rz"]],
                ]
            finger_cmd = {"left": FINGER_OPEN, "right": FINGER_OPEN}

            init_tcp_pos = {side: env.get_tcp_pos(side) for side in ("left", "right")}
            init_tcp_quat = {side: env.get_tcp_quat(side) for side in ("left", "right")}

            # Right gripper grasp pose for blue (top-down, jaws aligned to cube yaw).
            grasp_quat_r = calculate_cube_grasp_pose_umi(init_tcp_quat["right"], blue_quat0)
            approach_pos_r = np.array([blue_pos0[0], blue_pos0[1], blue_pos0[2] + 0.10])
            grasp_pos_r = np.array([blue_pos0[0], blue_pos0[1], blue_pos0[2] + 0.005])

            # Left gripper grasp pose for purple (refined when LEFT_APPROACH completes).
            grasp_quat_l = calculate_cube_grasp_pose_umi(init_tcp_quat["left"], purple_quat0)
            approach_pos_l = np.array([purple_pos0[0], purple_pos0[1], purple_pos0[2] + 0.10])
            grasp_pos_l = np.array([purple_pos0[0], purple_pos0[1], purple_pos0[2] + 0.005])

            phase = ALIGN
            left_phase = LEFT_IDLE
            right_lift_waypoints = None
            right_lift_idx = 0
            left_lift_waypoints = None
            left_lift_idx = 0
            right_pause_counter = 0
            left_pause_counter = 0
            left_grasp_done = False
            left_grasp_tcp_pos = None
            left_grasp_tcp_quat = None
            left_purple_pos_at_grasp = None
            left_purple_quat_at_grasp = None

            action_log = []
            joint_pos_log, joint_vel_log = [], []
            frank_ee_state_world, frank_ee_state_over, frank_ee_state_front = [], [], []
            frank_ee_action_executed_world, frank_ee_action_executed_over, frank_ee_action_executed_front = [], [], []
            blue_world_pose_log = []
            overhead_frames, front_frames = [], []

            ep_start = time.time()
            step_count = 0
            DONE_MAX_STEPS = 500

            while step_count < DONE_MAX_STEPS and not (phase == DONE and left_phase == LEFT_DONE):
                # ===================== RIGHT gripper phase machine =====================
                tcp_pos_r = env.get_tcp_pos("right")
                tcp_quat_r = env.get_tcp_quat("right")
                _, J_inv_r = compute_slide_jacobian_for(env, "right")

                if phase == ALIGN:
                    # Drive right TCP to (blue_xy, approach_z) with grasp orientation.
                    target_xy_r = np.array([approach_pos_r[0], approach_pos_r[1], tcp_pos_r[2]])
                    delta_xy = np.array([target_xy_r[0] - tcp_pos_r[0], target_xy_r[1] - tcp_pos_r[1], 0.0])
                    q_cmd["right"] = apply_world_delta_to_qcmd_local(delta_xy, q_cmd["right"], J_inv_r, max_step=0.008)
                    delta_z = np.array([0.0, 0.0, approach_pos_r[2] - tcp_pos_r[2]])
                    q_cmd["right"] = apply_world_delta_to_qcmd_local(delta_z, q_cmd["right"], J_inv_r, max_step=0.008)
                    delta_rot = compute_world_delta_rot(tcp_quat_r, grasp_quat_r)
                    q_cmd["right"][3:] = apply_world_rot_to_ori_cmd_local(delta_rot, q_cmd["right"][3:].copy(), env, "right", max_norm=np.deg2rad(4.0))
                    finger_cmd["right"] = FINGER_OPEN
                    pos_err = float(np.linalg.norm(tcp_pos_r - approach_pos_r))
                    ori_err = float(np.linalg.norm(delta_rot))
                    if pos_err < 0.012 and ori_err < np.deg2rad(3.0):
                        phase = CLOSE
                        print(f"[ep {ep}] RIGHT: ALIGN done -> CLOSE")

                elif phase == CLOSE:
                    delta = grasp_pos_r - tcp_pos_r
                    q_cmd["right"] = apply_world_delta_to_qcmd_local(delta, q_cmd["right"], J_inv_r, max_step=0.008)
                    z_err = grasp_pos_r[2] - tcp_pos_r[2]
                    if abs(z_err) < 0.01:
                        finger_cmd["right"] = float(np.clip(finger_cmd["right"] + 0.008, FINGER_OPEN, FINGER_CLOSED))
                    closed_frac = (finger_cmd["right"] - FINGER_OPEN) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)
                    if closed_frac >= 0.95 and float(np.linalg.norm(tcp_pos_r - grasp_pos_r)) < 0.01:
                        finger_cmd["right"] = FINGER_CLOSED
                        right_grasp_tcp_pos = tcp_pos_r.copy()
                        right_grasp_tcp_quat = tcp_quat_r.copy()
                        cur_blue_pos, cur_blue_quat = env.get_cube_pose("blue")
                        cur_orange_pos, cur_orange_quat = env.get_cube_pose("orange")
                        right_lift_waypoints = build_stack_waypoints_umi(
                            right_grasp_tcp_pos, right_grasp_tcp_quat,
                            cur_blue_pos, cur_blue_quat,
                            cur_orange_pos, cur_orange_quat,
                        )
                        right_lift_idx = 0
                        phase = LIFT
                        print(f"[ep {ep}] RIGHT: CLOSE done -> LIFT (4 wp)")

                elif phase == LIFT:
                    if right_lift_idx < len(right_lift_waypoints):
                        target_pos, target_quat = right_lift_waypoints[right_lift_idx]
                        delta = target_pos - tcp_pos_r
                        q_cmd["right"] = apply_world_delta_to_qcmd_local(delta, q_cmd["right"], J_inv_r, max_step=0.005)
                        delta_rot = compute_world_delta_rot(tcp_quat_r, target_quat)
                        q_cmd["right"][3:] = apply_world_rot_to_ori_cmd_local(delta_rot, q_cmd["right"][3:].copy(), env, "right", max_norm=np.deg2rad(4.0))
                        pos_err = float(np.linalg.norm(tcp_pos_r - target_pos))
                        quat_err = min(
                            float(np.linalg.norm(tcp_quat_r - target_quat)),
                            float(np.linalg.norm(tcp_quat_r + target_quat)),
                        )
                        is_align_wp = (right_lift_idx == 2)
                        if pos_err < 0.008 and quat_err < 0.03:
                            if is_align_wp and right_pause_counter < 15:
                                right_pause_counter += 1
                            else:
                                right_pause_counter = 0
                                # Trigger left gripper once right LIFT wp 0 finishes.
                                if right_lift_idx == 0 and left_phase == LEFT_IDLE:
                                    left_phase = LEFT_APPROACH
                                    print(f"[ep {ep}] RIGHT: LIFT wp 0 done -> trigger LEFT APPROACH")
                                right_lift_idx += 1
                    finger_cmd["right"] = FINGER_CLOSED
                    if right_lift_waypoints is not None and right_lift_idx >= len(right_lift_waypoints):
                        phase = OPEN
                        right_pause_counter = 0
                        print(f"[ep {ep}] RIGHT: LIFT done -> OPEN")

                elif phase == OPEN:
                    finger_cmd["right"] = float(np.clip(finger_cmd["right"] - 0.005, FINGER_OPEN, FINGER_CLOSED))
                    opened_frac = (FINGER_CLOSED - finger_cmd["right"]) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)
                    if opened_frac >= 0.95:
                        finger_cmd["right"] = FINGER_OPEN
                        # Retract waypoints pulling -x, -y, +z so the right gripper
                        # clears the workspace before the left one starts stacking.
                        pos_cur = tcp_pos_r.copy()
                        quat_cur = tcp_quat_r.copy()
                        wp0 = (pos_cur, quat_cur)
                        wp1 = (pos_cur + np.array([0.0, 0.0, 0.10]), quat_cur)
                        wp2 = (pos_cur + np.array([-0.05, -0.12, 0.15]), quat_cur)
                        wp3 = (pos_cur + np.array([-0.15, -0.20, 0.15]), quat_cur)
                        right_lift_waypoints = [wp0, wp1, wp2, wp3]
                        right_lift_idx = 0
                        phase = RETRACT
                        print(f"[ep {ep}] RIGHT: OPEN done -> RETRACT")

                elif phase == RETRACT:
                    if right_lift_idx < len(right_lift_waypoints):
                        target_pos, target_quat = right_lift_waypoints[right_lift_idx]
                        delta = target_pos - tcp_pos_r
                        q_cmd["right"] = apply_world_delta_to_qcmd_local(delta, q_cmd["right"], J_inv_r, max_step=0.008)
                        delta_rot = compute_world_delta_rot(tcp_quat_r, target_quat)
                        q_cmd["right"][3:] = apply_world_rot_to_ori_cmd_local(delta_rot, q_cmd["right"][3:].copy(), env, "right", max_norm=np.deg2rad(4.0))
                        pos_err = float(np.linalg.norm(tcp_pos_r - target_pos))
                        if pos_err < 0.012:
                            right_lift_idx += 1
                    else:
                        if right_pause_counter < 20:
                            right_pause_counter += 1
                        else:
                            # Trigger LEFT_CARRY now (right has fully completed retract).
                            if left_phase == LEFT_CLOSE and left_grasp_done:
                                left_phase = LEFT_CARRY
                                left_lift_waypoints = None  # lazy build
                                left_lift_idx = 0
                                print(f"[ep {ep}] RIGHT: RETRACT done -> trigger LEFT CARRY")
                            if left_phase == LEFT_DONE:
                                phase = DONE
                                print(f"[ep {ep}] BOTH grippers finished -> DONE")
                    finger_cmd["right"] = FINGER_OPEN

                # ===================== LEFT gripper phase machine =====================
                tcp_pos_l = env.get_tcp_pos("left")
                tcp_quat_l = env.get_tcp_quat("left")
                _, J_inv_l = compute_slide_jacobian_for(env, "left")

                if left_phase == LEFT_IDLE:
                    delta = init_tcp_pos["left"] - tcp_pos_l
                    q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.003)
                    finger_cmd["left"] = FINGER_OPEN

                elif left_phase == LEFT_APPROACH:
                    delta = approach_pos_l - tcp_pos_l
                    q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.005)
                    delta_rot = compute_world_delta_rot(tcp_quat_l, grasp_quat_l)
                    q_cmd["left"][3:] = apply_world_rot_to_ori_cmd_local(delta_rot, q_cmd["left"][3:].copy(), env, "left", max_norm=np.deg2rad(5.0))
                    finger_cmd["left"] = FINGER_OPEN
                    pos_err = float(np.linalg.norm(tcp_pos_l - approach_pos_l))
                    if pos_err < 0.012:
                        # Refine grasp pose using the current purple pose.
                        cur_purple_pos, cur_purple_quat = env.get_cube_pose("purple")
                        grasp_quat_l = calculate_cube_grasp_pose_umi(tcp_quat_l, cur_purple_quat)
                        approach_pos_l = np.array([cur_purple_pos[0], cur_purple_pos[1], cur_purple_pos[2] + 0.10])
                        grasp_pos_l = np.array([cur_purple_pos[0], cur_purple_pos[1], cur_purple_pos[2] + 0.005])
                        left_phase = LEFT_PREGRASP
                        print(f"[ep {ep}] LEFT: APPROACH done -> PREGRASP")

                elif left_phase == LEFT_PREGRASP:
                    delta = grasp_pos_l - tcp_pos_l
                    q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.008)
                    delta_rot = compute_world_delta_rot(tcp_quat_l, grasp_quat_l)
                    q_cmd["left"][3:] = apply_world_rot_to_ori_cmd_local(delta_rot, q_cmd["left"][3:].copy(), env, "left", max_norm=np.deg2rad(2.0))
                    finger_cmd["left"] = FINGER_OPEN
                    if float(np.linalg.norm(tcp_pos_l - grasp_pos_l)) < 0.008:
                        left_phase = LEFT_CLOSE
                        print(f"[ep {ep}] LEFT: PREGRASP done -> CLOSE")

                elif left_phase == LEFT_CLOSE:
                    # Hold at grasp pose, close finger; wait for right RETRACT
                    # to complete before CARRY.
                    delta = grasp_pos_l - tcp_pos_l
                    q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.005)
                    finger_cmd["left"] = float(np.clip(finger_cmd["left"] + 0.008, FINGER_OPEN, FINGER_CLOSED))
                    closed_frac = (finger_cmd["left"] - FINGER_OPEN) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)
                    if closed_frac >= 0.95 and not left_grasp_done:
                        left_grasp_done = True
                        finger_cmd["left"] = FINGER_CLOSED
                        left_grasp_tcp_pos = tcp_pos_l.copy()
                        left_grasp_tcp_quat = tcp_quat_l.copy()
                        cur_purple_pos, cur_purple_quat = env.get_cube_pose("purple")
                        left_purple_pos_at_grasp = cur_purple_pos.copy()
                        left_purple_quat_at_grasp = cur_purple_quat.copy()
                        print(f"[ep {ep}] LEFT: CLOSE done; holding purple, waiting for right RETRACT")
                    if left_grasp_done:
                        finger_cmd["left"] = FINGER_CLOSED

                elif left_phase == LEFT_CARRY:
                    if left_lift_waypoints is None:
                        # Lazy build using current blue pose (blue is now stacked on orange).
                        cur_blue_pos, cur_blue_quat = env.get_cube_pose("blue")
                        left_lift_waypoints = build_stack_waypoints_umi(
                            left_grasp_tcp_pos, left_grasp_tcp_quat,
                            left_purple_pos_at_grasp, left_purple_quat_at_grasp,
                            cur_blue_pos, cur_blue_quat,
                        )
                        left_lift_idx = 0
                        print(f"[ep {ep}] LEFT: CARRY waypoints built ({len(left_lift_waypoints)} wps)")

                    if left_lift_idx < len(left_lift_waypoints):
                        target_pos, target_quat = left_lift_waypoints[left_lift_idx]
                        delta = target_pos - tcp_pos_l
                        q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.007)
                        delta_rot = compute_world_delta_rot(tcp_quat_l, target_quat)
                        q_cmd["left"][3:] = apply_world_rot_to_ori_cmd_local(delta_rot, q_cmd["left"][3:].copy(), env, "left", max_norm=np.deg2rad(4.0))
                        pos_err = float(np.linalg.norm(tcp_pos_l - target_pos))
                        quat_err = min(
                            float(np.linalg.norm(tcp_quat_l - target_quat)),
                            float(np.linalg.norm(tcp_quat_l + target_quat)),
                        )
                        is_align_wp = (left_lift_idx == 2)
                        if pos_err < 0.008 and quat_err < 0.03:
                            if is_align_wp and left_pause_counter < 15:
                                left_pause_counter += 1
                            else:
                                left_pause_counter = 0
                                left_lift_idx += 1
                    finger_cmd["left"] = FINGER_CLOSED
                    if left_lift_idx >= len(left_lift_waypoints):
                        left_phase = LEFT_OPEN
                        print(f"[ep {ep}] LEFT: CARRY done -> OPEN")

                elif left_phase == LEFT_OPEN:
                    target_pos, target_quat = left_lift_waypoints[-1]
                    delta = target_pos - tcp_pos_l
                    q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.003)
                    finger_cmd["left"] = float(np.clip(finger_cmd["left"] - 0.005, FINGER_OPEN, FINGER_CLOSED))
                    if (FINGER_CLOSED - finger_cmd["left"]) / (FINGER_CLOSED - FINGER_OPEN + 1e-8) >= 0.95:
                        finger_cmd["left"] = FINGER_OPEN
                        pos_cur_l = tcp_pos_l.copy()
                        quat_cur_l = tcp_quat_l.copy()
                        left_lift_waypoints = [(pos_cur_l + np.array([0.0, 0.0, 0.10]), quat_cur_l)]
                        left_lift_idx = 0
                        left_phase = LEFT_RETRACT
                        print(f"[ep {ep}] LEFT: OPEN done -> RETRACT")

                elif left_phase == LEFT_RETRACT:
                    if left_lift_idx < len(left_lift_waypoints):
                        target_pos, target_quat = left_lift_waypoints[left_lift_idx]
                        delta = target_pos - tcp_pos_l
                        q_cmd["left"] = apply_world_delta_to_qcmd_local(delta, q_cmd["left"], J_inv_l, max_step=0.005)
                        if float(np.linalg.norm(tcp_pos_l - target_pos)) < 0.012:
                            left_lift_idx += 1
                            if left_lift_idx >= len(left_lift_waypoints):
                                left_phase = LEFT_DONE
                                print(f"[ep {ep}] LEFT: RETRACT done -> LEFT_DONE")
                    finger_cmd["left"] = FINGER_OPEN

                # ===================== Apply ctrl & step physics =====================
                for side in ("left", "right"):
                    env.data.ctrl[env.act[side]["x"]] = q_cmd[side][0]
                    env.data.ctrl[env.act[side]["y"]] = q_cmd[side][1]
                    env.data.ctrl[env.act[side]["z"]] = q_cmd[side][2]
                    env.data.ctrl[env.act[side]["rx"]] = q_cmd[side][3]
                    env.data.ctrl[env.act[side]["ry"]] = q_cmd[side][4]
                    env.data.ctrl[env.act[side]["rz"]] = q_cmd[side][5]
                    env.data.ctrl[env.act[side]["fingers"]] = finger_cmd[side]

                # ----- logging at time t (BEFORE stepping) -----
                # action: [left_q(6) + right_q(6) + left_grip(1) + right_grip(1)] = 14
                action = np.concatenate([q_cmd["left"], q_cmd["right"],
                                         np.array([finger_cmd["left"], finger_cmd["right"]])])
                action_log.append(action.copy())

                # joint_pos / joint_vel: left(6) + right(6) = 12 each.
                jp = np.zeros(12, dtype=np.float64)
                jv = np.zeros(12, dtype=np.float64)
                for i, (side, axis) in enumerate([(s, a) for s in ("left", "right") for a in ("x", "y", "z", "rx", "ry", "rz")]):
                    qadr = env.q[side][axis]
                    jp[i] = env.data.qpos[qadr]
                    jv[i] = env.data.qvel[qadr] if qadr < env.model.nv else 0.0
                joint_pos_log.append(jp)
                joint_vel_log.append(jv)

                # frank_ee_state (T, 14) = left_ee(7) + right_ee(7).
                state_world = np.concatenate([tcp_pos_l, tcp_quat_l, tcp_pos_r, tcp_quat_r])
                pos_l_over, quat_l_over = env.get_site_pose_in_camera(env.site["left"], "overhead_cam")
                pos_l_front, quat_l_front = env.get_site_pose_in_camera(env.site["left"], "front")
                pos_r_over, quat_r_over = env.get_site_pose_in_camera(env.site["right"], "overhead_cam")
                pos_r_front, quat_r_front = env.get_site_pose_in_camera(env.site["right"], "front")
                frank_ee_state_world.append(state_world)
                frank_ee_state_over.append(np.concatenate([pos_l_over, quat_l_over, pos_r_over, quat_r_over]))
                frank_ee_state_front.append(np.concatenate([pos_l_front, quat_l_front, pos_r_front, quat_r_front]))

                bp, bq = env.get_cube_pose("blue")
                blue_world_pose_log.append(np.concatenate([bp, bq]))

                obs = env.get_observation()
                if args.save_video:
                    overhead_frames.append(obs["overhead"])
                    front_frames.append(obs["front"])

                env.step_ctrl()

                # Log executed pose AFTER the step.
                tcp_pos_l_e = env.get_tcp_pos("left")
                tcp_quat_l_e = env.get_tcp_quat("left")
                tcp_pos_r_e = env.get_tcp_pos("right")
                tcp_quat_r_e = env.get_tcp_quat("right")
                exec_world = np.concatenate([tcp_pos_l_e, tcp_quat_l_e, tcp_pos_r_e, tcp_quat_r_e])
                pos_l_o, quat_l_o = env.get_site_pose_in_camera(env.site["left"], "overhead_cam")
                pos_l_f, quat_l_f = env.get_site_pose_in_camera(env.site["left"], "front")
                pos_r_o, quat_r_o = env.get_site_pose_in_camera(env.site["right"], "overhead_cam")
                pos_r_f, quat_r_f = env.get_site_pose_in_camera(env.site["right"], "front")
                frank_ee_action_executed_world.append(exec_world)
                frank_ee_action_executed_over.append(np.concatenate([pos_l_o, quat_l_o, pos_r_o, quat_r_o]))
                frank_ee_action_executed_front.append(np.concatenate([pos_l_f, quat_l_f, pos_r_f, quat_r_f]))

                step_count += 1
                if use_viewer and viewer_ctx_mgr is not None:
                    viewer_ctx_mgr.sync()

            ep_time = time.time() - ep_start
            print(f"[ep {ep}] Finished after {step_count} steps; phase={phase_name.get(phase)} left={phase_name.get(left_phase)}; t={ep_time:.2f}s")

            # ----- Success criterion -----
            blue_pos_final, blue_quat_final = env.get_cube_pose("blue")
            orange_pos_final, orange_quat_final = env.get_cube_pose("orange")
            purple_pos_final, purple_quat_final = env.get_cube_pose("purple")
            bm_xy = float(np.linalg.norm(blue_pos_final[:2] - orange_pos_final[:2]))
            bm_z = float(blue_pos_final[2] - orange_pos_final[2])
            mt_xy = float(np.linalg.norm(purple_pos_final[:2] - blue_pos_final[:2]))
            mt_z = float(purple_pos_final[2] - blue_pos_final[2])
            stacked_ok = ((bm_xy < 0.025) and (abs(bm_z - 0.055) < 0.025)
                          and (mt_xy < 0.025) and (abs(mt_z - 0.055) < 0.025))
            success = (phase == DONE and left_phase == LEFT_DONE and stacked_ok)
            print(f"  DEBUG -> task2(umi-dual): stacked_ok={stacked_ok}, "
                  f"blue-on-orange xy={bm_xy:.4f} z={bm_z:.4f}, "
                  f"purple-on-blue xy={mt_xy:.4f} z={mt_z:.4f}")

            if success:
                base_dir, trial_idx = success_root, success_idx
                success_idx += 1
                success_count += 1
                print(f"[ep {ep}] Marked as SUCCESS (trial {trial_idx:04d})")
            else:
                base_dir, trial_idx = failure_root, failure_idx
                failure_idx += 1
                failure_count += 1
                print(f"[ep {ep}] Marked as FAILURE (trial {trial_idx:04d})")

            trial_dir = base_dir / f"{trial_idx:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            npz_path = trial_dir / "umi_blue_grasp.npz"

            np.savez_compressed(
                npz_path,
                actions=np.stack(action_log, axis=0),
                joint_pos=np.stack(joint_pos_log, axis=0),
                joint_vel=np.stack(joint_vel_log, axis=0),
                frank_ee_state_world=np.stack(frank_ee_state_world, axis=0),
                frank_ee_state_overhead=np.stack(frank_ee_state_over, axis=0),
                frank_ee_state_front=np.stack(frank_ee_state_front, axis=0),
                frank_ee_action_executed_world=np.stack(frank_ee_action_executed_world, axis=0),
                frank_ee_action_executed_overhead=np.stack(frank_ee_action_executed_over, axis=0),
                frank_ee_action_executed_front=np.stack(frank_ee_action_executed_front, axis=0),
                blue_pose_world_init=np.concatenate([blue_pos0, blue_quat0]),
                blue_pose_world_final=np.concatenate([blue_pos_final, blue_quat_final]),
                orange_pose_world_init=np.concatenate([orange_pos0, orange_quat0]),
                orange_pose_world_final=np.concatenate([orange_pos_final, orange_quat_final]),
                purple_pose_world_init=np.concatenate([purple_pos0, purple_quat0]),
                purple_pose_world_final=np.concatenate([purple_pos_final, purple_quat_final]),
                success=np.array(success, dtype=bool),
                task=np.array(2, dtype=np.int32),
            )
            print(f"[ep {ep}] Saved dataset to {npz_path}")

            if args.save_video and len(overhead_frames) > 0:
                save_episode_videos(trial_dir, action_log, overhead_frames, front_frames, ep)
            elif args.save_video:
                print(f"[ep {ep}] WARNING: save_video=True but overhead_frames is empty")

        print("\n========== Collection summary ==========")
        print(f"Total episodes: {args.num}")
        print(f"Success: {success_count}, Failure: {failure_count}")
    finally:
        if viewer_ctx_mgr is not None:
            viewer_ctx_mgr.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2],
                        help="1 = stack two (single gripper), 2 = stack three (dual gripper)")
    parser.add_argument("--num", type=int, default=1, help="Number of rollouts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--save_video", action="store_true",
                        help="Save mp4 videos and per-frame PNGs (required for GLAM training data)")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Task data root (default: data/stack_two or data/stack_three)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if int(args.task) == 1:
        run_task1(args, rng)
    else:
        run_task2(args, rng)


if __name__ == "__main__":
    main()
