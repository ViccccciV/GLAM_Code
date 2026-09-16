#!/usr/bin/env python3
"""Scripted-expert data generation for the Frank (dual-arm Kinova) simulation.

Tasks:
  1 (stack two):   two cubes (blue, orange). The RIGHT arm grasps the blue
                   cube and stacks it onto the orange cube.
  2 (stack three): three cubes (blue, orange, purple). The right arm stacks
                   blue onto orange while the left arm — running its own phase
                   machine IN PARALLEL — grasps purple and stacks it on blue.

The arms are driven by online-smoothed task-space targets solved with mink IK.
Each episode saves a `frank_blue_grasp.npz` trajectory plus (with
`--save_video`) per-frame RGB images and mp4 videos for both cameras, in the
same layout as the released datasets:

    <out_root>/frank/task_0<task>/{success,failure}/<episode>/
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
import mink
import mujoco
import numpy as np
from scipy import spatial

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim_eval"))

from bimanual_suite.mjc import control
from bimanual_suite.mjc.cubes import (CubeBaseEnvironment, DUAL_ARM_CALIBRATION_POSE,
                                      TASK_PTU_TILT)

CTRL_HZ = 20
MAX_STEPS = 450

FINGER_OPEN = 0.0
FINGER_CLOSED = 1.0

# Data roots used by the README layout: task 1 = stack two, task 2 = stack three.
TASK_DATA_DIRNAME = {1: "stack_two", 2: "stack_three"}

# Right-arm (main) phases.
ALIGN = 0
CLOSE = 1
LIFT = 2
DONE = 3
OPEN = 4
RETRACT = 5

# Task 2 LEFT-arm parallel sub-stack phases. The right arm runs the standard
# ALIGN -> CLOSE -> LIFT -> OPEN -> RETRACT pipeline; the left arm runs this
# independent pipeline in PARALLEL (both arms move every step).
LEFT_IDLE = 100      # left arm holds at left_ee_init (not yet started)
LEFT_APPROACH = 101  # left arm flies to above-purple approach position
LEFT_PREGRASP = 102  # left arm descends to purple grasp height (finger open)
LEFT_CLOSE = 103     # left arm closes finger on purple
LEFT_CARRY = 104     # left arm executes the purple -> blue stack waypoints
LEFT_OPEN = 105      # left arm opens finger (releases purple)
LEFT_RETRACT = 106   # left arm lifts up after release
LEFT_DONE = 107      # left arm done, holds at retract position

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


# ---------------------------- Environments ----------------------------

class TwoCubesEnvironment(CubeBaseEnvironment):
    """Task 1: two cubes (blue, orange); right arm stacks blue onto orange."""

    xml_file = "two_cubes.xml"
    blue_qpos = None
    orange_qpos = None

    def get_cube_poses(self):
        cubes = {
            "blue": self.blue_qpos,
            "orange": self.orange_qpos,
        }
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
            self.step(self.calibration_actions)
        return self.observation


class ThreeCubesEnvironment(CubeBaseEnvironment):
    """Task 2: three cubes (blue, orange, purple). Right arm stacks blue on
    orange, then the left arm stacks purple on blue."""

    xml_file = "three_cubes.xml"
    blue_qpos = None
    orange_qpos = None
    purple_qpos = None

    def get_cube_poses(self):
        cubes = {
            "blue": self.blue_qpos,
            "orange": self.orange_qpos,
            "purple": self.purple_qpos,
        }
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
        self.data.qpos[self.blue_qpos: self.blue_qpos + 3] = [0.4, -0.13, 0.79]
        self.data.qpos[self.orange_qpos: self.orange_qpos + 3] = [0.5, 0.0, 0.79]
        self.data.qpos[self.purple_qpos: self.purple_qpos + 3] = [0.4, 0.13, 0.79]

    def reset(self, seed: int | None = None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.calibration_qpos[:]
        mujoco.mj_forward(self.model, self.data)

        if seed is not None:
            self.cube_init(seed)

        for _ in range(10):
            self.step(self.calibration_actions)
        return self.observation


# ---------------------------- Cube randomization ----------------------------

def reinit_cubes_task1(env: TwoCubesEnvironment, rng: np.random.Generator,
                       orange_center: np.ndarray, blue_center: np.ndarray,
                       perturb: float = 0.05, settle_steps: int = 5):
    """Task 1: orange and blue each perturbed around their own center."""
    blue_qpos = env.blue_qpos
    orange_qpos = env.orange_qpos

    x_o = float(rng.uniform(orange_center[0] - perturb, orange_center[0] + perturb))
    y_o = float(rng.uniform(orange_center[1] - perturb, orange_center[1] + perturb))
    ori_o = float(2.0 * np.pi * rng.uniform())
    quat_o = np.array([math.cos(ori_o / 2.0), 0.0, 0.0, math.sin(ori_o / 2.0)], dtype=np.float64)

    x_b = float(rng.uniform(blue_center[0] - perturb, blue_center[0] + perturb))
    y_b = float(rng.uniform(blue_center[1] - perturb, blue_center[1] + perturb))
    ori_b = float(2.0 * np.pi * rng.uniform())
    quat_b = np.array([math.cos(ori_b / 2.0), 0.0, 0.0, math.sin(ori_b / 2.0)], dtype=np.float64)

    env.data.qpos[blue_qpos: blue_qpos + 3] = [x_b, y_b, env.data.qpos[blue_qpos + 2]]
    env.data.qpos[blue_qpos + 3: blue_qpos + 7] = quat_b
    env.data.qpos[orange_qpos: orange_qpos + 3] = [x_o, y_o, env.data.qpos[orange_qpos + 2]]
    env.data.qpos[orange_qpos + 3: orange_qpos + 7] = quat_o

    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(settle_steps)):
        env.step(env.default_actions)


def reinit_cubes_task2(env: ThreeCubesEnvironment, rng: np.random.Generator,
                       blue_center: np.ndarray, orange_center: np.ndarray,
                       purple_center: np.ndarray,
                       perturb: float = 0.04, settle_steps: int = 5):
    """
    Task 2: three cubes for two-arm stacking.
      - Blue cube on -y side (right arm grasps it, stacks onto orange)
      - Purple cube on +y side (left arm grasps it, stacks onto blue)
      - Orange cube in the middle, slightly toward +x (the base of the stack)
    """
    blue_qpos = env.blue_qpos
    orange_qpos = env.orange_qpos
    purple_qpos = env.purple_qpos

    x_b = float(rng.uniform(blue_center[0] - perturb, blue_center[0] + perturb))
    y_b = float(rng.uniform(blue_center[1] - perturb, blue_center[1] + perturb))
    ori_b = float(2.0 * np.pi * rng.uniform())
    quat_b = np.array([math.cos(ori_b / 2.0), 0.0, 0.0, math.sin(ori_b / 2.0)], dtype=np.float64)

    x_o = float(rng.uniform(orange_center[0] - perturb, orange_center[0] + perturb))
    y_o = float(rng.uniform(orange_center[1] - perturb, orange_center[1] + perturb))
    ori_o = float(2.0 * np.pi * rng.uniform())
    quat_o = np.array([math.cos(ori_o / 2.0), 0.0, 0.0, math.sin(ori_o / 2.0)], dtype=np.float64)

    x_p = float(rng.uniform(purple_center[0] - perturb, purple_center[0] + perturb))
    y_p = float(rng.uniform(purple_center[1] - perturb, purple_center[1] + perturb))
    ori_p = float(2.0 * np.pi * rng.uniform())
    quat_p = np.array([math.cos(ori_p / 2.0), 0.0, 0.0, math.sin(ori_p / 2.0)], dtype=np.float64)

    env.data.qpos[blue_qpos: blue_qpos + 3] = [x_b, y_b, env.data.qpos[blue_qpos + 2]]
    env.data.qpos[blue_qpos + 3: blue_qpos + 7] = quat_b
    env.data.qpos[orange_qpos: orange_qpos + 3] = [x_o, y_o, env.data.qpos[orange_qpos + 2]]
    env.data.qpos[orange_qpos + 3: orange_qpos + 7] = quat_o
    env.data.qpos[purple_qpos: purple_qpos + 3] = [x_p, y_p, env.data.qpos[purple_qpos + 2]]
    env.data.qpos[purple_qpos + 3: purple_qpos + 7] = quat_p

    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(settle_steps)):
        env.step(env.default_actions)


# ---------------------------- Waypoint helpers ----------------------------

def _make_se3(quat_wxyz: np.ndarray, pos_xyz: np.ndarray) -> mink.SE3:
    return mink.SE3(np.concatenate([quat_wxyz, pos_xyz], axis=0))


def calculate_stacking_quat(tcp_quat_current: np.ndarray, held_cube_quat: np.ndarray,
                            target_cube_quat: np.ndarray) -> np.ndarray:
    """EE quat that aligns the held cube's yaw with the target cube's yaw
    (folded into [-45deg, 45deg] to exploit the cube's 90-degree symmetry)."""
    held_rot = spatial.transform.Rotation.from_quat(held_cube_quat, scalar_first=True)
    target_rot = spatial.transform.Rotation.from_quat(target_cube_quat, scalar_first=True)
    held_yaw = held_rot.as_euler('zyx')[0]
    target_yaw = target_rot.as_euler('zyx')[0]
    delta_yaw = target_yaw - held_yaw
    delta_yaw_adjusted = (delta_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    delta_rot = spatial.transform.Rotation.from_euler('z', delta_yaw_adjusted)
    tcp_rot = spatial.transform.Rotation.from_quat(tcp_quat_current, scalar_first=True)
    new_tcp_rot = delta_rot * tcp_rot
    return new_tcp_rot.as_quat(scalar_first=True)


def build_stack_waypoints(grasp_tcp_pose: mink.SE3, cube_pos_at_grasp: np.ndarray,
                          cube_quat_at_grasp: np.ndarray, target_cube_pos: np.ndarray,
                          target_cube_quat: np.ndarray) -> list[mink.SE3]:
    """
    Cube-stacking waypoints (executed after the gripper closes):
      WP1: lift straight up 10 cm (grasp orientation).
      WP2: translate above the target cube (grasp orientation, grasp offset).
      WP3: rotate in place to the stacking orientation. The TCP offset is
           rotated with the same yaw so the held cube's center stays put.
      WP4: descend onto the target cube (target_z + cube height + clearance).
    """
    pos0 = grasp_tcp_pose.wxyz_xyz[4:].copy()
    quat0 = grasp_tcp_pose.wxyz_xyz[:4].copy()
    target_quat = calculate_stacking_quat(quat0, cube_quat_at_grasp, target_cube_quat)

    offset = pos0 - cube_pos_at_grasp
    held_rot = spatial.transform.Rotation.from_quat(cube_quat_at_grasp, scalar_first=True)
    target_rot = spatial.transform.Rotation.from_quat(target_cube_quat, scalar_first=True)
    delta_yaw = target_rot.as_euler('zyx')[0] - held_rot.as_euler('zyx')[0]
    yaw_adjusted = (delta_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    cos_y = np.cos(yaw_adjusted)
    sin_y = np.sin(yaw_adjusted)
    offset_rot = np.array([
        offset[0] * cos_y - offset[1] * sin_y,
        offset[0] * sin_y + offset[1] * cos_y,
        offset[2],
    ], dtype=np.float64)

    wp1 = _make_se3(quat0, pos0 + np.array([0.0, 0.0, 0.1], dtype=np.float64))

    target_above = target_cube_pos + np.array([0.0, 0.0, 0.15], dtype=np.float64)
    wp2 = _make_se3(quat0, target_above + offset)
    wp3 = _make_se3(target_quat, target_above + offset_rot)

    target_drop = target_cube_pos + np.array([0.0, 0.0, 0.055 + 0.01], dtype=np.float64)
    wp4 = _make_se3(target_quat, target_drop + offset_rot)

    return [wp1, wp2, wp3, wp4]


# ---------------------------- Misc helpers ----------------------------

def _quat_to_rotmat_scalar_first(quat_wxyz: np.ndarray) -> np.ndarray:
    return spatial.transform.Rotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()


def get_blue_pose_in_camera(env: CubeBaseEnvironment, pos_world: np.ndarray,
                            quat_world: np.ndarray, camera_name: str):
    cam_id = env.model.camera(camera_name).id
    R_wc = env.data.cam_xmat[cam_id].reshape(3, 3)
    t_wc = env.data.cam_xpos[cam_id]
    R_w_cube = _quat_to_rotmat_scalar_first(quat_world)
    pos_cam = R_wc.T @ (pos_world - t_wc)
    R_c_cube = R_wc.T @ R_w_cube
    quat_cam = spatial.transform.Rotation.from_matrix(R_c_cube).as_quat(scalar_first=True)
    return pos_cam, quat_cam


def get_site_pose_in_camera(env: CubeBaseEnvironment, site_name: str, camera_name: str):
    site_id = env.model.site(site_name).id
    R_w_site = env.data.site_xmat[site_id].reshape(3, 3)
    t_w_site = env.data.site_xpos[site_id].copy()
    cam_id = env.model.camera(camera_name).id
    R_w_cam = env.data.cam_xmat[cam_id].reshape(3, 3)
    t_w_cam = env.data.cam_xpos[cam_id].copy()
    pos_cam = R_w_cam.T @ (t_w_site - t_w_cam)
    R_c_site = R_w_cam.T @ R_w_site
    quat_cam = spatial.transform.Rotation.from_matrix(R_c_site).as_quat(scalar_first=True)
    return pos_cam, quat_cam


def render_camera(env: CubeBaseEnvironment, camera_name: str):
    env.renderer.update_scene(env.data, camera=camera_name)
    return env.renderer.render()


def get_next_index(root: pathlib.Path) -> int:
    idxs = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(idxs) + 1 if idxs else 1


def randomize_ee_initial_qpos(env: CubeBaseEnvironment, rng: np.random.Generator,
                              active_arm: control.Arm, max_joint_delta_rad=0.03):
    """Perturb the active arm's 7 joints so episode starts are not identical."""
    try:
        if active_arm == control.Arm.LEFT:
            names = [f"left_kinova_arm_joint_{i}" for i in range(1, 8)]
        else:
            names = [f"right_kinova_arm_joint_{i}" for i in range(1, 8)]
        qidxs = [
            env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in names
        ]
        delta = rng.normal(scale=max_joint_delta_rad, size=(7,))
        for i, idx in enumerate(qidxs):
            env.data.qpos[idx] += float(delta[i])
        mujoco.mj_forward(env.model, env.data)
        return delta
    except Exception:
        return None


# ---------------------------- MAIN ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2],
                        help="1 = stack two (right arm), 2 = stack three (dual arm)")
    parser.add_argument("--num", type=int, default=1, help="Number of rollouts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--save_video", action="store_true",
                        help="Save mp4 videos and per-frame PNGs (required for GLAM training data)")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Task data root (default: data/stack_two or data/stack_three)")
    args = parser.parse_args()

    task = int(args.task)
    rng = np.random.default_rng(args.seed)

    if task == 1:
        env = TwoCubesEnvironment(seed=args.seed, render_height=480, render_width=640,
                                  ptu_tilt=TASK_PTU_TILT[1])
    else:
        env = ThreeCubesEnvironment(seed=args.seed, render_height=480, render_width=640,
                                    ptu_tilt=TASK_PTU_TILT[2],
                                    calibration_pose=DUAL_ARM_CALIBRATION_POSE)

    static_lift_cost = np.zeros((env.model.nv,))
    lift_idx = env.model.jnt_dofadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ewellix_lift_top_joint")]
    static_lift_cost[lift_idx] = 100.0

    tasks = [
        left_ee_task := mink.FrameTask(
            frame_name="left_ee", frame_type="site",
            position_cost=np.array([2.0, 2.0, 1.0]), orientation_cost=1.0,
        ),
        right_ee_task := mink.FrameTask(
            frame_name="right_ee", frame_type="site",
            position_cost=np.array([2.0, 2.0, 1.0]), orientation_cost=1.0,
        ),
        damping_task := mink.DampingTask(env.model, static_lift_cost),
    ]

    limits = [mink.ConfigurationLimit(model=env.model)]
    configuration = mink.Configuration(env.model, q=env.data.qpos.copy())
    SOLVER = "osqp"

    out_root = pathlib.Path(args.out_root) if args.out_root else \
        pathlib.Path("data") / TASK_DATA_DIRNAME[task]
    task_root = out_root / "frank" / f"task_{task:02d}"
    success_root = task_root / "success"
    failure_root = task_root / "failure"
    success_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)

    success_idx = get_next_index(success_root)
    failure_idx = get_next_index(failure_root)
    print(f"Saving task {task} Frank episodes under: {task_root}")

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
            print(f"\n========== Rollout {ep + 1}/{args.num} (task {task}) ==========")

            obs = env.reset(seed=args.seed + ep)
            ep_rng = np.random.default_rng(args.seed + ep)

            if task == 1:
                reinit_cubes_task1(env, ep_rng,
                                   orange_center=np.array([0.40, 0.13]),
                                   blue_center=np.array([0.26, 0.0]),
                                   perturb=0.05, settle_steps=5)
            else:
                reinit_cubes_task2(env, ep_rng,
                                   blue_center=np.array([0.26, -0.16]),
                                   orange_center=np.array([0.40, 0.00]),
                                   purple_center=np.array([0.26, 0.16]),
                                   perturb=0.05, settle_steps=30)
            obs = env.observation

            left_ee_init = mink.SE3.from_rotation_and_translation(mink.SO3(obs["left_ee_quat"].copy()), obs["left_ee_pos"].copy())
            right_ee_init = mink.SE3.from_rotation_and_translation(mink.SO3(obs["right_ee_quat"].copy()), obs["right_ee_pos"].copy())

            cube_poses = env.get_cube_poses()
            blue_pos0 = cube_poses["blue_pos"].copy()
            blue_quat0 = cube_poses["blue_quat"].copy()
            orange_pos0 = cube_poses["orange_pos"].copy()
            orange_quat0 = cube_poses["orange_quat"].copy()

            # Right arm is always the main (active) arm; left arm only acts in
            # task 2 through the LEFT_* phase machine.
            active_arm, passive_arm = control.Arm.RIGHT, control.Arm.LEFT
            active_ee_quat0 = obs["right_ee_quat"].copy()
            passive_ee_init = left_ee_init

            ee_init_joint_perturb = randomize_ee_initial_qpos(env, rng, active_arm, 0.03)
            if ee_init_joint_perturb is not None:
                obs = env.observation
                left_ee_init = mink.SE3.from_rotation_and_translation(mink.SO3(obs["left_ee_quat"].copy()), obs["left_ee_pos"].copy())
                right_ee_init = mink.SE3.from_rotation_and_translation(mink.SO3(obs["right_ee_quat"].copy()), obs["right_ee_pos"].copy())
                passive_ee_init = left_ee_init

            grasp_quat = control.calculate_cube_grasp_pose(current_quat=active_ee_quat0, target_quat=blue_quat0)
            grasp_SO3 = mink.SO3(grasp_quat)
            approach_height = blue_pos0[2] + 0.1
            grasp_height = blue_pos0[2] + control.GRASP_HEIGHT

            blue_xy = blue_pos0[:2]
            approach_pose = mink.SE3.from_rotation_and_translation(grasp_SO3, np.array([blue_xy[0], blue_xy[1], approach_height]))
            grasp_pose = mink.SE3.from_rotation_and_translation(grasp_SO3, np.array([blue_xy[0], blue_xy[1], grasp_height]))

            phase = ALIGN
            prev_error = 1e6
            finger_cmd = FINGER_OPEN

            # Task 2 LEFT-arm parallel state. Trigger schedule:
            #   right LIFT wp 0 completes -> LEFT_IDLE -> LEFT_APPROACH
            #   right RETRACT fully completes -> LEFT_CLOSE -> LEFT_CARRY
            left_phase = LEFT_IDLE
            left_finger_cmd = FINGER_OPEN
            left_grasp_done = False
            left_lift_waypoints = None
            left_lift_waypoint_idx = 0
            left_target_pose = left_ee_init
            left_grasp_tcp_pose = None
            left_purple_pos_at_grasp = None
            left_purple_quat_at_grasp = None
            left_grasp_pose_purple = None
            left_approach_pose_purple = None
            left_pause_counter = 0

            if task == 2:
                purple_pos0 = cube_poses["purple_pos"].copy()
                purple_quat0 = cube_poses["purple_quat"].copy()
                # Pre-compute the left arm's approach + grasp poses for purple
                # (refined again in LEFT_APPROACH from the current purple pose).
                left_ee_quat0 = obs["left_ee_quat"].copy()
                grasp_quat_p = control.calculate_cube_grasp_pose(
                    current_quat=left_ee_quat0, target_quat=purple_quat0)
                grasp_SO3_p = mink.SO3(grasp_quat_p)
                purple_xy = purple_pos0[:2]
                left_approach_pose_purple = mink.SE3.from_rotation_and_translation(
                    grasp_SO3_p, np.array([purple_xy[0], purple_xy[1], purple_pos0[2] + 0.10]))
                left_grasp_pose_purple = mink.SE3.from_rotation_and_translation(
                    grasp_SO3_p, np.array([purple_xy[0], purple_xy[1], purple_pos0[2] + control.GRASP_HEIGHT]))

            overhead_frames, front_frames = [], []
            action_log = []
            blue_world_pose_log, blue_over_pose_log, blue_front_pose_log = [], [], []
            joint_pos_log, joint_vel_log = [], []
            frank_ee_state_world, frank_ee_state_over, frank_ee_state_front = [], [], []
            frank_ee_action_executed_world, frank_ee_action_executed_over, frank_ee_action_executed_front = [], [], []

            ep_start = time.time()
            step_count = 0
            pause_counter = 0
            lift_waypoints = None
            lift_waypoint_idx = 0
            cube_pos_at_grasp = None

            with np.errstate(all="ignore"):
                while step_count < MAX_STEPS and phase != DONE:
                    obs_t = env.observation

                    left_pose = mink.SE3.from_rotation_and_translation(mink.SO3(obs_t["left_ee_quat"]), obs_t["left_ee_pos"])
                    right_pose = mink.SE3.from_rotation_and_translation(mink.SO3(obs_t["right_ee_quat"]), obs_t["right_ee_pos"])
                    active_pose = right_pose

                    cube_poses = env.get_cube_poses()
                    blue_pos = cube_poses["blue_pos"].copy()
                    blue_quat = cube_poses["blue_quat"].copy()

                    # ---------- select right-arm target pose ----------
                    if phase == ALIGN:
                        raw_target = approach_pose
                    elif phase == CLOSE:
                        raw_target = grasp_pose
                    elif phase in [LIFT, RETRACT]:
                        if lift_waypoints is not None and lift_waypoint_idx < len(lift_waypoints):
                            raw_target = lift_waypoints[lift_waypoint_idx]
                        else:
                            raw_target = lift_waypoints[-1] if lift_waypoints else grasp_pose
                    elif phase == OPEN:
                        raw_target = lift_waypoints[-1] if lift_waypoints else grasp_pose
                    else:
                        raw_target = grasp_pose

                    # --- Online target smoothing: bound task-space velocity ---
                    max_lin_vel = 0.3   # m/s
                    max_rot_vel = 2.0   # rad/s
                    max_step_pos = max_lin_vel * control.DT
                    max_step_rot = max_rot_vel * control.DT

                    pos_cur = active_pose.wxyz_xyz[4:]
                    pos_tgt = raw_target.wxyz_xyz[4:]
                    quat_cur = active_pose.wxyz_xyz[:4]
                    quat_tgt = raw_target.wxyz_xyz[:4]

                    pos_err = pos_tgt - pos_cur
                    dist = float(np.linalg.norm(pos_err))
                    if dist > max_step_pos:
                        pos_clamped = pos_cur + (pos_err / dist) * max_step_pos
                    else:
                        pos_clamped = pos_tgt

                    rot_cur = spatial.transform.Rotation.from_quat(quat_cur, scalar_first=True)
                    rot_tgt = spatial.transform.Rotation.from_quat(quat_tgt, scalar_first=True)
                    rot_diff = rot_cur.inv() * rot_tgt
                    angle = rot_diff.magnitude()

                    if angle > max_step_rot and angle > 1e-6:
                        slerp = spatial.transform.Slerp([0.0, 1.0], spatial.transform.Rotation.concatenate([rot_cur, rot_tgt]))
                        quat_clamped = slerp([max_step_rot / angle])[0].as_quat(scalar_first=True)
                    else:
                        quat_clamped = quat_tgt

                    target_pose = _make_se3(quat_clamped, pos_clamped)

                    # ---------- Task 2: PARALLEL left-arm phase machine ----------
                    if task == 2:
                        left_pose_cur = left_pose
                        if left_phase == LEFT_IDLE:
                            left_target_pose = left_ee_init
                            left_finger_cmd = FINGER_OPEN

                        elif left_phase == LEFT_APPROACH:
                            left_target_pose = left_approach_pose_purple
                            left_finger_cmd = FINGER_OPEN
                            pos_err_l = float(np.linalg.norm(
                                left_pose_cur.wxyz_xyz[4:] - left_approach_pose_purple.wxyz_xyz[4:]))
                            if pos_err_l < 0.012:
                                # Re-compute grasp pose from purple's CURRENT pose (it
                                # may have shifted during the right arm's motion).
                                cur_cube_poses = env.get_cube_poses()
                                purple_pos_now = cur_cube_poses["purple_pos"].copy()
                                purple_quat_now = cur_cube_poses["purple_quat"].copy()
                                left_ee_quat_now = obs_t["left_ee_quat"].copy()
                                grasp_quat_p2 = control.calculate_cube_grasp_pose(
                                    current_quat=left_ee_quat_now, target_quat=purple_quat_now)
                                grasp_SO3_p2 = mink.SO3(grasp_quat_p2)
                                purple_xy_now = purple_pos_now[:2]
                                left_approach_pose_purple = mink.SE3.from_rotation_and_translation(
                                    grasp_SO3_p2,
                                    np.array([purple_xy_now[0], purple_xy_now[1], purple_pos_now[2] + 0.10]))
                                left_grasp_pose_purple = mink.SE3.from_rotation_and_translation(
                                    grasp_SO3_p2,
                                    np.array([purple_xy_now[0], purple_xy_now[1], purple_pos_now[2] + control.GRASP_HEIGHT]))
                                left_phase = LEFT_PREGRASP
                                print(f"[ep {ep}] LEFT arm: APPROACH done -> PREGRASP")

                        elif left_phase == LEFT_PREGRASP:
                            left_target_pose = left_grasp_pose_purple
                            left_finger_cmd = FINGER_OPEN
                            pos_err_l = float(np.linalg.norm(
                                left_pose_cur.wxyz_xyz[4:] - left_grasp_pose_purple.wxyz_xyz[4:]))
                            if pos_err_l < 0.008:
                                left_phase = LEFT_CLOSE
                                print(f"[ep {ep}] LEFT arm: PREGRASP done -> CLOSE")

                        elif left_phase == LEFT_CLOSE:
                            left_target_pose = left_grasp_pose_purple
                            direction_l = np.sign(FINGER_CLOSED - FINGER_OPEN) or 1.0
                            left_finger_cmd = float(np.clip(
                                left_finger_cmd + direction_l * 0.2,
                                min(FINGER_OPEN, FINGER_CLOSED),
                                max(FINGER_OPEN, FINGER_CLOSED),
                            ))
                            closed_frac_l = (left_finger_cmd - FINGER_OPEN) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)
                            if closed_frac_l >= 0.95:
                                left_finger_cmd = FINGER_CLOSED
                                if not left_grasp_done:
                                    # Capture the grasp state ONCE; the left arm then
                                    # holds purple until the right arm's RETRACT
                                    # (waypoints + pause) has fully completed.
                                    left_grasp_tcp_pose = left_pose_cur
                                    cur_cube_poses = env.get_cube_poses()
                                    left_purple_pos_at_grasp = cur_cube_poses["purple_pos"].copy()
                                    left_purple_quat_at_grasp = cur_cube_poses["purple_quat"].copy()
                                    left_grasp_done = True
                                    print(f"[ep {ep}] LEFT arm: CLOSE done; waiting for right arm RETRACT")

                        elif left_phase == LEFT_CARRY:
                            if left_lift_waypoints is None:
                                # Lazy build using the current blue pose (blue is now
                                # stacked on orange).
                                cur_cube_poses = env.get_cube_poses()
                                blue_target_pos = cur_cube_poses["blue_pos"].copy()
                                blue_target_quat = cur_cube_poses["blue_quat"].copy()
                                left_lift_waypoints = build_stack_waypoints(
                                    left_grasp_tcp_pose,
                                    left_purple_pos_at_grasp,
                                    left_purple_quat_at_grasp,
                                    blue_target_pos,
                                    blue_target_quat,
                                )
                                left_lift_waypoint_idx = 0
                                print(f"[ep {ep}] LEFT arm: CARRY waypoints built ({len(left_lift_waypoints)} wps)")

                            if left_lift_waypoint_idx < len(left_lift_waypoints):
                                left_target_pose = left_lift_waypoints[left_lift_waypoint_idx]
                                pos_err_l = float(np.linalg.norm(
                                    left_pose_cur.wxyz_xyz[4:] - left_target_pose.wxyz_xyz[4:]))
                                quat_err_l = min(
                                    float(np.linalg.norm(left_pose_cur.wxyz_xyz[:4] - left_target_pose.wxyz_xyz[:4])),
                                    float(np.linalg.norm(left_pose_cur.wxyz_xyz[:4] + left_target_pose.wxyz_xyz[:4])),
                                )
                                if pos_err_l < 0.005 and quat_err_l < 0.02:
                                    # WP3 is the alignment WP; pause briefly there.
                                    if left_lift_waypoint_idx == 2 and left_pause_counter < 15:
                                        left_pause_counter += 1
                                    else:
                                        left_pause_counter = 0
                                        left_lift_waypoint_idx += 1
                                        if left_lift_waypoint_idx >= len(left_lift_waypoints):
                                            left_phase = LEFT_OPEN
                                            print(f"[ep {ep}] LEFT arm: CARRY done -> OPEN")
                            left_finger_cmd = FINGER_CLOSED

                        elif left_phase == LEFT_OPEN:
                            left_target_pose = left_lift_waypoints[-1] if left_lift_waypoints else left_ee_init
                            left_finger_cmd = float(np.clip(
                                left_finger_cmd - 0.06,
                                min(FINGER_OPEN, FINGER_CLOSED),
                                max(FINGER_OPEN, FINGER_CLOSED),
                            ))
                            opened_frac_l = (FINGER_CLOSED - left_finger_cmd) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)
                            if opened_frac_l >= 0.95:
                                left_finger_cmd = FINGER_OPEN
                                pos_cur_l = left_pose_cur.wxyz_xyz[4:].copy()
                                quat_cur_l = left_pose_cur.wxyz_xyz[:4].copy()
                                left_lift_waypoints = [_make_se3(quat_cur_l, pos_cur_l + np.array([0.0, 0.0, 0.10], dtype=np.float64))]
                                left_lift_waypoint_idx = 0
                                left_phase = LEFT_RETRACT
                                print(f"[ep {ep}] LEFT arm: OPEN done -> RETRACT")

                        elif left_phase == LEFT_RETRACT:
                            if left_lift_waypoint_idx < len(left_lift_waypoints):
                                left_target_pose = left_lift_waypoints[left_lift_waypoint_idx]
                                pos_err_l = float(np.linalg.norm(
                                    left_pose_cur.wxyz_xyz[4:] - left_target_pose.wxyz_xyz[4:]))
                                if pos_err_l < 0.01:
                                    left_lift_waypoint_idx += 1
                                    if left_lift_waypoint_idx >= len(left_lift_waypoints):
                                        left_phase = LEFT_DONE
                                        print(f"[ep {ep}] LEFT arm: RETRACT done -> LEFT_DONE")
                            left_finger_cmd = FINGER_OPEN

                        elif left_phase == LEFT_DONE:
                            if left_lift_waypoints is not None:
                                left_target_pose = left_lift_waypoints[-1]
                            left_finger_cmd = FINGER_OPEN

                    # ---------- IK target setting ----------
                    if task == 2:
                        # Both arms have independent targets.
                        right_ee_task.set_target(target_pose)
                        left_ee_task.set_target(left_target_pose)
                    else:
                        right_ee_task.set_target(target_pose)
                        left_ee_task.set_target(passive_ee_init)

                    # Completion is computed against the REAL raw target, not the
                    # clamped one.
                    complete, error, info = control.compute_complete(
                        raw_target, active_pose, prev_error,
                        x_threshold=0.01, y_threshold=env.cube_width / 2.0, z_threshold=env.cube_width * 0.75,
                        quat_threshold=0.1, rate_threshold=1e-4,
                    )
                    prev_error = error

                    # ---------- right-arm phase machine ----------
                    if phase == ALIGN and complete:
                        phase = CLOSE
                        print(f"[ep {ep}] Switch to CLOSE (ALIGN done)")
                    elif phase == CLOSE:
                        z_error = info["z_error"]
                        if z_error < 0.01:
                            direction = np.sign(FINGER_CLOSED - FINGER_OPEN) or 1.0
                            finger_cmd = np.clip(
                                finger_cmd + direction * 0.2,
                                min(FINGER_OPEN, FINGER_CLOSED), max(FINGER_OPEN, FINGER_CLOSED),
                            )
                        closed_frac = (finger_cmd - FINGER_OPEN) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)

                        if closed_frac >= 0.95 and complete:
                            finger_cmd = FINGER_CLOSED
                            cube_pos_at_grasp = blue_pos.copy()
                            grasp_tcp_pose = active_pose

                            target_cube_pos = cube_poses["orange_pos"].copy()
                            target_cube_quat = cube_poses["orange_quat"].copy()
                            lift_waypoints = build_stack_waypoints(
                                grasp_tcp_pose, cube_pos_at_grasp, blue_quat,
                                target_cube_pos, target_cube_quat,
                            )
                            lift_waypoint_idx = 0
                            phase = LIFT
                            print(f"[ep {ep}] Switch to LIFT (CLOSE done). Waypoints: {len(lift_waypoints)}")

                    elif phase == LIFT:
                        if lift_waypoints is not None and lift_waypoint_idx < len(lift_waypoints):
                            wp = lift_waypoints[lift_waypoint_idx]
                            pos_err_wp = np.linalg.norm(active_pose.wxyz_xyz[4:] - wp.wxyz_xyz[4:])
                            quat_err_wp = min(
                                np.linalg.norm(active_pose.wxyz_xyz[:4] - wp.wxyz_xyz[:4]),
                                np.linalg.norm(active_pose.wxyz_xyz[:4] + wp.wxyz_xyz[:4]),
                            )
                            if pos_err_wp < 0.002 and quat_err_wp < 0.01:
                                # WP3 is the alignment WP; pause briefly there.
                                is_alignment_wp = (lift_waypoint_idx == 2)
                                if is_alignment_wp and pause_counter < 15:
                                    pause_counter += 1
                                else:
                                    pause_counter = 0
                                    lift_waypoint_idx += 1
                                    # Task 2 trigger: when the right arm finishes LIFT
                                    # wp 0 (lifted blue 10 cm), start the left arm's
                                    # approach to purple in parallel.
                                    if task == 2 and lift_waypoint_idx == 1 and left_phase == LEFT_IDLE:
                                        left_phase = LEFT_APPROACH
                                        print(f"[ep {ep}] RIGHT arm: LIFT wp 0 done -> trigger LEFT arm APPROACH")
                                    if lift_waypoint_idx >= len(lift_waypoints):
                                        phase = OPEN
                                        print(f"[ep {ep}] All LIFT waypoints executed -> Switch to OPEN")
                        else:
                            if complete:
                                phase = DONE
                                print(f"[ep {ep}] LIFT done -> DONE")

                    elif phase == OPEN:
                        finger_cmd = np.clip(
                            finger_cmd - 0.06,
                            min(FINGER_OPEN, FINGER_CLOSED), max(FINGER_OPEN, FINGER_CLOSED),
                        )
                        opened_frac = (FINGER_CLOSED - finger_cmd) / (FINGER_CLOSED - FINGER_OPEN + 1e-8)

                        if opened_frac >= 0.95 and complete:
                            finger_cmd = FINGER_OPEN
                            pos_cur = active_pose.wxyz_xyz[4:].copy()
                            quat_cur = active_pose.wxyz_xyz[:4].copy()
                            wp0 = _make_se3(quat_cur, pos_cur)
                            wp1 = _make_se3(quat_cur, pos_cur + np.array([0.0, 0.0, 0.1], dtype=np.float64))

                            if task == 2:
                                # The right arm must clear the workspace before the
                                # left arm starts stacking: pull it in both -x and -y
                                # (and further up) so it is well out of the way.
                                wp2 = _make_se3(quat_cur, pos_cur + np.array([-0.05, -0.12, 0.15], dtype=np.float64))
                                wp3 = _make_se3(quat_cur, pos_cur + np.array([-0.15, -0.20, 0.15], dtype=np.float64))
                                lift_waypoints = [wp0, wp1, wp2, wp3]
                            else:
                                lift_waypoints = [wp0, wp1]
                            lift_waypoint_idx = 0
                            phase = RETRACT
                            print(f"[ep {ep}] Switch to RETRACT (OPEN done)")

                    elif phase == RETRACT:
                        if lift_waypoints is not None and lift_waypoint_idx < len(lift_waypoints):
                            wp = lift_waypoints[lift_waypoint_idx]
                            pos_err_wp = np.linalg.norm(active_pose.wxyz_xyz[4:] - wp.wxyz_xyz[4:])
                            if pos_err_wp < 0.007:
                                lift_waypoint_idx += 1
                                if lift_waypoint_idx >= len(lift_waypoints):
                                    print(f"[ep {ep}] All RETRACT waypoints executed -> Starting pause")
                        else:
                            if pause_counter < 20:
                                pause_counter += 1
                            else:
                                if task == 2:
                                    # The left arm has been holding purple during the
                                    # right arm's retract. Now that the right arm is
                                    # out of the way, kick off LEFT_CARRY, then wait
                                    # for the left arm to finish before DONE.
                                    if left_phase == LEFT_CLOSE and left_grasp_done:
                                        left_phase = LEFT_CARRY
                                        left_lift_waypoints = None  # lazy build with current blue pose
                                        left_lift_waypoint_idx = 0
                                        print(f"[ep {ep}] RIGHT arm RETRACT done -> LEFT arm CARRY")
                                    if left_phase == LEFT_DONE:
                                        phase = DONE
                                        print(f"[ep {ep}] BOTH arms finished -> DONE")
                                else:
                                    phase = DONE
                                    print(f"[ep {ep}] RETRACT pause done -> DONE")

                    # ---------- Solve IK & build action ----------
                    configuration.update(q=env.data.qpos.copy())
                    try:
                        vel = mink.solve_ik(configuration, tasks, control.DT, SOLVER, damping=1e-3, safety_break=False, limits=limits)
                    except mink.exceptions.NoSolutionFound:
                        print("IK failed; aborting episode.")
                        break

                    qvel_left = vel[env.left_joint_state_to_vel]
                    qvel_right = vel[env.right_joint_state_to_vel]

                    all_joint_pos = np.concatenate([obs_t["left_pos"][:7], obs_t["right_pos"][:7]], axis=0)
                    all_joint_vel = np.concatenate([qvel_left, qvel_right], axis=0)

                    max_norm = control.MAX_VEL_NORM
                    left_norm = np.linalg.norm(all_joint_vel[:7])
                    if left_norm > max_norm and left_norm > 1e-9:
                        all_joint_vel[:7] = all_joint_vel[:7] * (max_norm / left_norm)
                    right_norm = np.linalg.norm(all_joint_vel[7:14])
                    if right_norm > max_norm and right_norm > 1e-9:
                        all_joint_vel[7:14] = all_joint_vel[7:14] * (max_norm / right_norm)

                    new_joint_pos = all_joint_pos + control.DT * all_joint_vel
                    left_joint_cmd = new_joint_pos[:7]
                    right_joint_cmd = new_joint_pos[7:14]

                    if task == 2:
                        left_grip_cmd, right_grip_cmd = left_finger_cmd, finger_cmd
                    else:
                        left_grip_cmd, right_grip_cmd = FINGER_OPEN, finger_cmd

                    action = np.concatenate([left_joint_cmd, right_joint_cmd, np.array([left_grip_cmd, right_grip_cmd])], axis=0)

                    # ---------- logging at time t ----------
                    if args.save_video:
                        overhead_frames.append(render_camera(env, "overhead_cam"))
                        front_frames.append(render_camera(env, "front"))

                    blue_pos_over, blue_quat_over = get_blue_pose_in_camera(env, blue_pos, blue_quat, "overhead_cam")
                    blue_pos_front, blue_quat_front = get_blue_pose_in_camera(env, blue_pos, blue_quat, "front")

                    blue_world_pose_log.append(np.concatenate([blue_pos, blue_quat], axis=-1))
                    blue_over_pose_log.append(np.concatenate([blue_pos_over, blue_quat_over], axis=-1))
                    blue_front_pose_log.append(np.concatenate([blue_pos_front, blue_quat_front], axis=-1))

                    if task == 1:
                        # Single-arm episode: log only the right arm.
                        joint_pos_log.append(obs_t["right_pos"].copy())
                        joint_vel_log.append(obs_t["right_vel"].copy())
                    else:
                        joint_pos_log.append(np.concatenate([obs_t["left_pos"], obs_t["right_pos"]], axis=0))
                        joint_vel_log.append(np.concatenate([obs_t["left_vel"], obs_t["right_vel"]], axis=0))

                    # Frank EE state logging: task 1 logs only the right EE (T, 7);
                    # task 2 logs both arms as [left, right] (T, 14).
                    right_ee_pos, right_ee_quat = obs_t["right_ee_pos"].copy(), obs_t["right_ee_quat"].copy()
                    pos_over_r, quat_over_r = get_site_pose_in_camera(env, "right_ee", "overhead_cam")
                    pos_front_r, quat_front_r = get_site_pose_in_camera(env, "right_ee", "front")
                    if task == 2:
                        left_ee_pos_t, left_ee_quat_t = obs_t["left_ee_pos"].copy(), obs_t["left_ee_quat"].copy()
                        pos_over_l, quat_over_l = get_site_pose_in_camera(env, "left_ee", "overhead_cam")
                        pos_front_l, quat_front_l = get_site_pose_in_camera(env, "left_ee", "front")
                        frank_ee_state_world.append(np.concatenate([
                            left_ee_pos_t, left_ee_quat_t, right_ee_pos, right_ee_quat], axis=-1))
                        frank_ee_state_over.append(np.concatenate([
                            pos_over_l, quat_over_l, pos_over_r, quat_over_r], axis=-1))
                        frank_ee_state_front.append(np.concatenate([
                            pos_front_l, quat_front_l, pos_front_r, quat_front_r], axis=-1))
                    else:
                        frank_ee_state_world.append(np.concatenate([right_ee_pos, right_ee_quat], axis=-1))
                        frank_ee_state_over.append(np.concatenate([pos_over_r, quat_over_r], axis=-1))
                        frank_ee_state_front.append(np.concatenate([pos_front_r, quat_front_r], axis=-1))

                    if task == 1:
                        # DATASET action layout for single-arm episodes:
                        # [r_arm(7), r_grip(1)] sliced from the 16-d ENV action.
                        action_log.append(np.concatenate([action[7:14], action[15:16]]).copy())
                    else:
                        action_log.append(action.copy())

                    obs = env.step(action)

                    try:
                        exec_right_pos, exec_right_quat = obs["right_ee_pos"].copy(), obs["right_ee_quat"].copy()
                        exec_over_pos_r, exec_over_quat_r = get_site_pose_in_camera(env, "right_ee", "overhead_cam")
                        exec_front_pos_r, exec_front_quat_r = get_site_pose_in_camera(env, "right_ee", "front")
                        if task == 2:
                            exec_left_pos, exec_left_quat = obs["left_ee_pos"].copy(), obs["left_ee_quat"].copy()
                            exec_over_pos_l, exec_over_quat_l = get_site_pose_in_camera(env, "left_ee", "overhead_cam")
                            exec_front_pos_l, exec_front_quat_l = get_site_pose_in_camera(env, "left_ee", "front")
                            frank_ee_action_executed_world.append(np.concatenate([
                                exec_left_pos, exec_left_quat, exec_right_pos, exec_right_quat], axis=-1))
                            frank_ee_action_executed_over.append(np.concatenate([
                                exec_over_pos_l, exec_over_quat_l, exec_over_pos_r, exec_over_quat_r], axis=-1))
                            frank_ee_action_executed_front.append(np.concatenate([
                                exec_front_pos_l, exec_front_quat_l, exec_front_pos_r, exec_front_quat_r], axis=-1))
                        else:
                            frank_ee_action_executed_world.append(np.concatenate([exec_right_pos, exec_right_quat], axis=-1))
                            frank_ee_action_executed_over.append(np.concatenate([exec_over_pos_r, exec_over_quat_r], axis=-1))
                            frank_ee_action_executed_front.append(np.concatenate([exec_front_pos_r, exec_front_quat_r], axis=-1))
                    except Exception:
                        pad_size = 14 if task == 2 else 7
                        if len(frank_ee_action_executed_world) > 0:
                            frank_ee_action_executed_world.append(frank_ee_action_executed_world[-1].copy())
                            frank_ee_action_executed_over.append(frank_ee_action_executed_over[-1].copy())
                            frank_ee_action_executed_front.append(frank_ee_action_executed_front[-1].copy())
                        else:
                            frank_ee_action_executed_world.append(np.zeros(pad_size, dtype=float))
                            frank_ee_action_executed_over.append(np.zeros(pad_size, dtype=float))
                            frank_ee_action_executed_front.append(np.zeros(pad_size, dtype=float))

                    step_count += 1
                    if use_viewer:
                        viewer_ctx.sync()

            ep_time = time.time() - ep_start
            print(f"[ep {ep}] Finished after {step_count} steps, final phase = {phase_name.get(phase, 'UNKNOWN')}, episode time = {ep_time:.2f}s")

            cube_poses = env.get_cube_poses()
            blue_pos_final = cube_poses["blue_pos"].copy()
            blue_quat_final = cube_poses["blue_quat"].copy()

            waypoints_executed = False
            if lift_waypoints is not None:
                waypoints_executed = (lift_waypoint_idx >= len(lift_waypoints))

            # ---------- success criterion ----------
            if task == 1:
                orange_pos_final = cube_poses["orange_pos"].copy()
                xy_dist = float(np.linalg.norm(blue_pos_final[:2] - orange_pos_final[:2]))
                z_dist = float(abs(blue_pos_final[2] - orange_pos_final[2]))
                stacked_ok = (xy_dist < 0.02) and (abs(z_dist - 0.055) < 0.02)
                success = (phase == DONE) and waypoints_executed and stacked_ok
                debug_msg = f"task1(stack two): stacked_ok={stacked_ok}, xy_dist={xy_dist:.4f}, z_dist={z_dist:.4f}"
            else:
                # Three cubes stacked vertically: bottom = orange, middle = blue,
                # top = purple. Each layer ~5.5 cm above the previous and
                # xy-aligned within ~2.5 cm.
                orange_pos_final = cube_poses["orange_pos"].copy()
                purple_pos_final = cube_poses["purple_pos"].copy()

                bm_xy = float(np.linalg.norm(blue_pos_final[:2] - orange_pos_final[:2]))
                bm_z = float(blue_pos_final[2] - orange_pos_final[2])
                bm_ok = (bm_xy < 0.025) and (abs(bm_z - 0.055) < 0.025)

                mt_xy = float(np.linalg.norm(purple_pos_final[:2] - blue_pos_final[:2]))
                mt_z = float(purple_pos_final[2] - blue_pos_final[2])
                mt_ok = (mt_xy < 0.025) and (abs(mt_z - 0.055) < 0.025)

                stacked_ok = bm_ok and mt_ok
                success = (phase == DONE) and stacked_ok
                debug_msg = (
                    f"task2(stack three): stacked_ok={stacked_ok}, "
                    f"blue-on-orange xy={bm_xy:.4f} z={bm_z:.4f} ok={bm_ok}, "
                    f"purple-on-blue xy={mt_xy:.4f} z={mt_z:.4f} ok={mt_ok}"
                )

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

            action_log_arr = np.stack(action_log, axis=0)
            joint_pos_arr = np.stack(joint_pos_log, axis=0)
            joint_vel_arr = np.stack(joint_vel_log, axis=0)
            frank_ee_state_world_arr = np.stack(frank_ee_state_world, axis=0)
            frank_ee_state_over_arr = np.stack(frank_ee_state_over, axis=0)
            frank_ee_state_front_arr = np.stack(frank_ee_state_front, axis=0)

            min_len = min(len(action_log), len(frank_ee_action_executed_world))
            ee_log_dim = 14 if task == 2 else 7
            frank_ee_action_executed_world_arr = np.stack(frank_ee_action_executed_world[:min_len], axis=0) if min_len > 0 else np.zeros((0, ee_log_dim))
            frank_ee_action_executed_over_arr = np.stack(frank_ee_action_executed_over[:min_len], axis=0) if min_len > 0 else np.zeros((0, ee_log_dim))
            frank_ee_action_executed_front_arr = np.stack(frank_ee_action_executed_front[:min_len], axis=0) if min_len > 0 else np.zeros((0, ee_log_dim))

            npz_path = trial_dir / "frank_blue_grasp.npz"

            extra_kwargs = {
                "orange_pose_world_init": np.concatenate([orange_pos0, orange_quat0], axis=-1),
            }
            if task == 1:
                # Per-step cube poses, matching the released stack-two episodes.
                extra_kwargs["blue_pose_world"] = np.stack(blue_world_pose_log, axis=0)
                extra_kwargs["blue_pose_overhead"] = np.stack(blue_over_pose_log, axis=0)
                extra_kwargs["blue_pose_front"] = np.stack(blue_front_pose_log, axis=0)
                extra_kwargs["phase_final"] = np.array(phase, dtype=np.int32)
            else:
                orange_pos_final_save = cube_poses["orange_pos"].copy()
                orange_quat_final_save = cube_poses["orange_quat"].copy()
                purple_pos_final_save = cube_poses["purple_pos"].copy()
                purple_quat_final_save = cube_poses["purple_quat"].copy()
                extra_kwargs["orange_pose_world_final"] = np.concatenate([orange_pos_final_save, orange_quat_final_save], axis=-1)
                extra_kwargs["purple_pose_world_init"] = np.concatenate([purple_pos0, purple_quat0], axis=-1)
                extra_kwargs["purple_pose_world_final"] = np.concatenate([purple_pos_final_save, purple_quat_final_save], axis=-1)

            np.savez_compressed(
                npz_path,
                actions=action_log_arr,
                blue_pose_world_init=np.concatenate([blue_pos0, blue_quat0], axis=-1),
                blue_pose_world_final=np.concatenate([blue_pos_final, blue_quat_final], axis=-1),
                success=np.array(success, dtype=bool),
                joint_pos=joint_pos_arr,
                joint_vel=joint_vel_arr,
                frank_ee_state_world=frank_ee_state_world_arr,
                frank_ee_state_overhead=frank_ee_state_over_arr,
                frank_ee_state_front=frank_ee_state_front_arr,
                frank_ee_action_executed_world=frank_ee_action_executed_world_arr,
                frank_ee_action_executed_overhead=frank_ee_action_executed_over_arr,
                frank_ee_action_executed_front=frank_ee_action_executed_front_arr,
                ee_init_joint_perturb=ee_init_joint_perturb,
                task=np.array(task, dtype=np.int32),
                **extra_kwargs,
            )
            print("[ep", ep, "] Saved dataset to", npz_path)

            if args.save_video and len(overhead_frames) > 0:
                min_frames_len = min(len(action_log_arr), len(overhead_frames), len(front_frames))

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

    total_time = time.time() - total_start
    print("\n========== Collection summary ==========")
    print(f"Total episodes: {args.num}")
    print(f"Success: {success_count}, Failure: {failure_count}")
    print(f"Total wall-clock time: {total_time:.2f}s ({total_time / max(args.num, 1):.2f}s per episode)")


if __name__ == "__main__":
    main()
