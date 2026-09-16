"""Evaluate a trained policy in the MuJoCo Frank simulation.

Loads a policy checkpoint produced by policy/train.py (MIP baseline or
GLAM-augmented latent policy), rolls it out in the cube-stacking environments,
and reports the success rate.

Tasks: 1 = stack two (right arm), 2 = stack three (dual arm).

Two evaluation protocols:
  --val_indices <ids or file>  replay held-out episodes' initial states
  --use_random_init            randomized initial cube poses, grouped seeds
"""
import argparse
import sys
import os
import math
import csv
from pathlib import Path

import torch
import numpy as np
import imageio
import mujoco
from scipy.spatial.transform import Rotation as R
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bimanual_suite.mjc.cubes import CubeBaseEnvironment, GRIPPER_RANGE, CALIBRATION_ACTIONS, TASK_PTU_TILT
from bimanual_suite.mjc import control

# Policy models live in <repo>/policy; make them importable when running from sim_eval/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from model import PolicyNetwork, GuidedPolicy, VisionPolicyWrapper
from vision_encoder import get_vision_transform


# ==========================================
# Constants & Physics Params
# ==========================================

# Parked left-arm pose for the single-arm task (stack two).
LEFT_ARM_INIT_QPOS = np.array(
    [
        6.35072423e-01,
        -1.21914089e-02,
        5.71532831e-01,
        -1.18043698e-01,
        -6.24855581e-03,
        -5.48875718e-02,
        -1.25106400e-03,
        0.00000000e00,
    ]
)

# Dual-arm home pose for stack three, DATASET layout [l_arm(7), l_grip(1), r_arm(7), r_grip(1)].
DUAL_ARM_INIT_QPOS16 = np.array([
    -2.3011, -1.2122, -0.1129, -2.0551,  2.2533,  1.4109, -1.5422,  0.04,
    -0.7311,  1.2122,  0.1129,  2.0551, -2.2533, -1.4109,  1.5422,  0.04,
], dtype=np.float64)

# Cube centers for randomized initialization (match the data-generation conventions).
STACK_TWO_BLUE_CENTER = np.array([0.26, 0.0], dtype=np.float64)
STACK_TWO_ORANGE_CENTER = np.array([0.40, 0.13], dtype=np.float64)
STACK_THREE_BLUE_CENTER = np.array([0.26, -0.16], dtype=np.float64)
STACK_THREE_ORANGE_CENTER = np.array([0.4, 0.0], dtype=np.float64)
STACK_THREE_PURPLE_CENTER = np.array([0.26, 0.16], dtype=np.float64)

MAX_VEL_NORM = control.MAX_VEL_NORM
MAX_VEL_GRIPPER = 1.0
DT = control.DT

# Camera names in the MuJoCo scene (aligned with the data-generation script).
MUJOCO_CAMERA_NAMES = {"front": "front", "overhead": "overhead_cam"}


# ==========================================
# Helpers & Math
# ==========================================

def render_camera(env, camera_name: str):
    """Render a camera view from MuJoCo. Returns uint8 HWC numpy array."""
    env.renderer.update_scene(env.data, camera=camera_name)
    return env.renderer.render()


def quat_array_to_euler_xyz_deg(quats: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    """Convert (N, 4) quaternions to (N, 3) Euler XYZ in degrees, for plotting only.

    Rotations in this file use scalar_first=True (wxyz), matching MuJoCo's
    convention; the dataset's frank_ee_state_world quaternions follow the same
    convention.
    """
    quats = np.asarray(quats, dtype=np.float64)
    if quats.ndim == 1:
        quats = quats[None, :]
    return R.from_quat(quats, scalar_first=scalar_first).as_euler('xyz', degrees=True)


def plot_comparison(pred_data, gt_data, title, save_path, labels=None):
    if pred_data is None or len(pred_data) == 0:
        return
    pred_data = np.array(pred_data)
    if gt_data is not None:
        gt_data = np.array(gt_data)

    dim = pred_data.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    steps_pred = np.arange(len(pred_data))
    for i in range(dim):
        ax = axes[i]
        if gt_data is not None and len(gt_data) > 0:
            steps_gt = np.arange(len(gt_data))
            ax.plot(steps_gt, gt_data[:, i], label="GT (Expert)", color="black", alpha=0.5, linestyle="--")
        ax.plot(steps_pred, pred_data[:, i], label="Pred (Sim)", color="red", alpha=0.8)
        lbl = labels[i] if labels and i < len(labels) else f"Dim {i}"
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
            ax.set_title(title)

    plt.xlabel("Simulation Steps")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class Normalizer:
    """Mean/std normalizer using the stats stored in the policy checkpoint."""

    def __init__(self, stats=None):
        self.stats = stats if stats is not None else {}
        self.eps = 1e-6

    def normalize(self, key, data):
        if key not in self.stats: return data
        mean, std = self.stats[key]
        return (data - mean) / (std + self.eps)

    def unnormalize(self, key, data):
        if key not in self.stats: return data
        mean, std = self.stats[key]
        return data * (std + self.eps) + mean


def map_server_path_to_local(server_path, local_root):
    """Map a path from a training-server indices file (e.g. val_indices.txt) to the
    local copy of the raw frank episode: <local_root>/frank/.../frank_blue_grasp.npz."""
    parts = Path(server_path).parts
    try:
        frank_idx = parts.index("frank")
    except ValueError:
        return None
    rel_path_parts = list(parts[frank_idx:-1])
    local_dir = Path(local_root).joinpath(*rel_path_parts)
    local_file = local_dir / "frank_blue_grasp.npz"
    return local_file


# ==========================================
# Control helpers
# ==========================================

def apply_velocity_limit(current_qpos, target_qpos, dt_total):
    diff = target_qpos - current_qpos
    raw_vel = diff / max(dt_total, 1e-9)

    v_left = raw_vel[:7]; v_right = raw_vel[7:14]; v_lg = raw_vel[14]; v_rg = raw_vel[15]

    norm_l = np.linalg.norm(v_left)
    if norm_l > MAX_VEL_NORM and norm_l > 1e-9: v_left = v_left * (MAX_VEL_NORM / norm_l)

    norm_r = np.linalg.norm(v_right)
    if norm_r > MAX_VEL_NORM and norm_r > 1e-9: v_right = v_right * (MAX_VEL_NORM / norm_r)

    v_lg = np.clip(v_lg, -MAX_VEL_GRIPPER, MAX_VEL_GRIPPER)
    v_rg = np.clip(v_rg, -MAX_VEL_GRIPPER, MAX_VEL_GRIPPER)

    vel_constrained = np.concatenate([v_left, v_right, [v_lg], [v_rg]])
    return current_qpos + dt_total * vel_constrained


def set_robot_init(env, right_arm_qpos8=None):
    """Single-arm init (stack two): park the left arm, set the right arm."""
    left_j = LEFT_ARM_INIT_QPOS[:7]
    left_g = float(LEFT_ARM_INIT_QPOS[7])

    if right_arm_qpos8 is not None:
        r_q = np.asarray(right_arm_qpos8, dtype=np.float64).reshape(-1)
        right_j = r_q[:7]; right_g = float(r_q[7])
    else:
        current_right_j = env.data.qpos[env.right_joint_state_to_pos]
        current_right_g_idx = env.right_gripper_idx
        current_right_g_val = env.data.qpos[current_right_g_idx] / GRIPPER_RANGE
        right_j = current_right_j; right_g = float(np.mean(current_right_g_val))

    env.data.qpos[env.left_joint_state_to_pos] = left_j
    env.data.qpos[env.left_gripper_idx] = np.clip(left_g, 0.0, 1.0) * GRIPPER_RANGE

    if right_arm_qpos8 is not None:
        env.data.qpos[env.right_joint_state_to_pos] = right_j
        env.data.qpos[env.right_gripper_idx] = np.clip(right_g, 0.0, 1.0) * GRIPPER_RANGE

    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    env.data.ctrl[env.joint_action_map[:7]] = left_j
    env.data.ctrl[env.joint_action_map[7:14]] = right_j

    lg_ctrl = np.clip(left_g, 0.0, 1.0) * GRIPPER_RANGE
    rg_ctrl = np.clip(right_g, 0.0, 1.0) * GRIPPER_RANGE
    env.data.ctrl[env.left_kinova_arm_finger_1_id] = lg_ctrl
    env.data.ctrl[env.left_kinova_arm_finger_2_id] = lg_ctrl
    env.data.ctrl[env.right_kinova_arm_finger_1_id] = rg_ctrl
    env.data.ctrl[env.right_kinova_arm_finger_2_id] = rg_ctrl
    env.data.qfrc_applied[env.dof_ids] = env.data.qfrc_bias[env.dof_ids]


def set_robot_init_dual(env, init_qpos16=None):
    """Dual-arm init (stack three). init_qpos16 is in DATASET layout
    [l_arm(7), l_grip(1), r_arm(7), r_grip(1)] (matches data['joint_pos'][0])."""
    if init_qpos16 is None:
        init_qpos16 = DUAL_ARM_INIT_QPOS16

    q = np.asarray(init_qpos16, dtype=np.float64).reshape(-1)
    if q.shape[0] != 16:
        return  # silently skip if shape wrong

    left_j  = q[0:7]
    left_g  = float(q[7])
    right_j = q[8:15]
    right_g = float(q[15])

    env.data.qpos[env.left_joint_state_to_pos]  = left_j
    env.data.qpos[env.left_gripper_idx]         = np.clip(left_g,  0.0, 1.0) * GRIPPER_RANGE
    env.data.qpos[env.right_joint_state_to_pos] = right_j
    env.data.qpos[env.right_gripper_idx]        = np.clip(right_g, 0.0, 1.0) * GRIPPER_RANGE

    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    env.data.ctrl[env.joint_action_map[:7]]   = left_j
    env.data.ctrl[env.joint_action_map[7:14]] = right_j

    lg_ctrl = np.clip(left_g,  0.0, 1.0) * GRIPPER_RANGE
    rg_ctrl = np.clip(right_g, 0.0, 1.0) * GRIPPER_RANGE
    env.data.ctrl[env.left_kinova_arm_finger_1_id]  = lg_ctrl
    env.data.ctrl[env.left_kinova_arm_finger_2_id]  = lg_ctrl
    env.data.ctrl[env.right_kinova_arm_finger_1_id] = rg_ctrl
    env.data.ctrl[env.right_kinova_arm_finger_2_id] = rg_ctrl
    env.data.qfrc_applied[env.dof_ids] = env.data.qfrc_bias[env.dof_ids]


def warmup_hold_pose(env, steps: int = 10):
    if steps <= 0: return
    obs = env.observation
    left_j = obs["left_pos"][:7]
    right_j = obs["right_pos"][:7]
    left_g = float(obs["left_pos"][7])
    right_g = float(obs["right_pos"][7])
    full_action = np.concatenate([left_j, right_j, [left_g], [right_g]])
    for _ in range(int(steps)):
        env.step(full_action)


def evaluate_state(env, task: int):
    """Success criteria. Task 1: blue stacked on orange. Task 2: blue on orange AND purple on blue."""
    poses = env.get_cube_poses()
    curr_blue_pos = poses["blue_pos"]

    if task == 2:
        curr_orange_pos = poses["orange_pos"]
        curr_purple_pos = poses["purple_pos"]
        bm_xy = float(np.linalg.norm(curr_blue_pos[:2]   - curr_orange_pos[:2]))
        bm_z  = float(curr_blue_pos[2]   - curr_orange_pos[2])
        mt_xy = float(np.linalg.norm(curr_purple_pos[:2] - curr_blue_pos[:2]))
        mt_z  = float(curr_purple_pos[2] - curr_blue_pos[2])
        success = ((bm_xy < 0.025) and (abs(bm_z - 0.055) < 0.025)
                   and (mt_xy < 0.025) and (abs(mt_z - 0.055) < 0.025))
        if success: reason = "Success"
        else: reason = f"BO xy={bm_xy:.3f} z={bm_z:.3f} | PB xy={mt_xy:.3f} z={mt_z:.3f}"
        return success, max(bm_xy, mt_xy), curr_blue_pos[2], reason

    # Task 1: stack blue on orange
    curr_orange_pos = poses["orange_pos"]
    xy_dist = float(np.linalg.norm(curr_blue_pos[:2] - curr_orange_pos[:2]))
    z_dist = float(abs(curr_blue_pos[2] - curr_orange_pos[2]))

    success = (xy_dist < 0.02) and (abs(z_dist - 0.055) < 0.005)
    if success: reason = "Success"
    else: reason = f"XY Dist: {xy_dist:.3f}, Z Dist: {z_dist:.3f}"
    return success, xy_dist, curr_blue_pos[2], reason


# ==========================================
# Environments & randomized initialization
# ==========================================

class TwoCubesEnvironment(CubeBaseEnvironment):
    """Task 1 (stack two): stack blue on orange with the right arm."""
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
    """Task 2 (stack three): stack blue on orange, then purple on blue (dual-arm)."""
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


def reinit_cubes_stack_two(env, rng, orange_center, blue_center, perturb=0.05, settle_steps=5):
    """Task 1: orange and blue each perturbed around their centers, random yaw."""
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


def reinit_cubes_stack_three(env, rng: np.random.Generator,
                             orange_center: np.ndarray, blue_center: np.ndarray, purple_center: np.ndarray,
                             perturb: float = 0.05, settle_steps: int = 5):
    """Task 2: three cubes perturbed independently around their respective centers."""
    blue_qpos = env.blue_qpos
    orange_qpos = env.orange_qpos
    purple_qpos = env.purple_qpos

    x_o = float(rng.uniform(orange_center[0] - perturb, orange_center[0] + perturb))
    y_o = float(rng.uniform(orange_center[1] - perturb, orange_center[1] + perturb))
    x_b = float(rng.uniform(blue_center[0] - perturb, blue_center[0] + perturb))
    y_b = float(rng.uniform(blue_center[1] - perturb, blue_center[1] + perturb))
    x_p = float(rng.uniform(purple_center[0] - perturb, purple_center[0] + perturb))
    y_p = float(rng.uniform(purple_center[1] - perturb, purple_center[1] + perturb))

    quat_b = R.from_euler("z", float(rng.uniform(-np.pi, np.pi))).as_quat()
    quat_b = np.array([quat_b[3], quat_b[0], quat_b[1], quat_b[2]])
    quat_o = R.from_euler("z", float(rng.uniform(-np.pi, np.pi))).as_quat()
    quat_o = np.array([quat_o[3], quat_o[0], quat_o[1], quat_o[2]])
    quat_p = R.from_euler("z", float(rng.uniform(-np.pi, np.pi))).as_quat()
    quat_p = np.array([quat_p[3], quat_p[0], quat_p[1], quat_p[2]])

    env.data.qpos[blue_qpos:   blue_qpos + 3]    = [x_b, y_b, env.data.qpos[blue_qpos + 2]]
    env.data.qpos[blue_qpos + 3:   blue_qpos + 7]    = quat_b
    env.data.qpos[orange_qpos: orange_qpos + 3]  = [x_o, y_o, env.data.qpos[orange_qpos + 2]]
    env.data.qpos[orange_qpos + 3: orange_qpos + 7]  = quat_o
    env.data.qpos[purple_qpos: purple_qpos + 3]  = [x_p, y_p, env.data.qpos[purple_qpos + 2]]
    env.data.qpos[purple_qpos + 3: purple_qpos + 7]  = quat_p

    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(settle_steps)):
        env.step(env.default_actions)


# ==========================================
# Inference
# ==========================================

class PolicyEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.dual_arm = (args.task == 2)

        print(f"Loading checkpoint: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=self.device)
        self.stats = ckpt["stats"]
        self.normalizer = Normalizer(self.stats)
        saved_args = ckpt.get("args", {}) or {}

        self.is_base_policy = bool(saved_args.get("base_policy", False))
        self.use_mip = bool(saved_args.get("use_mip", False))
        self.act_chunk = int(saved_args.get("act_chunk", 1))

        self.camera_views = str(saved_args.get("camera_views", "front"))
        self.vision_feature_dim = int(saved_args.get("vision_feature_dim", 64))
        self.freeze_vision_encoder = bool(saved_args.get("freeze_vision_encoder", True))
        self.image_size = int(saved_args.get("image_size", 224))
        self.vision_backbone = str(saved_args.get("vision_backbone", "dinov2_vits14"))

        self.num_views = 2 if self.camera_views == "both" else 1
        self.img_transform = get_vision_transform(self.image_size)

        self.joint_dim = int(self.stats["obs_joint"][0].shape[0])
        self.action_dim = int(self.stats["action"][0].shape[0])
        input_dim = self.vision_feature_dim + self.joint_dim

        # Latent dim isn't stored in the policy args (it came from the labeled
        # dataset), so infer it from the decoder's first-layer weight shape.
        self.latent_dim = None
        if not self.is_base_policy:
            dec_state = ckpt.get("decoder_state", None)
            if dec_state is None:
                raise ValueError("Latent policy checkpoint has no decoder_state!")
            self.latent_dim = int(dec_state["net.0.weight"].shape[1])

        print(
            f"[Config] base={self.is_base_policy} mip={self.use_mip} act_chunk={self.act_chunk} "
            f"views={self.camera_views} feat_dim={self.vision_feature_dim} backbone={self.vision_backbone} "
            f"joint_dim={self.joint_dim} action_dim={self.action_dim} latent_dim={self.latent_dim} "
            f"input_dim={input_dim}"
        )

        # ---- Build model (mirrors policy/train.py) ----
        if self.is_base_policy:
            inner_model = PolicyNetwork(input_dim, self.action_dim * self.act_chunk,
                                        hidden_dim=256, num_layers=10, use_mip=self.use_mip)
        else:
            inner_model = GuidedPolicy(
                input_dim=input_dim, latent_dim=self.latent_dim, action_dim=self.action_dim,
                act_chunk=self.act_chunk, decoder_path=None, train_decoder=False,
                hidden_dim=256, policy_layers=10, use_mip=self.use_mip,
            )

        self.model = VisionPolicyWrapper(
            inner_model=inner_model,
            vision_feature_dim=self.vision_feature_dim,
            num_views=self.num_views,
            freeze_backbone=self.freeze_vision_encoder,
            image_size=self.image_size,
            vision_backbone=self.vision_backbone,
        ).to(self.device)

        if self.is_base_policy:
            self.model.inner_model.load_state_dict(ckpt["policy_state"])
        else:
            self.model.inner_model.policy.load_state_dict(ckpt["policy_state"])
            self.model.inner_model.decoder.load_state_dict(ckpt["decoder_state"])

        if "vision_encoder_state" in ckpt:
            self.model.vision_encoder.load_state_dict(ckpt["vision_encoder_state"])
            print("[Vision] Loaded vision encoder weights from checkpoint.")
        else:
            print("[Vision] WARNING: No vision_encoder_state in checkpoint! Using default init.")

        self._vision_enc = self.model.vision_encoder
        self._policy_net = self.model.inner_model.policy if not self.is_base_policy else self.model.inner_model
        self._decoder_net = self.model.inner_model.decoder if not self.is_base_policy else None

        self.model.eval()

        if self.dual_arm:
            self.env = ThreeCubesEnvironment(seed=0, render_height=480, render_width=640, ptu_tilt=TASK_PTU_TILT[args.task])
        else:
            self.env = TwoCubesEnvironment(seed=0, render_height=480, render_width=640, ptu_tilt=TASK_PTU_TILT[args.task])

    def _render_current_image(self):
        """
        Render camera image(s) from MuJoCo for the vision encoder.
        Returns a torch tensor:
          - single view:  (1, 3, H, W)
          - both views:   (1, 6, H, W)
        """
        view_list = ["front", "overhead"] if self.camera_views == "both" else [self.camera_views]
        view_tensors = []
        for vname in view_list:
            mj_cam_name = MUJOCO_CAMERA_NAMES[vname]
            img_np = render_camera(self.env, mj_cam_name)       # uint8 HWC (480, 640, 3)
            img_pil = Image.fromarray(img_np).convert("RGB")
            img_t = self.img_transform(img_pil)                 # (3, 224, 224)
            view_tensors.append(img_t)

        if len(view_tensors) == 1:
            img_tensor = view_tensors[0].unsqueeze(0)                  # (1, 3, H, W)
        else:
            img_tensor = torch.cat(view_tensors, dim=0).unsqueeze(0)   # (1, 6, H, W)

        return img_tensor.to(self.device)

    def _extract_joint_obs(self, obs_t):
        """Joint observation in DATASET layout.
        Single-arm: 8-d [r_arm(7), r_grip(1)]. Dual-arm: 16-d [l_arm(7), l_grip(1), r_arm(7), r_grip(1)]."""
        if self.dual_arm:
            left_j  = obs_t["left_pos"][:7].astype(np.float32)
            left_g  = float(obs_t["left_pos"][7])
            right_j = obs_t["right_pos"][:7].astype(np.float32)
            right_g = float(obs_t["right_pos"][7])
            return np.concatenate([left_j, [left_g], right_j, [right_g]]).astype(np.float32)
        return obs_t["right_pos"][:8].astype(np.float32).copy()

    def predict_action_chunk(self, obs_t):
        raw_joint = self._extract_joint_obs(obs_t)
        joint_n = self.normalizer.normalize("obs_joint", raw_joint).astype(np.float32)
        t_joint = torch.from_numpy(joint_n).to(self.device).float().unsqueeze(0)

        with torch.no_grad():
            B = 1
            img_tensor = self._render_current_image()   # (1, C, H, W)
            vis_feat = self._vision_enc(img_tensor)     # (1, vision_feature_dim)

            if self.is_base_policy:
                if self.use_mip:
                    zeros = torch.zeros((B, self.act_chunk * self.action_dim), device=self.device)
                    t0 = torch.zeros((B, 1), device=self.device)
                    pred_1_flat = self._policy_net(vis_feat, t_joint, in_act=zeros, t=t0)

                    in_act_2 = 0.9 * pred_1_flat
                    t1 = torch.ones((B, 1), device=self.device)
                    a_flat = self._policy_net(vis_feat, t_joint, in_act=in_act_2, t=t1)
                else:
                    a_flat = self._policy_net(vis_feat, t_joint)

                a = a_flat.view(B, self.act_chunk, self.action_dim)
            else:
                if self.use_mip:
                    zeros_z = torch.zeros((B, self.act_chunk * self.latent_dim), device=self.device)
                    t0_z = torch.zeros((B, 1), device=self.device)
                    z_pred_1_flat = self._policy_net(vis_feat, t_joint, in_act=zeros_z, t=t0_z)

                    in_act_z_2 = 0.9 * z_pred_1_flat
                    t1_z = torch.ones((B, 1), device=self.device)
                    z_final_flat = self._policy_net(vis_feat, t_joint, in_act=in_act_z_2, t=t1_z)
                else:
                    z_final_flat = self._policy_net(vis_feat, t_joint)

                z_final_step = z_final_flat.view(B * self.act_chunk, self.latent_dim)
                a_final_step = self._decoder_net(z_final_step)
                a = a_final_step.view(B, self.act_chunk, self.action_dim)

        a_np = a.squeeze(0).cpu().numpy()
        a_np = self.normalizer.unnormalize("action", a_np)
        return a_np

    def _load_gt_actions(self, data):
        if "actions" in data:
            gt = data["actions"]
            if gt.ndim == 2 and gt.shape[1] in (8, 16): return gt
        if "joint_pos" in data:
            raw_j = data["joint_pos"]
            if raw_j.ndim == 2 and raw_j.shape[1] == 16:
                if self.dual_arm:
                    # joint_pos is DATASET layout [l_arm,l_grip,r_arm,r_grip] while
                    # env.step wants ENV layout [l_arm,r_arm,l_grip,r_grip]. Convert.
                    out = np.zeros_like(raw_j)
                    out[:, 0:7]   = raw_j[:, 0:7]    # l_arm
                    out[:, 7:14]  = raw_j[:, 8:15]   # r_arm
                    out[:, 14:15] = raw_j[:, 7:8]    # l_grip
                    out[:, 15:16] = raw_j[:, 15:16]  # r_grip
                    return out
                return np.concatenate([raw_j[:, 8:15], raw_j[:, 15:16]], axis=1)
            if raw_j.ndim == 2 and raw_j.shape[1] == 8:
                return raw_j
        return None

    def run_episode(self, init_data_path=None, seed=None):
        data = None
        gt_actions = None

        if self.args.use_random_init:
            if seed is None: seed = 0
            self.env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            if self.dual_arm:
                reinit_cubes_stack_three(self.env, rng,
                                         orange_center=STACK_THREE_ORANGE_CENTER,
                                         blue_center=STACK_THREE_BLUE_CENTER,
                                         purple_center=STACK_THREE_PURPLE_CENTER,
                                         perturb=0.05, settle_steps=30)
                set_robot_init_dual(self.env, init_qpos16=None)
            else:
                reinit_cubes_stack_two(self.env, rng,
                                       orange_center=STACK_TWO_ORANGE_CENTER,
                                       blue_center=STACK_TWO_BLUE_CENTER,
                                       perturb=0.05, settle_steps=5)
                set_robot_init(self.env, right_arm_qpos8=None)
            blue_pos0 = self.env.observation["blue_pos"].copy()
            ep_name = f"random_seed_{seed}"
        else:
            try:
                data = np.load(init_data_path, allow_pickle=True)
            except Exception:
                return False, 0.0, "LoadError", np.zeros(3)

            gt_actions = self._load_gt_actions(data) if self.args.check_gt else None

            blue_pose_init = data["blue_pose_world_init"]
            blue_pos0 = blue_pose_init[:3]
            blue_quat0 = blue_pose_init[3:]

            init_qpos_right = None     # single-arm
            init_qpos16 = None         # dual-arm
            if "joint_pos" in data:
                raw_j0 = data["joint_pos"][0]
                if self.dual_arm:
                    if raw_j0.shape[0] == 16:
                        init_qpos16 = raw_j0
                else:
                    if   raw_j0.shape[0] == 8:  init_qpos_right = raw_j0
                    elif raw_j0.shape[0] == 16: init_qpos_right = np.concatenate([raw_j0[8:15], [raw_j0[15]]])
                    else:                       init_qpos_right = raw_j0[:8]

            self.env.reset(seed=0)
            if self.dual_arm:
                set_robot_init_dual(self.env, init_qpos16=init_qpos16)
            else:
                set_robot_init(self.env, init_qpos_right)

            blue_adr = self.env.blue_qpos if self.env.blue_qpos is not None else self.env.name_to_qpos_adr("blue_cube")
            self.env.data.qpos[blue_adr: blue_adr + 3] = blue_pos0
            self.env.data.qpos[blue_adr + 3: blue_adr + 7] = blue_quat0

            if "orange_pose_world_init" in data:
                orange_pose_init = data["orange_pose_world_init"]
                orange_adr = self.env.orange_qpos if self.env.orange_qpos is not None else self.env.name_to_qpos_adr("orange_cube")
                self.env.data.qpos[orange_adr: orange_adr + 3] = orange_pose_init[:3]
                self.env.data.qpos[orange_adr + 3: orange_adr + 7] = orange_pose_init[3:]

            if self.dual_arm and "purple_pose_world_init" in data:
                purple_pose_init = data["purple_pose_world_init"]
                purple_adr = self.env.purple_qpos if self.env.purple_qpos is not None else self.env.name_to_qpos_adr("purple_cube")
                self.env.data.qpos[purple_adr: purple_adr + 3] = purple_pose_init[:3]
                self.env.data.qpos[purple_adr + 3: purple_adr + 7] = purple_pose_init[3:]

            mujoco.mj_forward(self.env.model, self.env.data)
            ep_name = Path(init_data_path).parent.name

        warmup_hold_pose(self.env, steps=10)

        obs_init = self.env.observation
        fixed_left_j = obs_init["left_pos"][:7].copy()
        fixed_left_g = float(obs_init["left_pos"][7])

        last_action_target = np.concatenate(
            [obs_init["left_pos"][:7], obs_init["right_pos"][:7], [fixed_left_g], [float(obs_init["right_pos"][7])]]
        )

        frames = []
        max_steps = int(self.args.max_steps)

        hist_action, hist_cube_pos, hist_ee_pos = [], [], []
        hist_ee_quat = []  # EE rotation (world frame, scalar-first wxyz)
        final_success = False
        final_dist = 999.0
        final_reason = "TimeOut"

        step = 0

        while step < max_steps:
            obs_t = self.env.observation

            if self.args.check_gt and not self.args.use_random_init:
                if gt_actions is None or len(gt_actions) == 0:
                    return False, 999.0, "NoGTActions", blue_pos0
                if step >= len(gt_actions): break
                K = self.act_chunk
                gt_slice = gt_actions[step: step + K]
                if len(gt_slice) < K:
                    pad = np.repeat(gt_slice[-1][None, :], K - len(gt_slice), axis=0)
                    gt_slice = np.concatenate([gt_slice, pad], axis=0)
                if gt_slice.shape[1] == 16 and not self.dual_arm:
                    act_chunk = np.concatenate([gt_slice[:, 7:14], gt_slice[:, 15:16]], axis=1)
                else:
                    act_chunk = gt_slice.copy()
            else:
                act_chunk = self.predict_action_chunk(obs_t)

            for k in range(min(self.act_chunk, self.args.replan_every)):
                if step >= max_steps: break

                raw_action = act_chunk[k].copy()
                hist_action.append(raw_action.copy())
                hist_cube_pos.append(obs_t["blue_pos"].copy())
                hist_ee_pos.append(obs_t["right_ee_pos"].copy())
                hist_ee_quat.append(obs_t["right_ee_quat"].copy())

                steps_to_run = int(self.args.sim_steps) if int(self.args.sim_steps) > 0 else 1
                curr_obs = self.env.observation

                if self.dual_arm:
                    # Dual-arm: raw_action is 16-d ENV layout
                    # [l_arm(7), r_arm(7), l_grip(1), r_grip(1)] (matches data["actions"]
                    # and what env.step expects directly).
                    target_full_action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
                else:
                    target_j_right = raw_action[:7]
                    target_g_right = float(raw_action[7])
                    target_full_action = np.concatenate([fixed_left_j, target_j_right, [fixed_left_g], [target_g_right]])

                curr_real_state = np.concatenate([
                    curr_obs["left_pos"][:7], curr_obs["right_pos"][:7],
                    [float(curr_obs["left_pos"][7])], [float(curr_obs["right_pos"][7])]
                ])

                if self.args.use_clamp:
                    dt_total = DT * steps_to_run
                    target_clamped = apply_velocity_limit(curr_real_state, target_full_action, dt_total)
                else:
                    target_clamped = target_full_action

                for i in range(1, steps_to_run + 1):
                    alpha = i / steps_to_run
                    interp_cmd = (1 - alpha) * last_action_target + alpha * target_clamped
                    self.env.step(interp_cmd)

                last_action_target = target_clamped

                # success check after executing this step
                is_success, dist, curr_z, reason = evaluate_state(self.env, task=self.args.task)
                final_dist = dist
                final_reason = reason
                if is_success:
                    final_success = True

                # video
                if self.args.save_video and (step % 2 == 0):
                    self.env.renderer.update_scene(self.env.data, camera="front")
                    frames.append(self.env.renderer.render())

                step += 1
                obs_t = self.env.observation

            if self.args.early_stop and final_success:
                break

        # ---- plots ----
        plot_dir = Path(self.args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        gt_plot = None
        if data is not None:
            if "actions" in data: gt_plot = data["actions"]
            elif "joint_pos" in data:
                raw_j = data["joint_pos"]
                if raw_j.ndim == 2 and raw_j.shape[1] == 16:
                    if self.dual_arm:
                        # Convert dataset layout -> env layout, keep 16d
                        out = np.zeros_like(raw_j)
                        out[:, 0:7]   = raw_j[:, 0:7]
                        out[:, 7:14]  = raw_j[:, 8:15]
                        out[:, 14:15] = raw_j[:, 7:8]
                        out[:, 15:16] = raw_j[:, 15:16]
                        gt_plot = out
                    else:
                        gt_plot = np.concatenate([raw_j[:, 8:15], raw_j[:, 15:16]], axis=1)
                elif raw_j.ndim == 2 and raw_j.shape[1] == 8: gt_plot = raw_j

        if gt_plot is not None and gt_plot.ndim == 2 and gt_plot.shape[1] == 16 and not self.dual_arm:
            # Single-arm: convert 16d -> 8d (right arm only)
            gt_plot = np.concatenate([gt_plot[:, 7:14], gt_plot[:, 15:16]], axis=1)

        if self.dual_arm:
            plot_labels = (["L_J0","L_J1","L_J2","L_J3","L_J4","L_J5","L_J6",
                            "R_J0","R_J1","R_J2","R_J3","R_J4","R_J5","R_J6",
                            "L_Grip","R_Grip"])
        else:
            plot_labels = [f"J{i}" for i in range(7)] + ["Grip"]

        plot_comparison(
            hist_action, gt_plot, f"Action: {ep_name}", plot_dir / f"{ep_name}_action.png",
            labels=plot_labels
        )

        # ---- Optional: save raw action arrays for downstream multi-model figures ----
        if self.args.save_actions:
            try:
                pred_arr = np.array(hist_action, dtype=np.float64)
                np.save(plot_dir / f"{ep_name}_pred_actions.npy", pred_arr)
                if gt_plot is not None:
                    gt_arr = np.array(gt_plot, dtype=np.float64)
                    np.save(plot_dir / f"{ep_name}_gt_actions.npy", gt_arr)
                    print(f"   Saved actions: pred {pred_arr.shape} + gt {gt_arr.shape} -> {plot_dir}")
                else:
                    print(f"   Saved actions: pred {pred_arr.shape} (no GT available) -> {plot_dir}")
            except Exception as e:
                print(f"   [WARN] save_actions failed: {e}")

        gt_cube = data["blue_pose_world"][:, :3] if (data is not None and "blue_pose_world" in data) else None
        plot_comparison(hist_cube_pos, gt_cube, f"Cube: {ep_name}", plot_dir / f"{ep_name}_cube.png", labels=["x", "y", "z"])

        gt_ee = data["frank_ee_state_world"] if (data is not None and "frank_ee_state_world" in data) else None

        # ---- Build per-arm GT pos & quat for plotting ----
        # Single-arm: gt_ee = [pos(3), quat(4)] -> indices [0:3], [3:7]
        # Dual-arm:   gt_ee = [l_pos(3), l_quat(4), r_pos(3), r_quat(4)] -> right at [7:10], [10:14]
        gt_ee_pos_arm = None
        gt_ee_quat_arm = None
        if gt_ee is not None and gt_ee.ndim == 2:
            if gt_ee.shape[1] == 14 and self.dual_arm:
                gt_ee_pos_arm  = gt_ee[:, 7:10].astype(np.float64)
                gt_ee_quat_arm = gt_ee[:, 10:14].astype(np.float64)
            elif gt_ee.shape[1] >= 7:
                gt_ee_pos_arm  = gt_ee[:, :3].astype(np.float64)
                gt_ee_quat_arm = gt_ee[:, 3:7].astype(np.float64)
            elif gt_ee.shape[1] >= 3:
                gt_ee_pos_arm  = gt_ee[:, :3].astype(np.float64)
                gt_ee_quat_arm = None  # no rotation info

        # ---- EE plot: combine xyz + Euler-XYZ rotation in one figure ----
        if len(hist_ee_pos) > 0 and len(hist_ee_quat) > 0:
            hist_pos_arr = np.array(hist_ee_pos, dtype=np.float64)
            hist_quat_arr = np.array(hist_ee_quat, dtype=np.float64)
            hist_euler = quat_array_to_euler_xyz_deg(hist_quat_arr, scalar_first=True)
            hist_ee_full = np.concatenate([hist_pos_arr, hist_euler], axis=1)

            if gt_ee_pos_arm is not None and gt_ee_quat_arm is not None:
                gt_euler = quat_array_to_euler_xyz_deg(gt_ee_quat_arm, scalar_first=True)
                gt_ee_full = np.concatenate([gt_ee_pos_arm, gt_euler], axis=1)
            elif gt_ee_pos_arm is not None:
                # No rotation in GT: pad rotation columns with NaN so plot only shows pos
                pad = np.full((gt_ee_pos_arm.shape[0], 3), np.nan)
                gt_ee_full = np.concatenate([gt_ee_pos_arm, pad], axis=1)
            else:
                gt_ee_full = None

            plot_comparison(
                hist_ee_full,
                gt_ee_full,
                f"EE: {ep_name}", plot_dir / f"{ep_name}_ee.png",
                labels=["x", "y", "z", "roll(deg)", "pitch(deg)", "yaw(deg)"],
            )

        if self.args.save_video and len(frames) > 0:
            status_str = "succ" if final_success else "fail"
            vid_name = f"{ep_name}_{status_str}_{final_dist:.3f}.mp4"
            vid_path = Path(self.args.output_dir) / "videos" / vid_name
            vid_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(vid_path, fps=20)
            for f in frames: writer.append_data(f.astype(np.uint8))
            writer.close()

        return final_success, final_dist, final_reason, blue_pos0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=int, default=1, choices=[1, 2],
                        help="1 = stack two (right arm), 2 = stack three (dual arm)")
    parser.add_argument("--val_indices", type=str, nargs='+', default=None,
                        help="Episode IDs (e.g. 0 1 2 3) OR path to a text file with one ID/path per line. "
                             "When IDs are given, the script auto-builds paths under "
                             "<local_data_root>/frank/task_{task:02d}/success/<id_padded>/frank_blue_grasp.npz.")
    parser.add_argument("--local_data_root", type=str, default=None,
                        help="Root of the held-out frank episodes (required with --val_indices).")
    parser.add_argument("--output_dir", type=str, default="inference_results")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500, help="Max rollout steps per episode.")
    parser.add_argument("--sim_steps", type=int, default=1)

    parser.add_argument("--check_gt", action="store_true", help="Ignore policy and rollout using GT actions from dataset (sanity check for env/data alignment).")
    parser.add_argument("--use_clamp", action="store_true", help="Enable velocity clamp for joint targets (recommended for stability).")
    parser.add_argument("--early_stop", action="store_true", help="Stop episode as soon as success condition is met.")
    parser.add_argument("--replan_every", type=int, default=10, help="Predict a chunk, but only execute N steps before predicting again.")
    parser.add_argument("--use_random_init", action="store_true", help="If set, ignore val_indices and initialize cube poses randomly.")
    parser.add_argument("--num_groups", type=int, default=5, help="Random-init protocol: number of seed groups.")
    parser.add_argument("--trials_per_group", type=int, default=10, help="Random-init protocol: trials per seed group.")
    parser.add_argument("--save_actions", action="store_true",
                        help="If set, save per-episode predicted and GT action arrays as .npy "
                             "files next to the plots. Useful for overlaying multiple models' "
                             "trajectories on the same val_indices in a single figure.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for random initialization.")

    args = parser.parse_args()

    if not args.use_random_init and not args.val_indices:
        parser.error("--val_indices is required unless --use_random_init is set.")
    if args.val_indices and not args.use_random_init and not args.local_data_root:
        parser.error("--local_data_root is required with --val_indices.")

    runner = PolicyEvaluator(args)

    success_count = 0
    total_ran = 0

    if args.use_random_init:
        # Randomized-initial-state protocol: num_groups seed groups of
        # trials_per_group episodes each. Reports the overall success rate,
        # a Wilson 95% confidence interval, and mean ± std across groups.
        random_init_logs = []
        group_success_rates = []
        total_successes = 0
        num_groups = int(args.num_groups)
        trials_per_group = int(args.trials_per_group)
        total_trials = num_groups * trials_per_group

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for group_idx in range(num_groups):
            base_seed = args.seed + (group_idx * 100)
            group_successes = 0

            # Each seed group gets its own folder for videos and plots.
            group_dir = output_dir / f"seed_group_{base_seed}"
            print(f"\n=== Starting Seed Group {group_idx+1}/{num_groups} (Base Seed: {base_seed}) ===")

            for trial_idx in range(trials_per_group):
                current_seed = base_seed + trial_idx
                print(f"[{trial_idx + 1}/{trials_per_group}] Running: Random Init (Seed {current_seed})")

                runner.args.output_dir = str(group_dir)

                is_success, dist, reason, init_pos = runner.run_episode(init_data_path=None, seed=current_seed)

                random_init_logs.append({
                    "group_base_seed": base_seed, "trial_seed": current_seed,
                    "init_x": float(init_pos[0]), "init_y": float(init_pos[1]), "init_z": float(init_pos[2]),
                    "success": is_success, "fail_reason": reason, "final_dist": dist
                })

                if is_success:
                    group_successes += 1
                    total_successes += 1

                status = "SUCCESS" if is_success else f"FAILURE ({reason})"
                print(f"   -> {status}")

            group_sr = group_successes / trials_per_group
            group_success_rates.append(group_sr)
            print(f"--- Seed Group {base_seed} Success Rate: {group_successes}/{trials_per_group} ({group_sr*100:.1f}%) ---")

        runner.args.output_dir = str(output_dir)

        group_sr_arr = np.array(group_success_rates)
        avg_sr = np.mean(group_sr_arr) * 100.0
        std_sr = np.std(group_sr_arr) * 100.0

        # Wilson score interval (95% confidence, z ~ 1.96)
        n = total_trials
        p = total_successes / n
        z = 1.96
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        lower_bound = (center - half_width) * 100.0
        upper_bound = (center + half_width) * 100.0
        total_sr = p * 100.0

        print("\n" + "="*50)
        print(f"FINAL EVALUATION RESULTS (Task {args.task})")
        print("="*50)
        print(f"Total Trials: {total_trials}")
        print(f"Total Successes: {total_successes}")
        print(f"Total Success Rate: {total_sr:.1f}%")
        print(f"Wilson Score Interval (95%): [{lower_bound:.1f}%, {upper_bound:.1f}%]")
        print(f"Mean across {num_groups} seed groups: {avg_sr:.1f}% ± {std_sr:.1f}%")
        print("="*50 + "\n")

        csv_path = output_dir / "random_init_summary.csv"
        fieldnames = ["group_base_seed", "trial_seed", "init_x", "init_y", "init_z", "success", "fail_reason", "final_dist"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(random_init_logs)
            f.write("\n")
            f.write(f"Total Success Rate,{total_sr:.1f}%\n")
            f.write(f"Wilson Score (95%),[{lower_bound:.1f}% - {upper_bound:.1f}%]\n")
            f.write(f"Group Mean SR,{avg_sr:.1f}%\n")
            f.write(f"Group Std Dev,{std_sr:.1f}%\n")

        print(f"[Random Init] Summary saved to: {csv_path}")

        success_count = total_successes
        total_ran = total_trials
    else:
        # Build the list of episode npz paths to run.
        # `--val_indices` accepts either:
        #   (a) a list of episode IDs (digits only, e.g. "0 1 2") -> auto-build paths
        #       under <local_data_root>/frank/task_{task:02d}/success/<id_padded>/frank_blue_grasp.npz
        #   (b) a single token that is a file path -> read it, each line is a path
        episode_paths = []
        local_root = Path(args.local_data_root)
        is_id_list = (
            args.val_indices is not None
            and len(args.val_indices) > 0
            and all(tok.lstrip('-').isdigit() for tok in args.val_indices)
        )

        if is_id_list:
            for tok in args.val_indices:
                ep_id_padded = f"{int(tok):04d}"
                cand = local_root / "frank" / f"task_{int(args.task):02d}" / "success" / ep_id_padded / "frank_blue_grasp.npz"
                if not cand.exists():
                    alt = local_root / f"task_{int(args.task):02d}" / "success" / ep_id_padded / "frank_blue_grasp.npz"
                    if alt.exists(): cand = alt
                if cand.exists():
                    episode_paths.append(cand)
                else:
                    print(f"[Warn] Episode {ep_id_padded} not found under {local_root}")
        else:
            # Treat val_indices as one file path (or list of file paths)
            file_arg = args.val_indices[0] if isinstance(args.val_indices, list) else args.val_indices
            with open(file_arg, "r") as f:
                server_paths = [line.strip() for line in f.readlines() if line.strip()]
            for s_path in server_paths:
                local_path = map_server_path_to_local(s_path, args.local_data_root)
                if local_path is None or not local_path.exists(): continue
                episode_paths.append(local_path)

        episode_logs = []

        for local_path in episode_paths:
            if total_ran >= args.max_episodes: break

            ep_id = local_path.parent.name
            print(f"[{total_ran + 1}/{args.max_episodes}] Running: {ep_id}")
            is_success, dist, reason, _ = runner.run_episode(init_data_path=local_path)

            if is_success: success_count += 1
            total_ran += 1

            status = "SUCCESS" if is_success else f"FAILURE ({reason})"
            print(f"   -> {status}  Dist={dist:.4f}")

            episode_logs.append({
                "episode_id": ep_id,
                "success": bool(is_success),
                "fail_reason": reason,
                "final_dist": float(dist),
            })

        # ---- Save success-rate summary CSV ----
        if len(episode_logs) > 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "val_indices_summary.csv"

            fieldnames = ["episode_id", "success", "fail_reason", "final_dist"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(episode_logs)
                f.write("\n")
                f.write(f"Success Rate,{success_count}/{total_ran}\n")

            print(f"\n[Val Indices] Summary saved to: {csv_path}")

    rate = success_count / max(total_ran, 1) * 100.0
    print(f"\nSuccess Rate: {success_count}/{total_ran} ({rate:.2f}%)")


if __name__ == "__main__":
    main()
