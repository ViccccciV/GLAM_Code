from typing import Any, Dict, Tuple
import enum

import numpy as np
from scipy import spatial

CTRL_HZ = 20
DT = 1.0 / CTRL_HZ
MAX_VEL_NORM = 2.0

GRASP_HEIGHT = 0.01


class Arm(enum.StrEnum):
    LEFT = "left"
    RIGHT = "right"


def calculate_cube_grasp_pose(current_quat: np.ndarray,
                              target_quat: np.ndarray
    ) -> np.ndarray:
    "Simplify target orientation by taking 90deg rotational symmetry into account."
    current_rot = spatial.transform.Rotation.from_quat(current_quat, scalar_first=True)
    target_rot = spatial.transform.Rotation.from_quat(target_quat, scalar_first=True)
    target_z_in_current = target_rot.apply([0, 0, 1])
    relative_rot = current_rot.inv() * target_rot
    rotvec = relative_rot.as_rotvec()
    axis = rotvec / np.linalg.norm(rotvec)
    angle = np.linalg.norm(rotvec)
    # Component of rotation around target's Z-axis [0, 1]
    z_scale = np.dot(axis, target_z_in_current)
    z_angle = angle * z_scale
    # Fold into [-45deg, 45deg] so the shortest symmetric rotation is chosen.
    z_angle_adjusted = (z_angle + np.pi / 4) % (np.pi / 2) - np.pi / 4
    z_delta = z_angle_adjusted - z_angle
    delta_rot = spatial.transform.Rotation.from_rotvec(z_delta * np.array([0, 0, 1]))
    new_target_rot = target_rot * delta_rot
    return new_target_rot.as_quat(scalar_first=True)


def compute_complete(
    target_pose,
    actual_pose,
    previous_error: float,
    x_threshold: float = 0.01,
    y_threshold: float = 0.055 / 2,
    z_threshold: float = 0.055,
    quat_threshold: float = 5e-2,
    rate_threshold: float = 1e-3,
) -> Tuple[bool, float, Dict[str, Any]]:
    """Check whether a mink.SE3 target pose has been reached (per-axis position
    thresholds in the target frame, quaternion distance, and error-rate)."""
    target_quat, target_pos = target_pose.wxyz_xyz[:4], target_pose.wxyz_xyz[4:]
    actual_quat, actual_pos = actual_pose.wxyz_xyz[:4], actual_pose.wxyz_xyz[4:]
    diff_pos = target_pos - actual_pos
    grasp_rot = target_pose.rotation().as_matrix()
    relative_pos_local = grasp_rot.T @ diff_pos
    x_error, y_error, z_error = np.abs(relative_pos_local)
    x_ok = x_error < x_threshold
    y_ok = y_error < y_threshold
    z_ok = z_error < z_threshold
    pos_error = np.linalg.norm(target_pos - actual_pos)
    quat_error = min(
        np.linalg.norm(target_quat - actual_quat),
        np.linalg.norm(target_quat + actual_quat),
    )
    quat_ok = quat_error < quat_threshold
    error = pos_error + quat_error
    rate = abs(previous_error - error)
    rate_ok = rate < rate_threshold
    info = {
        "x_ok": x_ok,
        "x_error": x_error,
        "y_ok": y_ok,
        "y_error": y_error,
        "z_ok": z_ok,
        "z_error": z_error,
        "quat_ok": quat_ok,
        "quat_error": quat_error,
        "rate_ok": rate_ok,
        "rate": rate,
    }
    return rate_ok and x_ok and y_ok and z_ok and quat_ok, error, info
