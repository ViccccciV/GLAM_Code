"""Single floating UMI gripper environment and its world-frame control helpers.

The gripper base is a static body repositioned at reset; the 6 slide/hinge
joints (x, y, z, rx, ry, rz) are position-actuated, and the TCP is the
`umi_ee` site (midpoint between the two fingers).
"""

import mujoco
import numpy as np
from scipy import spatial

# Overhead camera calibration parameters (match the Frank environment).
LIFT_HEIGHT: float = 0.423
PTU_TILT: float = -1.0
PTU_PAN: float = 0.0


def _quat_to_rotmat_scalar_first(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    return spatial.transform.Rotation.from_quat(
        quat_wxyz, scalar_first=True
    ).as_matrix()


def _project_to_xy(vec: np.ndarray) -> np.ndarray:
    """Project a 3D vector onto the XY plane and normalize it."""
    v = vec.copy()
    v[2] = 0.0
    n = np.linalg.norm(v[:2])
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])  # fallback
    v[:2] = v[:2] / n
    v[2] = 0.0
    return v


def _angle_signed_z(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """Signed angle (radians) from vector a to vector b around the world z axis."""
    ax, ay = a_xy[0], a_xy[1]
    bx, by = b_xy[0], b_xy[1]
    cross_z = ax * by - ay * bx
    dot = ax * bx + ay * by
    return float(np.arctan2(cross_z, dot))


def compute_z_align_rot(u0_z_world: np.ndarray,
                        z_target_world: np.ndarray,
                        max_step: float = np.deg2rad(5.0)):
    """
    Compute a small world-frame rotation that aligns the "original UMI z-axis"
    (u0_z_world) with the target world z-axis (z_target_world).

    Returns:
        delta_rot_world: 3D rotation vector (world frame).
        z_ok: whether the angle between the two axes is within tolerance.
    """
    u = u0_z_world / (np.linalg.norm(u0_z_world) + 1e-12)
    t = z_target_world / (np.linalg.norm(z_target_world) + 1e-12)

    axis = np.cross(u, t)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.zeros(3), True  # already aligned
    axis /= norm

    cosang = np.clip(np.dot(u, t), -1.0, 1.0)
    angle = float(np.arccos(cosang))

    angle_step = min(angle, max_step)
    delta_rot = axis * angle_step

    z_ok = angle < np.deg2rad(5.0)
    return delta_rot, z_ok


def compute_yaw_align_rot(u0_x_world: np.ndarray,
                          cube_x_world: np.ndarray,
                          max_step: float = np.deg2rad(5.0)):
    """
    Align the original UMI x-axis in the horizontal plane with the cube's
    x-axis, exploiting the cube's 90-degree symmetry (error folded into
    [-pi/4, pi/4]).

    Returns:
        delta_rot_world: 3D rotation vector around world z-axis.
        yaw_ok: whether the yaw alignment is within tolerance.
    """
    u_xy = _project_to_xy(u0_x_world)
    c_xy = _project_to_xy(cube_x_world)

    angle_err0 = _angle_signed_z(u_xy, c_xy)
    angle_err0 = (angle_err0 + np.pi / 4) % (np.pi / 2) - np.pi / 4

    angle_step = float(np.clip(angle_err0, -max_step, max_step))
    ez = np.array([0.0, 0.0, 1.0])

    delta_rot = ez * angle_step
    yaw_ok = abs(angle_err0) < np.deg2rad(3.0)
    return delta_rot, yaw_ok


def compute_slide_jacobian(env):
    """
    Jacobian of the 3 slide joints in world coordinates: world_delta ~= J @ dq.

    Returns:
        J: 3x3 matrix, columns are world-space joint axes.
        J_inv: inverse of J (assuming full rank).
    """
    mujoco.mj_forward(env.model, env.data)

    joint_names = ["gripper_joint_x", "gripper_joint_y", "gripper_joint_z"]
    cols = []
    for name in joint_names:
        jid = env.model.joint(name).id
        bodyid = env.model.jnt_bodyid[jid]
        R_body = env.data.xmat[bodyid].reshape(3, 3)
        axis_local = env.model.jnt_axis[jid].copy()
        cols.append(R_body @ axis_local)

    J = np.column_stack(cols)
    J_inv = np.linalg.inv(J)
    return J, J_inv


def compute_rot_jacobian(env):
    """
    Jacobian of the 3 hinge joints (rx, ry, rz) in world coordinates:
    world_omega ~= J_rot @ dq_rot.

    Returns:
        J_rot: 3x3 matrix, columns are world-space joint axes.
        J_rot_inv: inverse of J_rot (assuming full rank).
    """
    mujoco.mj_forward(env.model, env.data)

    joint_names = ["gripper_joint_rx", "gripper_joint_ry", "gripper_joint_rz"]
    cols = []
    for name in joint_names:
        jid = env.model.joint(name).id
        bodyid = env.model.jnt_bodyid[jid]
        R_body = env.data.xmat[bodyid].reshape(3, 3)
        axis_local = env.model.jnt_axis[jid].copy()
        cols.append(R_body @ axis_local)

    J_rot = np.column_stack(cols)
    J_rot_inv = np.linalg.inv(J_rot)
    return J_rot, J_rot_inv


def apply_world_delta_to_qcmd(delta_world, q_cmd_local,
                              J_inv_local, max_step=0.005):
    """
    Map a world-frame translation delta to the 3 slide joints via J_inv.

    Args:
        delta_world: 3D translation in world coordinates.
        q_cmd_local: array [gx, gy, gz, rx, ry, rz].
        J_inv_local: inverse of the slide Jacobian.
        max_step: max allowed magnitude for delta_world (rate limiting).

    Returns:
        Updated q_cmd_local with modified [gx, gy, gz].
    """
    dist = np.linalg.norm(delta_world)
    if dist > max_step and dist > 1e-9:
        delta_world = delta_world * (max_step / dist)

    dq = J_inv_local @ delta_world
    q_cmd_local[0:3] += dq
    q_cmd_local[0:3] = np.clip(q_cmd_local[0:3], -1.2, 1.2)
    return q_cmd_local


def apply_world_rot_to_ori_cmd(delta_rot_world, ori_cmd_local,
                               env, max_norm=np.deg2rad(5.0)):
    """
    Map a world-frame rotation vector to the 3 hinge joints via the rotation
    Jacobian (with rate limiting).

    Returns:
        Updated ori_cmd_local ([rx, ry, rz]).
    """
    angle = np.linalg.norm(delta_rot_world)
    if angle > max_norm and angle > 1e-9:
        delta_rot_world = delta_rot_world * (max_norm / angle)

    _, J_rot_inv = compute_rot_jacobian(env)
    ori_cmd_local += J_rot_inv @ delta_rot_world
    return ori_cmd_local


def increase_umi_damping(env, factor=5.0):
    """
    Multiply the damping of the 6 UMI joints by `factor` to reduce overshoot
    and sliding. Modifies model.dof_damping at runtime (not the XML).
    """
    joint_names = [
        "gripper_joint_x",
        "gripper_joint_y",
        "gripper_joint_z",
        "gripper_joint_rx",
        "gripper_joint_ry",
        "gripper_joint_rz",
    ]
    for name in joint_names:
        jid = env.model.joint(name).id
        did = env.model.jnt_dofadr[jid]
        env.model.dof_damping[did] *= factor


# Original UMI frame U0 relative to the umi_ee site.
R_S_TO_U0 = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float64)


class UmiSingleGraspEnv:
    """
    TCP uses the umi_ee site (midpoint between the two fingers).
    """

    def __init__(self, xml_path: str,
                 ctrl_hz: int = 20,
                 render_height: int = 480,
                 render_width: int = 640,
                 ptu_tilt: float = PTU_TILT):
        self.ptu_tilt = float(ptu_tilt)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=render_height, width=render_width
        )

        self.ctrl_hz = ctrl_hz
        self.dt = self.model.opt.timestep
        self.sim_hz = int(1.0 / self.dt)
        self.substeps = int(self.sim_hz // self.ctrl_hz)

        # ---- UMI actuator ids ----
        self.act_gx = self.model.actuator("gripper_joint_x_act").id
        self.act_gy = self.model.actuator("gripper_joint_y_act").id
        self.act_gz = self.model.actuator("gripper_joint_z_act").id
        self.act_rx = self.model.actuator("gripper_joint_rx_act").id
        self.act_ry = self.model.actuator("gripper_joint_ry_act").id
        self.act_rz = self.model.actuator("gripper_joint_rz_act").id
        self.act_fingers = self.model.actuator("fingers_actuator").id

        # ---- UMI joint qpos indices ----
        def joint_qpos_adr(name: str) -> int:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        self.q_gx = joint_qpos_adr("gripper_joint_x")
        self.q_gy = joint_qpos_adr("gripper_joint_y")
        self.q_gz = joint_qpos_adr("gripper_joint_z")
        self.q_rx = joint_qpos_adr("gripper_joint_rx")
        self.q_ry = joint_qpos_adr("gripper_joint_ry")
        self.q_rz = joint_qpos_adr("gripper_joint_rz")

        # ---- blue cube free joint qpos index ----
        def body_qpos_adr(name: str) -> int:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            return self.model.jnt_qposadr[jid]

        self.blue_qpos = body_qpos_adr("blue_cube")

        # ---- cameras ----
        self.cam_overhead = self.model.camera("overhead_cam").id
        self.cam_front = self.model.camera("front").id

        # ---- umi_ee site id ----
        self.site_umi_ee = self.model.site("umi_ee").id
        # Base site is used to find (and reposition) the gripper base body.
        try:
            self.site_gripper_base = self.model.site("gripper_base_site").id
        except Exception:
            self.site_gripper_base = None

        # Disable reflections for speed / consistency
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        self.umi_qcmd = None

        # Set lift + PTU to the calibration pose so overhead_cam matches the
        # camera pose used for the Frank data.
        self._configure_overhead_camera_pose()
        mujoco.mj_forward(self.model, self.data)

    # ----------------- camera calibration helpers -----------------

    def _safe_set_joint_qpos(self, joint_name: str, value: float):
        """Set qpos/qvel for a joint if it exists; otherwise do nothing."""
        try:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        except Exception:
            return
        qadr = self.model.jnt_qposadr[jid]
        dadr = self.model.jnt_dofadr[jid]
        self.data.qpos[qadr] = value
        self.data.qvel[dadr] = 0.0

    def _safe_set_actuator_ctrl(self, act_name: str, value: float):
        """Set ctrl for an actuator if it exists; otherwise do nothing."""
        try:
            aid = self.model.actuator(act_name).id
        except Exception:
            return
        self.data.ctrl[aid] = value

    def _configure_overhead_camera_pose(self):
        self._safe_set_joint_qpos("ewellix_lift_top_joint", LIFT_HEIGHT)
        self._safe_set_joint_qpos("ptu_pan", PTU_PAN)
        self._safe_set_joint_qpos("ptu_tilt", self.ptu_tilt)

        self._safe_set_actuator_ctrl("lift", LIFT_HEIGHT)
        self._safe_set_actuator_ctrl("ptu_pan", PTU_PAN)
        self._safe_set_actuator_ctrl("ptu_tilt", self.ptu_tilt)

    # ----------------- randomize umi base pose -----------------

    def randomize_umi_pose(self, rng: np.random.Generator,
                           pos_center=(0.3, -0.3, 1.3),
                           pos_range=0.05,
                           max_rot_deg=10.0,
                           quat_center=None):
        """
        Randomize the static base body by perturbing position and orientation.
        quat_center (scalar-first, optional) sets the center orientation; when
        omitted the XML default is used.
        """
        if self.site_gripper_base is None:
            return

        try:
            body_id = int(self.model.site[self.site_gripper_base].bodyid)
        except Exception:
            try:
                body_id = int(self.model.site_bodyid[self.site_gripper_base])
            except Exception:
                return

        center = np.asarray(pos_center, dtype=float)
        offset = rng.uniform(-pos_range, pos_range, size=(3,))
        new_pos = center + offset

        small_euler = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, size=3))
        R_pert = spatial.transform.Rotation.from_euler('xyz', small_euler)

        if quat_center is not None:
            qc = np.asarray(quat_center, dtype=float)
            R_base = spatial.transform.Rotation.from_quat([qc[1], qc[2], qc[3], qc[0]])
        else:
            try:
                base_quat = np.asarray(self.model.body_quat[body_id], dtype=float)
                base_quat = base_quat / np.linalg.norm(base_quat)
                R_base = spatial.transform.Rotation.from_quat(
                    [base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
            except Exception:
                R_base = spatial.transform.Rotation.from_quat([0, 0, 0, 1])

        R_new = R_base * R_pert

        q_xyzw = R_new.as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)

        self.model.body_pos[body_id] = new_pos
        self.model.body_quat[body_id] = quat_wxyz

        mujoco.mj_forward(self.model, self.data)

    # ----------------- standard env interface -----------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # After reset, reapply lift + PTU calibration pose.
        self._configure_overhead_camera_pose()
        mujoco.mj_forward(self.model, self.data)

        self.umi_qcmd = np.array([
            self.data.qpos[self.q_gx],
            self.data.qpos[self.q_gy],
            self.data.qpos[self.q_gz],
            self.data.qpos[self.q_rx],
            self.data.qpos[self.q_ry],
            self.data.qpos[self.q_rz],
        ], dtype=np.float64)
        warmup_action = np.zeros(7, dtype=np.float64)
        warmup_action[:6] = self.umi_qcmd
        warmup_action[6] = 0.0

        for _ in range(10):
            self.step(warmup_action)
        return self.get_observation()

    def step(self, action: np.ndarray):
        assert action.shape == (7,)
        gx_cmd, gy_cmd, gz_cmd, rx_cmd, ry_cmd, rz_cmd, f_cmd = action

        self.data.ctrl[self.act_gx] = gx_cmd
        self.data.ctrl[self.act_gy] = gy_cmd
        self.data.ctrl[self.act_gz] = gz_cmd
        self.data.ctrl[self.act_rx] = rx_cmd
        self.data.ctrl[self.act_ry] = ry_cmd
        self.data.ctrl[self.act_rz] = rz_cmd
        self.data.ctrl[self.act_fingers] = f_cmd

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation()

    def render(self, camera_name: str):
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    # ---------- TCP / cube pose helpers ----------

    def get_umi_tcp_pos(self) -> np.ndarray:
        """TCP = umi_ee site position in world frame."""
        return self.data.site_xpos[self.site_umi_ee].copy()

    def get_umi_tcp_quat(self) -> np.ndarray:
        """TCP orientation as quaternion (wxyz) from umi_ee site."""
        R_ws = self.data.site_xmat[self.site_umi_ee].reshape(3, 3)
        return spatial.transform.Rotation.from_matrix(R_ws).as_quat(scalar_first=True)

    def get_blue_pose_world(self):
        """Return (position, quaternion) of the blue cube in the world frame."""
        qidx = self.blue_qpos
        pos = self.data.qpos[qidx: qidx + 3].copy()
        quat = self.data.qpos[qidx + 3: qidx + 7].copy()
        return pos, quat

    def get_blue_pose_in_camera(self, camera_name: str):
        """Return (position, quaternion) of the blue cube in a camera frame."""
        if camera_name == "overhead_cam":
            cam_id = self.cam_overhead
        elif camera_name == "front":
            cam_id = self.cam_front
        else:
            cam_id = self.model.camera(camera_name).id

        pos_w, quat_w = self.get_blue_pose_world()
        R_w_cube = spatial.transform.Rotation.from_quat(
            quat_w, scalar_first=True
        ).as_matrix()

        R_wc = self.data.cam_xmat[cam_id].reshape(3, 3)
        t_wc = self.data.cam_xpos[cam_id]

        pos_cam = R_wc.T @ (pos_w - t_wc)
        R_c_cube = R_wc.T @ R_w_cube
        quat_cam = spatial.transform.Rotation.from_matrix(
            R_c_cube
        ).as_quat(scalar_first=True)

        return pos_cam, quat_cam

    def get_site_pose_in_camera(self, site_id: int, camera_name: str):
        """Return (pos, quat) of a site in the frame of camera_name."""
        if camera_name == "overhead_cam":
            cam_id = self.cam_overhead
        elif camera_name == "front":
            cam_id = self.cam_front
        else:
            cam_id = self.model.camera(camera_name).id

        pos_w = self.data.site_xpos[site_id].copy()
        R_w_site = self.data.site_xmat[site_id].reshape(3, 3)

        R_wc = self.data.cam_xmat[cam_id].reshape(3, 3)
        t_wc = self.data.cam_xpos[cam_id]

        pos_cam = R_wc.T @ (pos_w - t_wc)
        R_c_site = R_wc.T @ R_w_site
        quat_cam = spatial.transform.Rotation.from_matrix(R_c_site).as_quat(scalar_first=True)

        return pos_cam, quat_cam

    def get_observation(self):
        overhead = self.render("overhead_cam")
        front = self.render("front")

        blue_pos_w, blue_quat_w = self.get_blue_pose_world()
        blue_pos_over, blue_quat_over = self.get_blue_pose_in_camera("overhead_cam")
        blue_pos_front, blue_quat_front = self.get_blue_pose_in_camera("front")

        umi_pos_w = self.get_umi_tcp_pos()
        umi_quat_w = self.get_umi_tcp_quat()
        umi_pos_over, umi_quat_over = self.get_site_pose_in_camera(self.site_umi_ee, "overhead_cam")
        umi_pos_front, umi_quat_front = self.get_site_pose_in_camera(self.site_umi_ee, "front")

        obs = {
            "overhead": overhead,
            "front": front,
            "blue_pos_world": blue_pos_w,
            "blue_quat_world": blue_quat_w,
            "blue_pos_overhead": blue_pos_over,
            "blue_quat_overhead": blue_quat_over,
            "blue_pos_front": blue_pos_front,
            "blue_quat_front": blue_quat_front,
            "umi_tcp_pos": umi_pos_w,
            "umi_tcp_quat": umi_quat_w,
            "umi_pos_overhead": umi_pos_over,
            "umi_quat_overhead": umi_quat_over,
            "umi_pos_front": umi_pos_front,
            "umi_quat_front": umi_quat_front,
        }
        return obs

    def close(self):
        pass
