import cv2
import mujoco
import numpy as np
import pathlib
import scipy.spatial as spatial

CTRL_FREQ = 20

GRIPPER_RANGE: float = 0.8
LIFT_HEIGHT: float = 0.4
PTU_TILT: float = -1.0
PTU_PAN: float = 0.0

# Overhead-camera tilt used when the training data was rendered, per task
# (1 = stack two, 2 = stack three). Evaluation scripts pass this to the env
# so the rendered views match the data the models were trained on.
TASK_PTU_TILT = {1: -1.0, 2: -0.9}

# Home pose used for calibration: left arm parked, right arm ready to manipulate.
CALIBRATION_POSE = {
    "ewellix_lift_top_joint": LIFT_HEIGHT,
    "ptu_pan": PTU_PAN,
    "ptu_tilt": PTU_TILT,
    "left_kinova_arm_joint_1": 6.35072423e-01,
    "left_kinova_arm_joint_2": -1.21914089e-02,
    "left_kinova_arm_joint_3": 5.71532831e-01,
    "left_kinova_arm_joint_4": -1.18043698e-01,
    "left_kinova_arm_joint_5": -6.24855581e-03,
    "left_kinova_arm_joint_6": -5.48875718e-02,
    "left_kinova_arm_joint_7": -1.25106400e-03,
    "right_kinova_arm_joint_1": -0.7311,
    "right_kinova_arm_joint_2": 1.2122,
    "right_kinova_arm_joint_3": 0.1129,
    "right_kinova_arm_joint_4": 2.0551,
    "right_kinova_arm_joint_5": -2.2533,
    "right_kinova_arm_joint_6": -1.4109,
    "right_kinova_arm_joint_7": 1.5422,
}

CALIBRATION_ACTIONS = np.zeros((16,))
CALIBRATION_ACTIONS[0:14] = np.asarray(list(CALIBRATION_POSE.values()))[3:]

# Home pose for the dual-arm task (stack three): both arms ready to manipulate.
DUAL_ARM_CALIBRATION_POSE = {
    "ewellix_lift_top_joint": LIFT_HEIGHT,
    "ptu_pan": PTU_PAN,
    "ptu_tilt": PTU_TILT,
    "left_kinova_arm_joint_1": -2.3011,
    "left_kinova_arm_joint_2": -1.2122,
    "left_kinova_arm_joint_3": -0.1129,
    "left_kinova_arm_joint_4": -2.0551,
    "left_kinova_arm_joint_5": 2.2533,
    "left_kinova_arm_joint_6": 1.4109,
    "left_kinova_arm_joint_7": -1.5422,
    "right_kinova_arm_joint_1": -0.7311,
    "right_kinova_arm_joint_2": 1.2122,
    "right_kinova_arm_joint_3": 0.1129,
    "right_kinova_arm_joint_4": 2.0551,
    "right_kinova_arm_joint_5": -2.2533,
    "right_kinova_arm_joint_6": -1.4109,
    "right_kinova_arm_joint_7": 1.5422,
}

DUAL_ARM_CALIBRATION_ACTIONS = np.zeros((16,))
DUAL_ARM_CALIBRATION_ACTIONS[0:14] = np.asarray(list(DUAL_ARM_CALIBRATION_POSE.values()))[3:]


def crop_wrist(image: np.ndarray):
    h, w, c = image.shape
    g = w // 4
    f = h // 2
    image = image[-f:, g:-g, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=[w, h], interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class CubeBaseEnvironment:
    """MuJoCo environment for the dual-arm Frank robot with tabletop cubes.

    Subclasses set `xml_file` and implement `cube_init` / `get_cube_poses`.
    Actions are 16-d ENV layout: [left_arm(7), right_arm(7), left_grip(1), right_grip(1)],
    with gripper commands in [0, 1].
    """

    default_actions = CALIBRATION_ACTIONS

    def __init__(self, seed: int, render_height: int = 144, render_width: int = 256,
                 ptu_tilt: float = PTU_TILT, calibration_pose: dict | None = None):
        self.ptu_tilt = float(ptu_tilt)
        self.model = mujoco.MjModel.from_xml_path(
            str(pathlib.Path(__file__).parents[0] / self.xml_file)
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.dt = 1 / CTRL_FREQ
        self.substeps = int(1 / (CTRL_FREQ * self.model.opt.timestep))

        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        self.data.ctrl[self.model.actuator("lift").id] = LIFT_HEIGHT
        self.data.ctrl[self.model.actuator("ptu_tilt").id] = self.ptu_tilt
        self.data.ctrl[self.model.actuator("ptu_pan").id] = PTU_PAN

        left_joint_names = [f"left_kinova_arm_joint_{i}" for i in range(1, 8)]
        right_joint_names = [f"right_kinova_arm_joint_{i}" for i in range(1, 8)]
        self.dof_ids = np.array([self.model.joint(name).dofadr[0] for name in left_joint_names + right_joint_names])
        self.joint_action_map = np.array(
            [
                self.model.actuator(name).id
                for name in (left_joint_names + right_joint_names)
            ]
        ).astype(np.int32)

        calibration_pose = dict(calibration_pose if calibration_pose is not None else CALIBRATION_POSE)
        calibration_pose["ptu_tilt"] = self.ptu_tilt
        self.calibration_actions = np.zeros((16,))
        self.calibration_actions[0:14] = np.asarray(list(calibration_pose.values()))[3:]
        self.default_actions = self.calibration_actions
        for name, value in calibration_pose.items():
            idx = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            self.data.qpos[idx] = value
            idx = self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            self.data.qvel[idx] = 0.0

        self.left_joint_state_to_pos = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in left_joint_names
            ]
        )
        self.right_joint_state_to_pos = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in right_joint_names
            ]
        )
        self.left_joint_state_to_vel = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in left_joint_names
            ]
        )
        self.right_joint_state_to_vel = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in right_joint_names
            ]
        )
        self.left_gripper_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_kinova_arm_finger_joint"
            )
        ]
        self.right_gripper_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_kinova_arm_finger_joint"
            )
        ]

        self.left_kinova_arm_finger_1_id = self.model.actuator(
            "left_kinova_arm_finger_1"
        ).id
        self.left_kinova_arm_finger_2_id = self.model.actuator(
            "left_kinova_arm_finger_2"
        ).id
        self.right_kinova_arm_finger_1_id = self.model.actuator(
            "right_kinova_arm_finger_1"
        ).id
        self.right_kinova_arm_finger_2_id = self.model.actuator(
            "right_kinova_arm_finger_2"
        ).id

        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cube_geom")
        half_width = self.model.geom_size[cube_geom_id][0]  # 0.0275
        self.cube_width = half_width * 2  # 0.055

        self.cube_init(seed)

        self.calibration_qpos = np.copy(self.data.qpos)

    def name_to_qpos_adr(self, name: str):
        part_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = self.model.body_jntadr[part_id]
        return self.model.jnt_qposadr[joint_id]

    def cube_init(self, seed: int):
        raise NotImplementedError

    def get_cube_poses(self):
        raise NotImplementedError

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.calibration_qpos[:]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            self.step(self.calibration_actions)
        return self.observation

    def step(self, actions: np.ndarray):
        self.data.ctrl[self.model.actuator("lift").id] = LIFT_HEIGHT
        self.data.ctrl[self.model.actuator("ptu_tilt").id] = self.ptu_tilt
        self.data.ctrl[self.model.actuator("ptu_pan").id] = PTU_PAN

        self.data.ctrl[self.joint_action_map] = actions[:14]

        self.data.qfrc_applied[self.dof_ids] = self.data.qfrc_bias[self.dof_ids]

        # actions are [0, 1]
        left_gripper, right_gripper = GRIPPER_RANGE * actions[14:16]
        self.data.ctrl[self.left_kinova_arm_finger_1_id] = left_gripper
        self.data.ctrl[self.left_kinova_arm_finger_2_id] = left_gripper
        self.data.ctrl[self.right_kinova_arm_finger_1_id] = right_gripper
        self.data.ctrl[self.right_kinova_arm_finger_2_id] = right_gripper

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        return self.observation

    def get_site_pose(self, id: int):
        pos = self.data.site_xpos[id]
        mat = self.data.site_xmat[id].reshape(3, 3)
        quat = spatial.transform.Rotation.from_matrix(mat).as_quat(scalar_first=True)
        return pos, quat

    @property
    def observation(self):
        self.renderer.update_scene(self.data, camera="overhead_cam")
        overhead_cam = self.renderer.render()
        self.renderer.update_scene(self.data, camera="left_wrist_cam")
        left_wrist_cam = self.renderer.render()
        self.renderer.update_scene(self.data, camera="right_wrist_cam")
        right_wrist_cam = self.renderer.render()
        self.renderer.update_scene(self.data, camera="back")
        back = self.renderer.render()
        left_wrist_crop = crop_wrist(left_wrist_cam)
        right_wrist_crop = crop_wrist(right_wrist_cam)
        user_camera = np.concatenate((back, overhead_cam), axis=1)
        wrist_camera = np.concatenate((left_wrist_crop, right_wrist_crop), axis=1)
        user_camera = np.concatenate((user_camera, wrist_camera), axis=0)

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        left_gripper_pos = np.clip(
            qpos[self.left_gripper_idx] / GRIPPER_RANGE, a_min=0.0, a_max=1.0
        ) * np.ones((1,))
        right_gripper_pos = np.clip(
            qpos[self.right_gripper_idx] / GRIPPER_RANGE, a_min=0.0, a_max=1.0
        ) * np.ones((1,))

        left_ee_pos, left_ee_quat = self.get_site_pose(self.model.site("left_ee").id)
        right_ee_pos, right_ee_quat = self.get_site_pose(self.model.site("right_ee").id)
        overhead_cam_pos, overhead_cam_quat = self.get_site_pose(self.model.site("overhead_cam").id)
        left_cam_pos, left_cam_quat = self.get_site_pose(self.model.site("left_wrist_cam").id)
        right_cam_pos, right_cam_quat = self.get_site_pose(self.model.site("right_wrist_cam").id)

        cube_pose = self.get_cube_poses()
        observations = {
            "left_pos": np.concatenate(
                (qpos[self.left_joint_state_to_pos], left_gripper_pos)
            ),
            "right_pos": np.concatenate(
                (qpos[self.right_joint_state_to_pos], right_gripper_pos)
            ),
            "left_vel": qvel[self.left_joint_state_to_vel],
            "right_vel": qvel[self.right_joint_state_to_pos],
            "left_ee_pos": left_ee_pos.copy(),
            "left_ee_quat": left_ee_quat.copy(),
            "right_ee_pos": right_ee_pos.copy(),
            "right_ee_quat": right_ee_quat.copy(),
            "overhead_cam_pos": overhead_cam_pos.copy(),
            "overhead_cam_quat": overhead_cam_quat.copy(),
            "left_cam_pos": left_cam_pos.copy(),
            "left_cam_quat": left_cam_quat.copy(),
            "right_cam_pos": right_cam_pos.copy(),
            "right_cam_quat": right_cam_quat.copy(),
            "overhead_camera": overhead_cam,
            "left_camera": left_wrist_cam,
            "right_camera": right_wrist_cam,
            "user_camera": user_camera,
            "left_camera_crop": left_wrist_crop,
            "right_camera_crop": right_wrist_crop,
        }
        return {**observations, **cube_pose}

    def close(self):
        pass
