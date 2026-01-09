import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time


class RoArmVisionEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_steps=300, image_size=(84, 84), use_depth=False):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.image_size = image_size
        self.use_depth = use_depth

        # Camera
        self.camera_width = image_size[0]
        self.camera_height = image_size[1]
        self.camera_fov = 60
        self.camera_near = 0.02
        self.camera_far = 1.0

        self.physics_client = None

        self.robot_urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'robot_files',
            'roarm.urdf'
        )

        # Robot
        self.joint_indices = [0, 1, 2, 3, 4]
        self.gripper_joint_idx = 4
        self.num_joints = 5

        self.joint_lower_limits = np.array([-3.14, -1.8, -1.5, 0.0, 0.0])
        self.joint_upper_limits = np.array([ 3.14,  1.8,  3.14, 1.5, 1.5])

        self.action_space = spaces.Box(
            low=self.joint_lower_limits,
            high=self.joint_upper_limits,
            dtype=np.float32
        )

        image_channels = 4 if use_depth else 3
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                0, 255,
                shape=(image_channels, image_size[0], image_size[1]),
                dtype=np.uint8
            ),
            'proprioception': spaces.Box(
                -np.inf, np.inf,
                shape=(10,),
                dtype=np.float32
            )
        })

        # Success thresholds
        self.success_distance = 0.03
        self.lift_height = 0.07

        self.current_step = 0
        self.closest_distance = float('inf')

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240.)

        p.loadURDF("plane.urdf")

        # Table
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.02])
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.02],
                                        rgbaColor=[0.6, 0.4, 0.2, 1])
        p.createMultiBody(0, table_col, table_vis, [0.25, 0, 0.0])

        # Robot
        self.robot_id = p.loadURDF(
            self.robot_urdf_path,
            [0, 0, 0.02],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )

        self.tcp_link_idx = p.getNumJoints(self.robot_id) - 1

        init_q = [0.0, -0.6, 1.3, 0.6, 0.0]
        for i, j in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j, init_q[i])

        # Pen
        pen_pos = [np.random.uniform(0.22, 0.32),
                   np.random.uniform(-0.1, 0.1),
                   0.025]

        pen_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.005, height=0.15)
        pen_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.005, length=0.15,
                                      rgbaColor=[0, 0, 1, 1])
        self.pen_id = p.createMultiBody(0.05, pen_col, pen_vis,
                                        pen_pos,
                                        p.getQuaternionFromEuler([0, np.pi / 2, 0]))

        for _ in range(50):
            p.stepSimulation()

        self.current_step = 0
        self.closest_distance = float('inf')

        return self._get_observation(), self._get_info()

    # ---------------- STEP ----------------
    def step(self, action):
        for i, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=100,
                maxVelocity=2.0
            )

        p.stepSimulation()

        obs = self._get_observation()
        reward, terminated, info = self._compute_reward()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, info

    # ---------------- OBS ----------------
    def _get_observation(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        q = np.array([s[0] for s in joint_states])
        dq = np.array([s[1] for s in joint_states])

        return {
            'image': self._get_camera_image(),
            'proprioception': np.concatenate([q, dq]).astype(np.float32)
        }

    def _get_camera_image(self):
        link = p.getLinkState(self.robot_id, self.tcp_link_idx, computeForwardKinematics=True)
        pos, orn = link[0], link[1]

        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos = np.array(pos) + R @ np.array([0.03, 0, 0])
        cam_target = cam_pos + R @ np.array([0.15, 0, 0])
        up = R @ np.array([0, 0, 1])

        view = p.computeViewMatrix(cam_pos, cam_target, up)
        proj = p.computeProjectionMatrixFOV(self.camera_fov, 1.0,
                                            self.camera_near, self.camera_far)

        _, _, rgb, _, _ = p.getCameraImage(
            self.camera_width, self.camera_height,
            view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        img = np.array(rgb, dtype=np.uint8).reshape(
            self.camera_height,
            self.camera_width,
            4
        )[:, :, :3]
        return np.transpose(img, (2, 0, 1))

    # ---------------- REWARD ----------------
    def _compute_reward(self):
        grip = p.getLinkState(self.robot_id, self.tcp_link_idx)[0]
        pen = p.getBasePositionAndOrientation(self.pen_id)[0]

        grip = np.array(grip)
        pen = np.array(pen)

        dist = np.linalg.norm(grip - pen)
        self.closest_distance = min(self.closest_distance, dist)

        gripper_pos = p.getJointState(self.robot_id, self.gripper_joint_idx)[0]
        gripper_closed = gripper_pos > 0.8

        contacts = p.getContactPoints(self.robot_id, self.pen_id)
        pen_lifted = pen[2] > self.lift_height
        grasp_success = pen_lifted and gripper_closed and len(contacts) > 0

        reward = -2.0 * dist - 0.1
        terminated = False

        if len(contacts) > 0:
            reward += 100.0
        if gripper_closed and len(contacts) > 0:
            reward += 150.0
        if pen_lifted:
            reward += 300.0
            terminated = True
        if grasp_success:
            reward += 300.0
            terminated = True

        info = {
            'distance_to_pen': float(dist),
            'closest_distance': float(self.closest_distance),
            'success': grasp_success,
            'pen_lifted': pen_lifted
        }

        return reward, terminated, info

    # ---------------- INFO ----------------
    def _get_info(self):
        return {
            'closest_distance': float(self.closest_distance),
            'success': False
        }

    def close(self):
        """Clean up"""
        try:
            if self.physics_client is not None and p.isConnected(self.physics_client):
                p.disconnect(physicsClientId=self.physics_client)
        except Exception as e:
            print(f"Warning: close() failed: {e}")
        finally:
            self.physics_client = None
