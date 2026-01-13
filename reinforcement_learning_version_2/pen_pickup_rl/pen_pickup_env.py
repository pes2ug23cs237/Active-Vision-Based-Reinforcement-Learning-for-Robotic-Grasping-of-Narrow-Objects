"""
PyBullet Environment for Pen Pickup Task
The robot starts 3cm from the pen and must learn to pick it up.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import os
import time


class PenPickupEnv(gym.Env):
    """
    Custom Environment for Robot Arm Pen Pickup Task
   
    The robot gripper starts 3cm from a randomly spawned pen.
    Goal: Learn to pick up the pen and lift it above a threshold height.
    """
   
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
   
    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()
       
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
       
        # Connect to PyBullet with unique client per environment
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
       
        # Joint information - will be populated in reset()
        self.num_joints = 4  # base, shoulder, elbow, gripper (total 4 joints)
        self.joint_indices = []
        self.joint_limits = []
       
        # Action space: 4 joint commands total
        # Joints 0-2: arm joints (velocity control)
        # Joint 3: gripper (position control for open/close)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),  # 4 joints total (3 arm + 1 gripper)
            dtype=np.float32
        )
       
        # Observation space:
        # - 4 joint positions
        # - 4 joint velocities
        # - 3 gripper position (x, y, z)
        # - 3 pen position (x, y, z)
        # - 3 relative position (gripper to pen)
        # - 1 gripper state
        # - 1 pen height
        # Total: 19 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),
            dtype=np.float32
        )
       
        # Robot and pen IDs
        self.robot_id = None
        self.pen_id = None
        self.plane_id = None
       
        # Pen pickup parameters
        self.pen_initial_height = 0.05
        self.pickup_height_threshold = 0.15  # Height to lift pen to succeed
        self.distance_from_pen = 0.03  # 3cm initial distance
       
        # Get the path to URDF
        self.urdf_path = os.path.join(os.path.dirname(__file__), "roarm.urdf")
        self.mesh_path = os.path.join(os.path.dirname(__file__), "meshes")
       
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
       
        if seed is not None:
            np.random.seed(seed)
       
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1./240., physicsClientId=self.client)
       
        # Set search path for PyBullet's built-in assets (plane.urdf)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
       
        # Load robot
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
       
        # Change directory to load meshes correctly
        original_dir = os.getcwd()
        urdf_dir = os.path.dirname(self.urdf_path)
        if urdf_dir:
            os.chdir(urdf_dir)
       
        try:
            self.robot_id = p.loadURDF(
                "roarm.urdf",
                robot_start_pos,
                robot_start_orientation,
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
                physicsClientId=self.client
            )
        finally:
            os.chdir(original_dir)

        # Get controllable joint indices and their limits
        self.joint_indices = []
        self.joint_limits = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:  # Only revolute joints
                self.joint_indices.append(i)
                joint_lower = joint_info[8]  # lower limit
                joint_upper = joint_info[9]  # upper limit
                self.joint_limits.append((joint_lower, joint_upper))
           
        # Only print once per unique environment (avoid spam in multi-env training)
        if self.render_mode == "human" or len(self.joint_indices) != 4:
            print(f"[Client {self.client}] Controllable joints: {self.joint_indices}")
            print(f"[Client {self.client}] Joint limits: {self.joint_limits}")
       
        # Ensure we have exactly 4 controllable joints
        if len(self.joint_indices) != 4:
            raise ValueError(f"Expected 4 controllable joints, found {len(self.joint_indices)}")
       
        # Set joint damping
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            p.changeDynamics(self.robot_id, i, linearDamping=0.04, angularDamping=0.04, physicsClientId=self.client)
       
        # Spawn pen at random position
        pen_pos = self._spawn_pen()
       
        # Position robot gripper 3cm from pen
        self._position_robot_near_pen(pen_pos)
       
        # Reset step counter
        self.current_step = 0
       
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
       
        return observation, info
   
    def _spawn_pen(self):
        """Spawn pen at random reachable position"""
        # Random position within robot's workspace
        x = np.random.uniform(0.15, 0.30)
        y = np.random.uniform(-0.15, 0.15)
        z = self.pen_initial_height
       
        pen_pos = [x, y, z]
        pen_orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.random.uniform(0, 2*np.pi)])
       
        # Create pen (cylinder)
        pen_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.004,  # 4mm radius
            height=0.14,    # 14cm length
            physicsClientId=self.client
        )
        pen_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.004,
            length=0.14,
            rgbaColor=[0, 0, 1, 1],  # Blue pen
            physicsClientId=self.client
        )
       
        self.pen_id = p.createMultiBody(
            baseMass=0.01,  # 10 grams
            baseCollisionShapeIndex=pen_collision_shape,
            baseVisualShapeIndex=pen_visual_shape,
            basePosition=pen_pos,
            baseOrientation=pen_orientation,
            physicsClientId=self.client
        )
       
        # Set friction
        p.changeDynamics(self.pen_id, -1,
                        lateralFriction=1.0,
                        spinningFriction=0.005,
                        rollingFriction=0.005,
                        physicsClientId=self.client)
       
        return pen_pos
   
    def _position_robot_near_pen(self, pen_pos):
        """Position robot gripper 3cm from pen using inverse kinematics"""
        # Calculate target gripper position (3cm offset from pen)
        angle = np.random.uniform(0, 2*np.pi)
        offset_x = self.distance_from_pen * np.cos(angle)
        offset_y = self.distance_from_pen * np.sin(angle)
       
        target_pos = [
            pen_pos[0] + offset_x,
            pen_pos[1] + offset_y,
            pen_pos[2] + 0.05  # Slightly above pen
        ]
       
        # Use IK to find joint positions
        # End effector is at hand_tcp link (last link)
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            num_joints - 1,  # hand_tcp link
            target_pos,
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client
        )
       
        # Set joint positions for the controllable joints only
        for i, joint_idx in enumerate(self.joint_indices[:4]):
            if i < len(joint_positions):
                p.resetJointState(self.robot_id, joint_idx, joint_positions[i], physicsClientId=self.client)
       
        # Open gripper initially
        p.resetJointState(self.robot_id, self.joint_indices[3], 0.0, physicsClientId=self.client)
       
        # Let physics settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
   
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
       
        # Apply actions to joints
        # Action: [joint0_vel, joint1_vel, joint2_vel, gripper_pos]
        max_velocity = 2.0
        max_force = 100
       
        # Apply joint velocities to arm joints (first 3 controllable joints)
        for i in range(min(3, len(self.joint_indices))):
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[i],
                p.VELOCITY_CONTROL,
                targetVelocity=action[i] * max_velocity,
                force=max_force,
                physicsClientId=self.client
            )
       
        # Apply gripper control (4th controllable joint)
        # Map action[3] from [-1, 1] to [0, 1.5] (gripper range from URDF)
        # action[3] = -1 -> gripper_target = 0 (closed)
        # action[3] = +1 -> gripper_target = 1.5 (open)
        if len(self.joint_indices) >= 4:
            gripper_target = (action[3] + 1.0) / 2.0 * 1.5
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[3],
                p.POSITION_CONTROL,
                targetPosition=gripper_target,
                force=50,
                physicsClientId=self.client
            )
       
        # Step simulation
        p.stepSimulation(physicsClientId=self.client)
       
        # Get observation
        observation = self._get_observation()
       
        # Calculate reward
        reward = self._calculate_reward()
       
        # Check if episode is done
        terminated = self._check_success()
        truncated = self.current_step >= self.max_steps
       
        info = self._get_info()
       
        if self.render_mode == "human":
            time.sleep(1./240.)
       
        return observation, reward, terminated, truncated, info
   
    def _get_observation(self):
        """Get current observation"""
        # Joint states - use only the controllable joints
        joint_states = p.getJointStates(self.robot_id, self.joint_indices[:4], physicsClientId=self.client)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
       
        # Gripper position (end effector)
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        gripper_state = p.getLinkState(self.robot_id, num_joints - 1, physicsClientId=self.client)
        gripper_pos = np.array(gripper_state[0])
       
        # Pen position
        pen_state = p.getBasePositionAndOrientation(self.pen_id, physicsClientId=self.client)
        pen_pos = np.array(pen_state[0])
       
        # Relative position
        relative_pos = pen_pos - gripper_pos
       
        # Gripper state
        gripper_joint_state = p.getJointState(self.robot_id, self.joint_indices[3], physicsClientId=self.client)
        gripper_opening = np.array([gripper_joint_state[0]])
       
        # Pen height
        pen_height = np.array([pen_pos[2]])
       
        # Concatenate all observations
        observation = np.concatenate([
            joint_positions,      # 4
            joint_velocities,     # 4
            gripper_pos,          # 3
            pen_pos,              # 3
            relative_pos,         # 3
            gripper_opening,      # 1
            pen_height            # 1
        ])
       
        return observation.astype(np.float32)
   
    def _calculate_reward(self):
        """Calculate reward for current state"""
        # Get gripper and pen positions
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        gripper_state = p.getLinkState(self.robot_id, num_joints - 1, physicsClientId=self.client)
        gripper_pos = np.array(gripper_state[0])
       
        pen_state = p.getBasePositionAndOrientation(self.pen_id, physicsClientId=self.client)
        pen_pos = np.array(pen_state[0])
       
        # Distance to pen
        distance = np.linalg.norm(gripper_pos - pen_pos)
       
        # Reward components
        # 1. Distance reward (negative distance)
        distance_reward = -distance * 10.0
       
        # 2. Height reward (encourage lifting pen)
        height_reward = max(0.0, pen_pos[2] - self.pen_initial_height) * 200.0
       
        # 3. Contact reward (check if gripper is touching pen)
        contact_points = p.getContactPoints(self.robot_id, self.pen_id, physicsClientId=self.client)
        contact_reward = 0.0 if len(contact_points) > 0 else 0.0
       
        # 4. Success reward (if pen is lifted above threshold)
        success_reward = 1000.0 if pen_pos[2] > self.pickup_height_threshold else 0.0
       
        # 5. Gripper closing reward when near pen
        if distance < 0.05:
            gripper_state = p.getJointState(self.robot_id, self.joint_indices[3], physicsClientId=self.client)
            gripper_closing = gripper_state[0] / 1.5  # Normalize to [0, 1]
            gripper_reward = gripper_closing * 10.0
        else:
            gripper_reward = 0.0

        
        if distance < 0.03 and pen_pos[2] > self.pen_initial_height + 0.01:
            gripper_reward = gripper_closing * 20.0
        else:
            gripper_reward = 0.0

       
        # Total rewardF
        total_reward = (distance_reward + height_reward + contact_reward +
                       success_reward + gripper_reward)
       
        return total_reward
   
    def _check_success(self):
        """Check if pen pickup is successful"""
        pen_state = p.getBasePositionAndOrientation(self.pen_id, physicsClientId=self.client)
        pen_pos = np.array(pen_state[0])
       
        # Success if pen is lifted above threshold
        return pen_pos[2] > self.pickup_height_threshold
   
    def _get_info(self):
        """Get additional info"""
        pen_state = p.getBasePositionAndOrientation(self.pen_id, physicsClientId=self.client)
        pen_pos = np.array(pen_state[0])
       
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        gripper_state = p.getLinkState(self.robot_id, num_joints - 1, physicsClientId=self.client)
        gripper_pos = np.array(gripper_state[0])
       
        distance = np.linalg.norm(gripper_pos - pen_pos)
       
        return {
            'distance_to_pen': distance,
            'pen_height': pen_pos[2],
            'success': self._check_success()
        }
   
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.3, 0, 0.1],
                distance=0.8,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))
            return rgb_array[:, :, :3]
   
    def close(self):
        """Close the environment"""
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1


if __name__ == "__main__":
    # Test the environment
    env = PenPickupEnv(render_mode="human")
   
    print("Testing environment with random actions...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
   
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
       
        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Distance={info['distance_to_pen']:.4f}m, Height={info['pen_height']:.4f}m")
       
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Success: {info['success']}")
            break
   
    env.close()
