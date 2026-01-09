import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time
import cv2


class RoArmVisionEnv(gym.Env):
    """
    Vision-based Gym Environment for RoArm picking a pen using camera feedback.
    
    Key Features:
    - Camera mounted on gripper (active vision)
    - No ground-truth pen position given to agent
    - Image processing for pen detection
    - Enhanced reward shaping
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, max_steps=300, image_size=(84, 84), use_depth=False):
        super(RoArmVisionEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.image_size = image_size
        self.use_depth = use_depth
        
        # Camera parameters
        self.camera_width = image_size[0]
        self.camera_height = image_size[1]
        self.camera_fov = 60
        self.camera_near = 0.02
        self.camera_far = 1.0
        
        # Physics client
        self.physics_client = None
        
        # Load robot and objects
        self.robot_urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'robot_files', 
            'roarm.urdf'
        )
        
        # Robot components
        self.robot_id = None
        self.pen_id = None
        self.plane_id = None
        self.table_id = None
        self.tcp_link_idx = None
        
        # Robot joint info
        self.num_joints = 5
        self.joint_indices = [0, 1, 2, 3, 4]  # Added joint 4 (gripper)
        self.gripper_joint_idx = 4  # Joint index for gripper

        # Joint limits - joint 4 is gripper (0=open, 1.5=closed)
        self.joint_lower_limits = np.array([-3.1416, -1.8, -1.5, 0.0, 0.0])
        self.joint_upper_limits = np.array([3.1416, 1.8, 3.1416, 1.5, 1.5])       
        
         
        # Action space: target joint positions
        self.action_space = spaces.Box(
            low=self.joint_lower_limits,
            high=self.joint_upper_limits,
            dtype=np.float32
        )
        
        # Observation space: RGB image + proprioception
        # Image: (3, H, W) if RGB, (4, H, W) if RGB-D
        image_channels = 4 if use_depth else 3
        
        # Proprioception: 8 values (4 joint positions + 4 joint velocities)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, 
                high=255, 
                shape=(image_channels, image_size[0], image_size[1]), 
                dtype=np.uint8
            ),
            'proprioception': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            )
        })
        
        # Workspace bounds for pen spawning
        self.workspace_x = [0.18, 0.38]
        self.workspace_y = [-0.18, 0.18]
        self.workspace_z = [0.05, 0.20]
        
        # Success criteria
        self.success_distance = 0.03   # 3cm
        self.grasp_distance = 0.015    # 1.5cm for close proximity bonus
        self.lift_height = 0.07       # 7cm above table
        
        # Tracking variables
        self.previous_distance = None
        self.closest_distance = float('inf')
        self.pen_initial_pos = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Connect to PyBullet if not connected
        if self.physics_client is None:
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            else:
                self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Reset simulation
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # Load table
        table_pos = [0.25, 0, 0.0]
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.02],
            physicsClientId=self.physics_client
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.02],
            rgbaColor=[0.6, 0.4, 0.2, 1],
            physicsClientId=self.physics_client
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=table_pos,
            physicsClientId=self.physics_client
        )
        p.changeDynamics(
            self.table_id, 
            -1, 
            lateralFriction=1.5,
            physicsClientId=self.physics_client
        )
        
        # Load robot
        robot_start_pos = [0, 0, 0.02]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            self.robot_urdf_path,
            robot_start_pos,
            robot_start_orientation,
            useFixedBase=True,
            physicsClientId=self.physics_client
        )
        
        # Find end-effector link
        self.num_joints_total = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        self.tcp_link_idx = None
        
        for i in range(self.num_joints_total):
            link_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            link_name = link_info[12].decode('utf-8')
            if 'hand_tcp' in link_name.lower():
                self.tcp_link_idx = i
                break
        
        if self.tcp_link_idx is None:
            self.tcp_link_idx = self.num_joints_total - 1
        
        # Set initial joint positions (neutral pose)
        initial_joint_positions = [0.0, -0.6, 1.3, 0.6, 0.0] # Last value = gripper open
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, initial_joint_positions[i], physicsClientId=self.physics_client)
        
        # Curriculum learning: start with pen closer, gradually farther
        # Count total resets to implement curriculum
        if not hasattr(self, 'reset_count'):
            self.reset_count = 0
        self.reset_count += 1

        # Gradually expand workspace based on resets
        if self.reset_count < 200:  # First 200 episodes: very close
            x_range = [0.25, 0.30]
            y_range = [-0.05, 0.05]
        elif self.reset_count < 500:  # Next 300 episodes: medium
            x_range = [0.22, 0.33]
            y_range = [-0.10, 0.10]
        else:  # After 500 episodes: full workspace
            x_range = [0.18, 0.38]
            y_range = [-0.18, 0.18]

        pen_pos = [
            np.random.uniform(0.22,0.32),
            np.random.uniform(-0.10,0.10),
            0.025
        ]
        
        # Random orientation for more challenging scenarios
        # pen_yaw = np.random.uniform(-np.pi, np.pi)
        pen_orientation = p.getQuaternionFromEuler([0, np.pi/2, 0])
        
        # Create pen (cylinder)
        pen_length = 0.15
        pen_radius = 0.005
        pen_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=pen_radius,
            height=pen_length,
            physicsClientId=self.physics_client
        )
        pen_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=pen_radius,
            length=pen_length,
            rgbaColor=[0, 0, 1, 1],  # Blue pen
            physicsClientId=self.physics_client
        )
        self.pen_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=pen_collision_shape,
            baseVisualShapeIndex=pen_visual_shape,
            basePosition=pen_pos,
            baseOrientation=pen_orientation,
            physicsClientId=self.physics_client
        )
        p.changeDynamics(
            self.pen_id, 
            -1, 
            lateralFriction=1.5,
            spinningFriction=0.5,
            rollingFriction=0.03,
            linearDamping=0.8,
            angularDamping=0.8,
            physicsClientId=self.physics_client
        )
        
        # Store initial pen position
        self.pen_initial_pos = np.array(pen_pos)
        
        # Step simulation to stabilize
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # Reset tracking variables
        self.current_step = 0
        self.previous_distance = None
        self.closest_distance = float('inf')
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action (set target joint positions)
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=100,
                maxVelocity=2.0,
                physicsClientId=self.physics_client
            )
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.physics_client)
        if self.render_mode == "human":
            time.sleep(1./240.)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward, terminated, info = self._compute_reward()
        
        # Check truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation: camera image + proprioception"""
        # Get proprioceptive information
        joint_states = p.getJointStates(
            self.robot_id, 
            self.joint_indices,  # Now includes gripper joint!
            physicsClientId=self.physics_client
        )
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        proprioception = np.concatenate([
            joint_positions,    # Now 5 values (includes gripper)
            joint_velocities    # Now 5 values
        ]).astype(np.float32)  # Total: 10 dimensions
        
        # Get camera image from gripper
        camera_image = self._get_camera_image()
        
        return {
            'image': camera_image,
            'proprioception': proprioception
        }
    
    def _get_camera_image(self):
        """Get RGB(-D) image from gripper-mounted camera"""
        # Get gripper pose
        link_state = p.getLinkState(
            self.robot_id, 
            self.tcp_link_idx, 
            computeForwardKinematics=True,
            physicsClientId=self.physics_client
        )
        gripper_pos = link_state[0]
        gripper_orn = link_state[1]
        
        # Calculate camera pose (offset from gripper)
        camera_offset = [0.03, 0, 0]  # 3cm forward from gripper
        rotation_matrix = np.array(p.getMatrixFromQuaternion(gripper_orn)).reshape(3, 3)
        camera_pos = np.array(gripper_pos) + rotation_matrix @ np.array(camera_offset)
        
        # Camera target (looking forward)
        target_offset = [0.15, 0, 0]
        camera_target = camera_pos + rotation_matrix @ np.array(target_offset)
        
        # Camera up vector
        up_vector = rotation_matrix @ np.array([0, 0, 1])
        
        # View and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,
            physicsClientId=self.physics_client
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=1.0,
            nearVal=self.camera_near,
            farVal=self.camera_far,
            physicsClientId=self.physics_client
        )
        
        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_client
        )
    
        # Process RGB image
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        
        # Transpose to (C, H, W) for PyTorch
        rgb_array = np.transpose(rgb_array, (2, 0, 1))
        
        if self.use_depth:
            # Process depth image
            depth_array = np.array(depth_img, dtype=np.float32).reshape(height, width, 1)
            # Convert depth buffer to real depth
            depth_array = self.camera_far * self.camera_near / (
                self.camera_far - (self.camera_far - self.camera_near) * depth_array
            )
            # Normalize depth
            depth_array = (depth_array * 255 / self.camera_far).astype(np.uint8)
            depth_array = np.transpose(depth_array, (2, 0, 1))
            
            # Concatenate RGB and D
            image = np.concatenate([rgb_array, depth_array], axis=0)
        else:
            image = rgb_array
        
        return image
    
    def _check_grasp_success(self):
            """
            Check if pen is successfully grasped and lifted
            
            Returns:
                grasp_success (bool): True if pen is grasped and lifted
                pen_lifted (bool): True if pen is above table
                gripper_closed (bool): True if gripper is closed
                pen_in_gripper (bool): True if pen is touching gripper
            """
            # Get pen position
            pen_pos, _ = p.getBasePositionAndOrientation(
                self.pen_id,
                physicsClientId=self.physics_client
            )
            pen_height = pen_pos[2]
            
            # Get gripper joint state (joint index 3 is the gripper)
            gripper_joint_state = p.getJointState(
                self.robot_id,
                self.gripper_joint_idx,
                physicsClientId=self.physics_client
            )
            gripper_position = gripper_joint_state[0]
            
            # Check if gripper is closed (> 0.8 means mostly closed)
            gripper_closed = gripper_position > 0.8
            
            # Check if pen is lifted (> 7cm from ground)
            pen_lifted = pen_height > 0.07
            
            # Check if pen is in contact with gripper
            contact_points = p.getContactPoints(
                self.robot_id,
                self.pen_id,
                physicsClientId=self.physics_client
            )
            pen_in_gripper = len(contact_points) > 0
            
            # Success = pen lifted AND gripper closed AND pen touching gripper
            grasp_success = pen_lifted and gripper_closed and pen_in_gripper
            
            return grasp_success, pen_lifted, gripper_closed, pen_in_gripper

    def _compute_reward(self):
        """
        Compute reward with enhanced shaping for vision-based learning
        
        Reward components:
        1. Distance reduction reward (dense)
        2. Proximity bonus (sparse)
        3. Success reward (sparse)
        4. Action smoothness penalty
        5. Time penalty
        """
            # Get positions
        link_state = p.getLinkState(
            self.robot_id, 
            self.tcp_link_idx, 
            computeForwardKinematics=True,
            physicsClientId=self.physics_client
        )
        gripper_pos = np.array(link_state[0])
        
        pen_pos, pen_orn = p.getBasePositionAndOrientation(
            self.pen_id, 
            physicsClientId=self.physics_client
        )
        pen_pos = np.array(pen_pos)
        
        # Calculate distance
        distance = np.linalg.norm(gripper_pos - pen_pos)
        if distance < self.closest_distance:
            self.closest_distance = distance

        
        # Get gripper state
        gripper_joint_state = p.getJointState(
            self.robot_id,
            self.gripper_joint_idx,
            physicsClientId=self.physics_client
        )
        gripper_position = gripper_joint_state[0]
        
        # Check grasp status
        grasp_success, pen_lifted, gripper_closed, pen_in_gripper = self._check_grasp_success()
        
        # Initialize
        reward = 0.0
        terminated = False
        info = {
            'distance_to_pen': float(distance),
            'success': False,
            'grasp_success': grasp_success,
            'pen_lifted': pen_lifted,
            'gripper_closed': gripper_closed,
            'pen_in_gripper': pen_in_gripper,
            'pen_fell': False
        }
        
        # === GRASPING REWARD STRUCTURE ===
        
        # Phase 1: Approach (get close to pen)
        reward += -distance * 2.0  # Dense distance penalty
        
        if distance < 0.15:
            reward += 20.0
        if distance < 0.10:
            reward += 30.0
        if distance < 0.05:
            reward += 50.0
        if distance < 0.03:
            reward += 80.0
        
        # Phase 2: Contact (touch the pen)
        if pen_in_gripper:
            reward += 100.0
            
            # Bonus for closing gripper when in contact
            if gripper_closed:
                reward += 150.0
        
        # Phase 3: Lift (raise the pen)
        if pen_lifted:
            reward += 200.0
            
            # Extra bonus if lifted while gripper closed
            if gripper_closed:
                reward += 150.0
            
            terminated = True  # Episode can end if pen is lifted
        
        # Phase 4: SUCCESS (complete grasp)
        if grasp_success:
            reward += 300.0  # HUGE SUCCESS REWARD!
            terminated = True
            info['success'] = True
            return reward, terminated, info
        
        # === PENALTIES ===
        
        # Time penalty (encourages efficiency)
        reward -= 0.1
        
        # Pen fell off table
        if pen_pos[2] < 0.0:
            reward -= 200.0
            terminated = True
            info['pen_fell'] = True
        
        # Gripper collision with table
        if gripper_pos[2] < 0.01:
            reward -= 10.0
        
        # Closing gripper when far from pen (wasteful)
        if distance > 0.10 and gripper_closed:
            reward -= 5.0
        
        # Out of workspace
        if not (0.10 < gripper_pos[0] < 0.45):
            reward -= 5.0
        if not (-0.25 < gripper_pos[1] < 0.25):
            reward -= 5.0
        
        return reward, terminated, info
    
    def _get_info(self):
        """Get additional info for logging"""
        # Get current distance
        link_state = p.getLinkState(
            self.robot_id, 
            self.tcp_link_idx, 
            computeForwardKinematics=True,
            physicsClientId=self.physics_client
        )
        gripper_pos = np.array(link_state[0])
        
        pen_pos, _ = p.getBasePositionAndOrientation(
            self.pen_id, 
            physicsClientId=self.physics_client
        )
        pen_pos = np.array(pen_pos)
        
        distance = np.linalg.norm(gripper_pos - pen_pos)
        
        return {
            'distance_to_pen': float(distance),
            'success': distance < self.success_distance,
            'closest_distance': float(self.closest_distance)
        }
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            pass
        return None
    
    def close(self):
        """Clean up"""
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None


if __name__ == "__main__":
    # Test the environment
    print("Testing RoArm Vision Environment...")
    env = RoArmVisionEnv(render_mode="human", use_depth=False)
    
    obs, info = env.reset()
    print(f"Image shape: {obs['image'].shape}")
    print(f"Proprioception shape: {obs['proprioception'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial distance to pen: {info['distance_to_pen']:.3f}m")
    
    # Run random actions
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Reward={reward:.2f}, Distance={info['distance_to_pen']:.3f}m")
        
        if terminated or truncated:
            print(f"Episode finished at step {i}")
            print(f"Success: {info.get('success', False)}")
            obs, info = env.reset()
    
    env.close()
    print("Environment test complete!")
