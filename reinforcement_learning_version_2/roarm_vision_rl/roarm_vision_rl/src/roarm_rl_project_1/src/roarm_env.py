import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time


class RoArmPickEnv(gym.Env):
    """
    Custom Gym Environment for RoArm picking a pen using RL.
    State: Ground-truth positions (joint angles, gripper pose, pen pose)
    Action: Joint position control for 4 joints
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, max_steps=250):
        super(RoArmPickEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Physics client will be created in reset()
        self.physics_client = None
        
        # Load robot and objects
        self.robot_urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'robot_files', 
            'roarm.urdf'
        )
        
        # Will be set in reset()
        self.robot_id = None
        self.pen_id = None
        self.plane_id = None
        self.tcp_link_idx = None
        
        # Robot joint info (4 controllable joints)
        self.num_joints = 4
        self.joint_indices = [0, 1, 2, 3]
        
        # Joint limits (from URDF)
        self.joint_lower_limits = np.array([-3.1416, -1.5708, -1.0, 0.0])
        self.joint_upper_limits = np.array([3.1416, 1.5708, 3.1416, 1.5])
        
        # Action space: target joint positions (continuous)
        self.action_space = spaces.Box(
            low=self.joint_lower_limits,
            high=self.joint_upper_limits,
            dtype=np.float32
        )
        
        # Observation space
        obs_dim = 18
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Workspace bounds for pen spawning
        self.workspace_x = [0.15, 0.35]
        self.workspace_y = [-0.15, 0.15]
        self.workspace_z = [0.05, 0.25]
        
        # Success criteria
        self.success_distance = 0.03
        self.grasp_distance = 0.02
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Connect to PyBullet if not connected
        if self.physics_client is None:
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Reset simulation
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # Load robot
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            self.robot_urdf_path,
            robot_start_pos,
            robot_start_orientation,
            useFixedBase=True,
            physicsClientId=self.physics_client
        )
        
        # Get joint info and find end-effector link
        self.num_joints_total = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # Find the hand_tcp or gripper_link index
        self.tcp_link_idx = None
        for i in range(self.num_joints_total):
            link_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            link_name = link_info[12].decode('utf-8')
            if 'hand_tcp' in link_name.lower() or link_name == 'hand_tcp':
                self.tcp_link_idx = i
                break
        
        # If hand_tcp not found, use gripper_link or the last link
        if self.tcp_link_idx is None:
            for i in range(self.num_joints_total):
                link_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                link_name = link_info[12].decode('utf-8')
                if 'gripper' in link_name.lower():
                    self.tcp_link_idx = i
                    break
        
        # Fall back to last link if still not found
        if self.tcp_link_idx is None:
            self.tcp_link_idx = self.num_joints_total - 1
        
        # Debug: Print which link is being used (only once)
        if self.tcp_link_idx is not None and self.tcp_link_idx < self.num_joints_total:
            link_info = p.getJointInfo(self.robot_id, self.tcp_link_idx, physicsClientId=self.physics_client)
            link_name = link_info[12].decode('utf-8')
            if not hasattr(self, '_printed_tcp'):
                print(f"Using link '{link_name}' (index {self.tcp_link_idx}) as end-effector")
                self._printed_tcp = True
        
        # Set initial joint positions (neutral pose)
        initial_joint_positions = [0.0, 0.0, 1.57, 0.5]
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, initial_joint_positions[i], physicsClientId=self.physics_client)
        
        # Randomize pen position
        pen_pos = [
            np.random.uniform(self.workspace_x[0], self.workspace_x[1]),
            np.random.uniform(self.workspace_y[0], self.workspace_y[1]),
            np.random.uniform(self.workspace_z[0], self.workspace_z[1])
        ]
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
            rgbaColor=[0, 0, 1, 1],
            physicsClientId=self.physics_client
        )
        self.pen_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=pen_collision_shape,
            baseVisualShapeIndex=pen_visual_shape,
            basePosition=pen_pos,
            baseOrientation=pen_orientation,
            physicsClientId=self.physics_client
        )
        
        # Step simulation to stabilize
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        self.current_step = 0
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
        reward, terminated, info = self._compute_reward(observation)
        
        # Check truncation (max steps)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation from environment"""
        # Joint positions
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.physics_client)
        joint_positions = np.array([state[0] for state in joint_states])
        
        # Gripper (end-effector) pose
        link_state = p.getLinkState(self.robot_id, self.tcp_link_idx, computeForwardKinematics=True, physicsClientId=self.physics_client)
        gripper_pos = np.array(link_state[0])
        gripper_orn = np.array(link_state[1])
        
        # Pen pose
        pen_pos, pen_orn = p.getBasePositionAndOrientation(self.pen_id, physicsClientId=self.physics_client)
        pen_pos = np.array(pen_pos)
        pen_euler = np.array(p.getEulerFromQuaternion(pen_orn))
        
        # Distance to pen
        distance = np.linalg.norm(gripper_pos - pen_pos)
        
        # Concatenate observation
        obs = np.concatenate([
            joint_positions,
            gripper_pos,
            gripper_orn,
            pen_pos,
            pen_euler,
            [distance]
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, observation):
        """Compute reward based on current observation"""
        gripper_pos = observation[4:7]
        pen_pos = observation[11:14]
        distance = observation[17]
        
        reward = 0.0
        terminated = False
        
        # Always include distance_to_pen and success in info
        info = {
            'distance_to_pen': float(distance),
            'success': False,
            'collision': False
        }
        
        # Distance-based reward
        reward_distance = -distance * 5.0
        reward += reward_distance
        
        # Success bonus
        if distance < self.success_distance:
            reward += 100.0
            terminated = True
            info['success'] = True
        
        # Small time penalty
        reward -= 0.1
        
        # Check if pen fell off table
        if pen_pos[2] < 0.0:
            reward -= 50.0
            terminated = True
        
        return reward, terminated, info
    
    def _get_info(self):
        """Get additional info"""
        observation = self._get_observation()
        distance = observation[17]
        return {
            'distance_to_pen': float(distance),
            'success': distance < self.success_distance
        }
    
    def render(self):
        """Render environment (handled by PyBullet GUI)"""
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
    print("Testing RoArm Environment...")
    env = RoArmPickEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial distance to pen: {info['distance_to_pen']:.3f}m")
    
    # Run random actions
    for i in range(100):
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