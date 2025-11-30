import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import cv2
import time
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback


class CameraSearchEnv(gym.Env):
    def __init__(self, urdf_path, render_mode="human"):
        super(CameraSearchEnv, self).__init__()
        self.urdf_path = urdf_path
        self.render_mode = render_mode

        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.camera_width = 84
        self.camera_height = 84

        self._load_scene()

        # Observation = image from robot camera
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.camera_height, self.camera_width, 3),
            dtype=np.uint8
        )

        # Action = 3 DOF continuous joint movement
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Joint limits for better control
        self.joint_limits = []
        for i in range(3):
            joint_info = p.getJointInfo(self.robot, i)
            lower_limit = joint_info[8]  
            upper_limit = joint_info[9]  
            self.joint_limits.append((lower_limit, upper_limit))
        
        print(f"Joint limits: {self.joint_limits}")

    def _load_scene(self):
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot)
        print("Joints and Links:")
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            print(f"Joint {i}: {joint_name} -> Link: {link_name}")

        # Find the joint that connects to gripper_link
        self.camera_joint_index = None
        self.camera_link_index = None
        
        # First, find the gripper_link
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "gripper_link":
                self.camera_link_index = i
                self.camera_joint_index = i  
                break
        
        # If not found, try to find any joint that might be related to gripper
        if self.camera_joint_index is None:
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot, i)
                joint_name = joint_info[1].decode('utf-8').lower()
                link_name = joint_info[12].decode('utf-8').lower()
                if 'gripper' in joint_name or 'gripper' in link_name:
                    self.camera_joint_index = i
                    self.camera_link_index = i
                    print(f"Found gripper-related joint {i}: {joint_info[1].decode('utf-8')}")
                    break
        
        # Fallback: use the last joint (often end-effector)
        if self.camera_joint_index is None:
            self.camera_joint_index = self.num_joints - 1
            self.camera_link_index = self.num_joints - 1
            print(f"Using last joint {self.camera_joint_index} as camera mount")

        print(f"Camera mounted on joint {self.camera_joint_index}, link index: {self.camera_link_index}")

        # Load object (target) - randomize position slightly
        self.reset_object_position()

    def reset_object_position(self):
        # Randomize object position within reachable area
        x = np.random.uniform(0.2, 0.5)
        y = np.random.uniform(-0.3, 0.3)
        z = 0.02
        self.object_id = p.loadURDF("cube_small.urdf", [x, y, z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset robot to neutral position
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1])
        
        # Reset joints to random positions within limits
        self.current_joint_positions = []
        for i in range(min(3, self.num_joints)):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if lower < upper:  
                    random_pos = np.random.uniform(lower * 0.5, upper * 0.5)
                else:
                    random_pos = 0.0
            else:
                random_pos = 0.0
            
            p.resetJointState(self.robot, i, random_pos)
            self.current_joint_positions.append(random_pos)

        # Reset object position
        p.removeBody(self.object_id)
        self.reset_object_position()
        
        # Step simulation to stabilize
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_observation(), {}

    def _get_observation(self):
        rgb_img = self._get_camera_image()
        return rgb_img

    def _get_camera_image(self):
        if self.camera_link_index is not None:
            # Get the state of the link where camera is mounted
            link_state = p.getLinkState(self.robot, self.camera_link_index, computeForwardKinematics=True)
            link_pos = link_state[0]
            link_orient_quat = link_state[1]
            
            # Convert quaternion to rotation matrix
            link_orient = p.getMatrixFromQuaternion(link_orient_quat)
            
            # Camera orientation vectors
            forward = np.array([link_orient[0], link_orient[3], link_orient[6]])
            up = np.array([link_orient[2], link_orient[5], link_orient[8]])
            
            # Camera target point (looking forward)
            target = np.array(link_pos) + forward * 0.3
            
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=link_pos,
                cameraTargetPosition=target.tolist(),
                cameraUpVector=up.tolist()
            )
        else:
            # Fallback static view
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0.5, 0, 0.4],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1]
            )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.01,
            farVal=2.0
        )

        _, _, rgba_img, depth_img, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to numpy and reshape
        rgba_img = np.array(rgba_img, dtype=np.uint8)
        rgba_img = rgba_img.reshape((self.camera_height, self.camera_width, 4))
        rgb_img = rgba_img[:, :, :3]

        # Display in OpenCV window
        if self.render_mode == "human":
            cv2.imshow("Robot Camera", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return rgb_img

    def step(self, action):
        # Scale actions to joint limits and apply
        for i in range(min(3, len(action))):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if lower < upper:
                    # Scale action from [-1, 1] to [lower, upper]
                    target_pos = lower + (action[i] + 1) * 0.5 * (upper - lower)
                else:
                    target_pos = action[i] * 0.5  # Fallback for unlimited joints
            else:
                target_pos = action[i] * 0.5
            
            # Update current position
            self.current_joint_positions[i] = target_pos
            
            # Apply motor control
            p.setJointMotorControl2(
                self.robot, i, 
                p.POSITION_CONTROL, 
                targetPosition=target_pos, 
                force=1000,
                maxVelocity=1.0
            )

        # Step simulation
        for _ in range(5):
            p.stepSimulation()

        obs = self._get_observation()

        # Calculate reward based on distance to object and camera view
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        if self.camera_link_index is not None:
            ee_state = p.getLinkState(self.robot, self.camera_link_index)
            ee_pos = ee_state[0]
        else:
            ee_pos = [0, 0, 0]
        
        # Distance reward
        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        distance_reward = -distance * 2.0
        
        # Bonus for being close
        proximity_bonus = 0.0
        if distance < 0.1:
            proximity_bonus = 1.0
        elif distance < 0.2:
            proximity_bonus = 0.5
            
        # Penalty for extreme joint positions
        joint_penalty = 0.0
        for i, pos in enumerate(self.current_joint_positions):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if lower < upper:
                    normalized_pos = (pos - lower) / (upper - lower)
                    if normalized_pos < 0.1 or normalized_pos > 0.9:
                        joint_penalty -= 0.1

        reward = distance_reward + proximity_bonus + joint_penalty
        
        # Episode termination
        done = distance < 0.05
        truncated = False
        
        info = {
            'distance': distance,
            'ee_pos': ee_pos,
            'obj_pos': obj_pos
        }
        
        return obs, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
        cv2.destroyAllWindows()


def train_camera_search_sac():
    """Train the robot using SAC algorithm"""
    print("Creating environment...")
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    print("Creating SAC model...")
    # SAC is better suited for continuous control tasks
    model = SAC(
        "CnnPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        tensorboard_log="./sac_roarm_tensorboard/"
    )
    
    print("Starting training...")
    model.learn(total_timesteps=100000)
    
    print("Saving model...")
    model.save("sac_roarm_camera")
    print("Model saved as 'sac_roarm_camera'")


def test_trained_model_sac():
    """Test the trained SAC model"""
    print("Loading environment for testing...")
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    print("Loading trained model...")
    model = SAC.load("sac_roarm_camera", env=env)
    
    print("Testing model...")
    obs = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        print(f"Step {step}: Reward = {reward[0]:.3f}, Total = {total_reward:.3f}")
        
        time.sleep(0.05)
        
        if done[0]:
            print(f"Episode completed! Total reward: {total_reward:.3f}")
            obs = env.reset()
            total_reward = 0
    
    env.close()


def evaluate_model():
    """Evaluate the model performance"""
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="rgb_array"  
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    
    model = SAC.load("sac_roarm_camera", env=env)
    
    num_episodes = 10
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                if info[0]['distance'] < 0.05:  # Success threshold
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    
    env.close()


if __name__ == "__main__":
    # Train model with SAC
    train_camera_search_sac()

    # To test after training
    # test_trained_model_sac()
    
    # To evaluate model performance
    # evaluate_model()