# Imports and dependencies remain unchanged
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
from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback to track training progress and save model at intervals
    """
    def __init__(self, save_freq=1000, save_path="./models/", verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Track episode rewards
        if 'reward' in self.locals:
            self.current_episode_reward += self.locals['reward']
        
        # Check if episode is done
        if 'done' in self.locals and self.locals['done']:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count}: Reward = {self.current_episode_reward:.2f}")
            
            self.current_episode_reward = 0
        
        # Save model at specified intervals
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"sac_roarm_camera_step_{self.n_calls}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}: {model_path}")
        
        return True


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

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.camera_height, self.camera_width, 3),
            dtype=np.uint8
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.joint_limits = []
        for i in range(3):
            joint_info = p.getJointInfo(self.robot, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            self.joint_limits.append((lower_limit, upper_limit))

    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)  # Ensure gravity is set
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot)

        self.camera_joint_index = None
        self.camera_link_index = None

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "gripper_link":
                self.camera_link_index = i
                self.camera_joint_index = i
                break

        if self.camera_joint_index is None:
            self.camera_joint_index = self.num_joints - 1
            self.camera_link_index = self.num_joints - 1

        self.reset_object_position()

    def reset_object_position(self):
        # Remove existing object if it exists
        if hasattr(self, 'object_id'):
            try:
                p.removeBody(self.object_id)
            except:
                pass
        
        # Create object at fixed position on the ground
        # Position it slightly above ground (0.02) to account for cube size
        self.object_id = p.loadURDF("cube_small.urdf", [0.4, 0.0, 0.02])
        
        # Set object properties to prevent floating
        # Make it heavier and add friction
        p.changeDynamics(self.object_id, -1, 
                        mass=1.0,  # Increase mass
                        lateralFriction=0.8,  # Add friction
                        spinningFriction=0.3,
                        rollingFriction=0.1,
                        restitution=0.1)  # Reduce bounciness
        
        # Let physics settle for a few steps
        for _ in range(50):
            p.stepSimulation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1])
        self.current_joint_positions = []

        for i in range(min(3, self.num_joints)):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                random_pos = np.random.uniform(lower * 0.5, upper * 0.5) if lower < upper else 0.0
            else:
                random_pos = 0.0
            p.resetJointState(self.robot, i, random_pos)
            self.current_joint_positions.append(random_pos)

        # Reset object position
        self.reset_object_position()

        # Let the simulation settle
        for _ in range(20):
            p.stepSimulation()

        return self._get_observation(), {}

    def _get_observation(self):
        return self._get_camera_image()

    def _get_camera_image(self):
        if self.camera_link_index is not None:
            link_state = p.getLinkState(self.robot, self.camera_link_index, computeForwardKinematics=True)
            link_pos = link_state[0]
            link_orient_quat = link_state[1]
            link_orient = p.getMatrixFromQuaternion(link_orient_quat)

            forward = np.array([link_orient[0], link_orient[3], link_orient[6]])
            up = np.array([link_orient[2], link_orient[5], link_orient[8]])
            target = np.array(link_pos) + forward * 0.3

            view_matrix = p.computeViewMatrix(link_pos, target.tolist(), up.tolist())
        else:
            view_matrix = p.computeViewMatrix([0.5, 0, 0.4], [0, 0, 0], [0, 0, 1])

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.01,
            farVal=2.0
        )

        _, _, rgba_img, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgba_img = np.array(rgba_img, dtype=np.uint8).reshape((self.camera_height, self.camera_width, 4))
        rgb_img = rgba_img[:, :, :3]

        if self.render_mode == "human":
            cv2.imshow("Robot Camera", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return rgb_img

    def step(self, action):
        for i in range(min(3, len(action))):
            lower, upper = self.joint_limits[i]
            target_pos = lower + (action[i] + 1) * 0.5 * (upper - lower) if lower < upper else 0.0
            self.current_joint_positions[i] = target_pos
            p.setJointMotorControl2(
                self.robot, i,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=1000,
                maxVelocity=1.0
            )

        for _ in range(5):
            p.stepSimulation()

        obs = self._get_observation()
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        ee_pos = p.getLinkState(self.robot, self.camera_link_index)[0]

        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        distance_reward = -distance * 2.0

        proximity_bonus = 1.0 if distance < 0.1 else 0.5 if distance < 0.2 else 0.0
        joint_penalty = sum(
            -0.1 if ((pos - lower) / (upper - lower) < 0.1 or (pos - lower) / (upper - lower) > 0.9)
            else 0.0
            for (lower, upper), pos in zip(self.joint_limits, self.current_joint_positions)
        )

        reward = distance_reward + proximity_bonus + joint_penalty
        done = distance < 0.05
        
        return obs, reward, done, False, {
            'distance': distance,
            'ee_pos': ee_pos,
            'obj_pos': obj_pos
        }

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
        cv2.destroyAllWindows()


def train_camera_search_sac():
    print("=== Starting SAC Training ===")
    
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    # Create callback to track training progress
    callback = TrainingProgressCallback(
        save_freq=2000,  # Save every 2000 steps
        save_path="./models/",
        verbose=1
    )

    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./sac_roarm_tensorboard/",
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1
    )

    print("Training SAC model...")
    print("Model will be saved every 2000 steps in ./models/ directory")
    
    model.learn(total_timesteps=10000, callback=callback)  # Reduced for demo
    
    # Save final model
    final_model_path = "sac_roarm_camera_final"
    model.save(final_model_path)
    print(f"Final model saved as: {final_model_path}")
    
    # Print training summary
    if len(callback.episode_rewards) > 0:
        print(f"\n=== Training Summary ===")
        print(f"Total Episodes: {callback.episode_count}")
        print(f"Average Reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"Best Episode Reward: {max(callback.episode_rewards):.2f}")
        print(f"Final 10 Episodes Average: {np.mean(callback.episode_rewards[-10:]):.2f}")


def test_trained_model_sac(model_path="sac_roarm_camera_final"):
    print(f"=== Testing Model: {model_path} ===")
    
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    try:
        model = SAC.load(model_path, env=env)
        print(f"Successfully loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train the model first.")
        return

    obs = env.reset()
    total_reward = 0
    episode_count = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if step % 50 == 0:  # Print less frequently
            print(f"Step {step}: Reward = {reward[0]:.3f}, Total = {total_reward:.3f}")
        
        time.sleep(0.05)

        if done[0]:
            episode_count += 1
            print(f"Episode {episode_count} completed! Total reward: {total_reward:.3f}")
            obs = env.reset()
            total_reward = 0

    env.close()


def evaluate_model(model_path="sac_roarm_camera_final"):
    print(f"=== Evaluating Model: {model_path} ===")
    
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="rgb_array"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    try:
        model = SAC.load(model_path, env=env)
        print(f"Successfully loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train the model first.")
        return

    num_episodes = 10
    rewards = []
    lengths = []
    success = 0

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_len = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_len += 1
            if done[0]:
                if info[0]['distance'] < 0.05:
                    success += 1
                break
        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"Episode {ep + 1}: Reward = {ep_reward:.2f}, Steps = {ep_len}, Distance = {info[0]['distance']:.3f}")

    print(f"\n=== Evaluation Results for {model_path} ===")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg Length: {np.mean(lengths):.1f}")
    print(f"Success Rate: {success}/{num_episodes} ({success * 10}%)")

    env.close()


def compare_models():
    """Compare different saved models"""
    models_dir = "./models/"
    if not os.path.exists(models_dir):
        print("No models directory found. Train a model first.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        print("No model files found in ./models/ directory.")
        return
    
    print("=== Comparing Multiple Models ===")
    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file.replace('.zip', ''))
        print(f"\nEvaluating {model_file}:")
        evaluate_model(model_path)


if __name__ == "__main__":
    # Train the model
    # train_camera_search_sac()
    
    # Test the final model
    # test_trained_model_sac()
    
    # Evaluate the final model
    evaluate_model()
    
    # Compare all saved models
    # compare_models()