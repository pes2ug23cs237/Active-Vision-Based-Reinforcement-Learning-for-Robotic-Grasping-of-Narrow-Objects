# Imports and dependencies
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import cv2
import time
import os
import math
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback to track training progress and save model at intervals
    """
    def __init__(self, save_freq=5000, save_path="./models/", verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_lengths = []
        self.current_episode_length = 0
        self.success_count = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_length += 1
        
        # Track episode rewards
        if 'rewards' in self.locals:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode is done
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Track success
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                if self.locals['infos'][0].get('success', False):
                    self.success_count += 1
            
            if self.verbose > 0:
                success_rate = (self.success_count / self.episode_count * 100) if self.episode_count > 0 else 0
                print(f"Episode {self.episode_count}: Reward = {self.current_episode_reward:.2f}, Length = {self.current_episode_length}, Success Rate = {success_rate:.1f}%")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Save model at specified intervals
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"sac_roarm_camera_step_{self.n_calls}")
            self.model.save(model_path)
            if self.verbose > 0:
                success_rate = (self.success_count / self.episode_count * 100) if self.episode_count > 0 else 0
                print(f"Model saved at step {self.n_calls}: {model_path} (Success Rate: {success_rate:.1f}%)")
        
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

        self.camera_width = 128
        self.camera_height = 128

        # Calculate robot's approximate reach BEFORE loading scene
        self._calculate_robot_reach()

        self._load_scene()

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.camera_height, self.camera_width, 3),
            dtype=np.uint8
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # 4 joints

        # Get joint limits for controllable joints (first 4 in your URDF)
        self.joint_limits = []
        self.num_controllable_joints = min(4, self.num_joints)  # Your URDF has 4 revolute joints
        
        for i in range(self.num_controllable_joints):
            joint_info = p.getJointInfo(self.robot, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            joint_name = joint_info[1].decode('utf-8')
            print(f"Joint {i} ({joint_name}): limits [{lower_limit:.3f}, {upper_limit:.3f}]")
            self.joint_limits.append((lower_limit, upper_limit))
        
        self.episode_steps = 0
        self.max_episode_steps = 300  # Increased for more exploration time
        self.previous_distance = None

    def _calculate_robot_reach(self):
        """Calculate the robot's approximate workspace based on URDF dimensions"""
        # From URDF analysis:
        # Base height: 0.123m (base_link_to_link1 z-offset)
        # Link2 length: 0.237m (link2_to_link3 x-offset: 0.236815)
        # Link3 length: 0.216m (link3_to_gripper_link y-offset: -0.21599, so 0.21599)
        # TCP offset: 0.064m (link3_to_hand_tcp additional offset: -0.2802 vs -0.21599 = 0.0642)
        
        self.base_height = 0.123
        self.link2_length = 0.237  # Approximately from URDF
        self.link3_length = 0.216  # Approximately from URDF  
        self.tcp_offset = 0.064    # Additional TCP reach
        
        # Maximum horizontal reach (when arm fully extended)
        self.max_horizontal_reach = self.link2_length + self.link3_length + self.tcp_offset
        # Practical reach is about 75% of theoretical max (more conservative)
        self.practical_reach = self.max_horizontal_reach * 0.75
        
        print(f"Robot reach analysis (from URDF):")
        print(f"  Base height: {self.base_height:.3f}m")
        print(f"  Link2 length: {self.link2_length:.3f}m")
        print(f"  Link3 length: {self.link3_length:.3f}m") 
        print(f"  TCP offset: {self.tcp_offset:.3f}m")
        print(f"  Max horizontal reach: {self.max_horizontal_reach:.3f}m") 
        print(f"  Practical reach: {self.practical_reach:.3f}m")

    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot)

        # Find gripper link for camera mounting
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
        """Reset object position within robot's reach"""
        if hasattr(self, 'object_id'):
            try:
                p.removeBody(self.object_id)
            except:
                pass
        
        # Position object within robot's practical reach
        # Use polar coordinates for better workspace coverage
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1, self.practical_reach * 0.85)  # Conservative radius
        
        x_pos = radius * np.cos(angle)
        y_pos = radius * np.sin(angle)
        
        # Ensure object is above ground but reachable
        z_pos = np.random.uniform(0.02, min(0.2, self.base_height + 0.1))  # Reasonable height
        
        # Clamp to reasonable workspace based on actual robot dimensions
        max_reach = self.practical_reach
        x_pos = np.clip(x_pos, -max_reach, max_reach)
        y_pos = np.clip(y_pos, -max_reach, max_reach)
        
        self.object_id = p.loadURDF("cube_small.urdf", [x_pos, y_pos, z_pos])
        
        # Make object more stable
        p.changeDynamics(self.object_id, -1, 
                        mass=0.1,  # Lighter object
                        lateralFriction=1.0,
                        spinningFriction=0.5,
                        rollingFriction=0.3,
                        restitution=0.1)
        
        # Let physics settle
        for _ in range(30):
            p.stepSimulation()
            
        # Store target position for reference
        self.target_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        if hasattr(self, 'render_mode') and self.render_mode != "rgb_array":
            print(f"Object spawned at: ({self.target_pos[0]:.3f}, {self.target_pos[1]:.3f}, {self.target_pos[2]:.3f})")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1])
        
        self.current_joint_positions = []

        # Reset joints to reasonable starting positions
        for i in range(self.num_controllable_joints):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                
                if i == 0:  # Base rotation - start pointing roughly toward object area
                    random_pos = np.random.uniform(-np.pi/4, np.pi/4)
                elif i == 1:  # Shoulder - start with arm partially raised
                    random_pos = np.random.uniform(-0.5, 0.5)
                elif i == 2:  # Elbow - start with slight bend
                    random_pos = np.random.uniform(0.5, 2.0)
                elif i == 3:  # Wrist/gripper - neutral position
                    random_pos = np.random.uniform(0.2, 0.8)
                else:
                    random_pos = np.random.uniform(lower * 0.3, upper * 0.3) if lower < upper else 0.0
                    
                # Clamp to joint limits
                random_pos = np.clip(random_pos, lower, upper)
            else:
                random_pos = 0.0
                
            p.resetJointState(self.robot, i, random_pos)
            self.current_joint_positions.append(random_pos)

        self.reset_object_position()
    
        self.episode_steps = 0
        self.previous_distance = None

        # Let everything settle
        for _ in range(50):
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

            # Camera points in the direction of the gripper
            forward = np.array([link_orient[0], link_orient[3], link_orient[6]])
            up = np.array([link_orient[2], link_orient[5], link_orient[8]])
            target = np.array(link_pos) + forward * 0.3  # Closer focus distance

            view_matrix = p.computeViewMatrix(link_pos, target.tolist(), up.tolist())
        else:
            # Fallback camera position
            view_matrix = p.computeViewMatrix([0.3, 0, 0.3], [0, 0, 0.1], [0, 0, 1])

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=75,  # Wider field of view
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
        self.episode_steps += 1
        
        # Apply actions to joints with better scaling
        for i in range(min(self.num_controllable_joints, len(action))):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if lower < upper:
                    # Map action from [-1, 1] to joint range
                    target_pos = lower + (action[i] + 1) * 0.5 * (upper - lower)
                    target_pos = np.clip(target_pos, lower, upper)
                else:
                    target_pos = 0.0
                
                self.current_joint_positions[i] = target_pos
                p.setJointMotorControl2(
                    self.robot, i,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=3000,  # Increased force for better control
                    maxVelocity=1.5
                )

        # Step simulation
        for _ in range(8):  # Fewer steps for faster training
            p.stepSimulation()

        # Get observations and calculate reward
        obs = self._get_observation()
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        ee_pos = p.getLinkState(self.robot, self.camera_link_index)[0]

        # Calculate 3D distance
        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        
        # Enhanced reward function
        reward = 0.0
        
        # Distance-based reward (main component)
        max_distance = self.max_horizontal_reach * 1.5  # Normalize by max possible distance
        distance_reward = -distance / max_distance * 10.0
        reward += distance_reward
     
        # Progress reward (encourage getting closer)
        if self.previous_distance is not None:
            improvement = self.previous_distance - distance
            reward += improvement * 20.0  # Increased progress reward
        
        self.previous_distance = distance
        
        # Proximity bonuses (staged rewards)
        if distance < 0.03:  # Very close
            reward += 100.0
        elif distance < 0.05:  # Close
            reward += 50.0
        elif distance < 0.08:  # Getting close
            reward += 20.0
        elif distance < 0.12:  # Approaching
            reward += 10.0
        elif distance < 0.2:  # In general area
            reward += 5.0
        
        # Joint limit penalty (encourage staying within comfortable range)
        joint_penalty = 0.0
        for i, ((lower, upper), pos) in enumerate(zip(self.joint_limits, self.current_joint_positions)):
            if upper > lower:
                normalized_pos = (pos - lower) / (upper - lower)
                # Penalty for being too close to limits
                if normalized_pos < 0.05 or normalized_pos > 0.95:
                    joint_penalty -= 2.0
                elif normalized_pos < 0.1 or normalized_pos > 0.9:
                    joint_penalty -= 0.5
        reward += joint_penalty
        
        # Small time penalty to encourage efficiency
        reward -= 0.02
        
        # Workspace penalty - discourage end effector going too far from base
        ee_distance_from_base = np.linalg.norm(np.array(ee_pos[:2]))  # 2D distance from base
        if ee_distance_from_base > self.practical_reach:
            reward -= 5.0  # Strong penalty for going outside practical workspace
        
        # Episode termination conditions
        done = False
        success = False
        
        if distance < 0.03:  # Success condition (tighter tolerance)
            done = True
            success = True
            reward += 200.0  # Big success bonus
            print(f"SUCCESS! Distance: {distance:.4f}m at step {self.episode_steps}")
        elif self.episode_steps >= self.max_episode_steps:
            done = True
            reward -= 20.0  # Timeout penalty
        elif ee_distance_from_base > self.practical_reach * 1.2:  # Way outside workspace
            done = True
            reward -= 50.0  # Strong penalty for leaving workspace
        
        info = {
            'distance': distance,
            'ee_pos': ee_pos,
            'obj_pos': obj_pos,
            'episode_steps': self.episode_steps,
            'success': success,
            'workspace_violation': ee_distance_from_base > self.practical_reach
        }
        
        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
        cv2.destroyAllWindows()


def evaluate_model_during_training(model, env, n_episodes=10):
    """Evaluate model performance during training"""
    successes = 0
    total_distance = 0
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_length += 1
            
            if done:
                if info[0].get('success', False):
                    successes += 1
                total_distance += info[0].get('distance', 0)
                episode_lengths.append(episode_length)
                break
    
    success_rate = successes / n_episodes * 100
    avg_distance = total_distance / n_episodes
    avg_length = np.mean(episode_lengths)
    
    return success_rate, avg_distance, avg_length


def train_camera_search_sac():
    print("=== Starting SAC Training with Improved Environment ===")
    
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    callback = TrainingProgressCallback(
        save_freq=20000,  # Save every 20k steps
        save_path="./models/",
        verbose=1
    )

    # Improved SAC hyperparameters
    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./sac_roarm_tensorboard/",
        learning_rate=3e-4,  # Higher learning rate
        buffer_size=100000,   # Larger buffer
        learning_starts=10000,  # More initial exploration
        batch_size=256,      # Larger batch size
        tau=0.01,            # Faster target network updates
        gamma=0.995,         # Slightly higher discount factor
        train_freq=4,       
        gradient_steps=4,   
        ent_coef='auto',     # Automatic entropy coefficient tuning
        target_update_interval=1,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Larger network
            activation_fn=torch.nn.ReLU,
        )
    )

    print("Training SAC model with improved environment...")
    print("Model will be saved every 20000 steps in ./models/ directory")
    
    # Create evaluation environment for periodic testing
    eval_env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="rgb_array"  # No GUI for evaluation
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecTransposeImage(eval_env)
    
    # Training with periodic evaluation
    total_timesteps = 300000
    eval_freq = 25000
    
    for step in range(0, total_timesteps, eval_freq):
        # Train for eval_freq steps
        steps_to_train = min(eval_freq, total_timesteps - step)
        model.learn(total_timesteps=steps_to_train, callback=callback, reset_num_timesteps=False)
        
        # Evaluate current performance
        if step > 0:
            print(f"\n=== Evaluation at {step + steps_to_train} steps ===")
            success_rate, avg_distance, avg_length = evaluate_model_during_training(model, eval_env, n_episodes=10)
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Average Final Distance: {avg_distance:.4f}m")
            print(f"Average Episode Length: {avg_length:.1f}")
            print("=" * 50)
    
    # Save final model
    final_model_path = "sac_roarm_camera_improved_final"
    model.save(final_model_path)
    print(f"Final model saved as: {final_model_path}")
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    success_rate, avg_distance, avg_length = evaluate_model_during_training(model, eval_env, n_episodes=20)
    print(f"Final Success Rate: {success_rate:.1f}%")
    print(f"Final Average Distance: {avg_distance:.4f}m")
    print(f"Final Average Episode Length: {avg_length:.1f}")
    
    # Print training summary
    if len(callback.episode_rewards) > 0:
        print(f"\n=== Training Summary ===")
        print(f"Total Episodes: {callback.episode_count}")
        print(f"Total Successes: {callback.success_count}")
        print(f"Overall Success Rate: {callback.success_count/callback.episode_count*100:.1f}%")
        print(f"Average Reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"Best Episode Reward: {max(callback.episode_rewards):.2f}")
        print(f"Final 10 Episodes Average: {np.mean(callback.episode_rewards[-10:]):.2f}")
        print(f"Average Episode Length: {np.mean(callback.episode_lengths):.1f}")
    
    eval_env.close()


def test_trained_model_sac(model_path="sac_roarm_camera_improved_final"):
    print(f"=== Testing Improved Model: {model_path} ===")
    
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
    success_count = 0

    for step in range(5000):  # Longer testing
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if step % 50 == 0:
            distance = info[0].get('distance', 'N/A')
            ee_pos = info[0].get('ee_pos', [0,0,0])
            print(f"Step {step}: Reward = {reward[0]:.3f}, Distance = {distance:.4f}, EE = ({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})")
        
        time.sleep(0.02)

        if done[0]:
            episode_count += 1
            distance = info[0].get('distance', 'N/A')
            success = info[0].get('success', False)
            if success:
                success_count += 1
            
            status = "SUCCESS!" if success else "Failed"
            print(f"Episode {episode_count} completed! Total reward: {total_reward:.3f}, Final distance: {distance:.4f} - {status}")
            print(f"Success rate so far: {success_count}/{episode_count} ({success_count/episode_count*100:.1f}%)")
            
            obs = env.reset()
            total_reward = 0

    env.close()
    print(f"Final success rate: {success_count}/{episode_count} ({success_count/episode_count*100:.1f}%)")


def evaluate_model(model_path="sac_roarm_camera_improved_final", n_episodes=50):
    print(f"=== Evaluating Model: {model_path} ===")
    
    env = CameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="rgb_array"  # Faster evaluation without GUI
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    try:
        model = SAC.load(model_path, env=env)
        print(f"Successfully loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train the model first.")
        return

    rewards = []
    lengths = []
    distances = []
    success_count = 0

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_len = 0
        final_distance = 0
        
        for step in range(300):  # Match max episode steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_len += 1
            final_distance = info[0]['distance']
            
            if done[0]:
                if info[0].get('success', False):
                    success_count += 1
                break
                
        rewards.append(ep_reward)
        lengths.append(ep_len)
        distances.append(final_distance)
        
        status = "SUCCESS" if info[0].get('success', False) else "FAILED"
        print(f"Episode {ep + 1:2d}: Reward = {ep_reward:6.2f}, Steps = {ep_len:3d}, Distance = {final_distance:.4f}m - {status}")

    print(f"\n=== Evaluation Results for {model_path} ===")
    print(f"Success Rate: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg Episode Length: {np.mean(lengths):.1f}")
    print(f"Avg Final Distance: {np.mean(distances):.4f}m")
    print(f"Min Final Distance: {min(distances):.4f}m")
    print(f"Max Final Distance: {max(distances):.4f}m")

    env.close()
    
    return {
        'success_rate': success_count/n_episodes*100,
        'avg_reward': np.mean(rewards),
        'avg_distance': np.mean(distances),
        'avg_length': np.mean(lengths)
    }


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
    results = {}
    
    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file.replace('.zip', ''))
        print(f"\nEvaluating {model_file}:")
        try:
            result = evaluate_model(model_path, n_episodes=20)
            results[model_file] = result
        except Exception as e:
            print(f"Error evaluating {model_file}: {e}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Success Rate':<12} {'Avg Distance':<12}")
    print(f"{'-'*60}")
    
    for model_name, result in results.items():
        success_rate = result['success_rate']
        avg_distance = result['avg_distance']
        print(f"{model_name:<35} {success_rate:>8.1f}%     {avg_distance:>8.4f}m")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            train_camera_search_sac()
        elif command == "test":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "sac_roarm_camera_improved_final"
            test_trained_model_sac(model_path)
        elif command == "evaluate":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "sac_roarm_camera_improved_final"
            evaluate_model(model_path)
        elif command == "compare":
            compare_models()
        else:
            print("Usage: python script.py [train|test|evaluate|compare] [model_path]")
    else:
        # Default behavior - train the model
        # train_camera_search_sac()
        
        # Uncomment these to run after training:
        test