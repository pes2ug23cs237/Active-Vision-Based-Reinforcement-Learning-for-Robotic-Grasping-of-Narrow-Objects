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


class SimpleDebugCallback(BaseCallback):
    def __init__(self, save_freq=5000, save_path="./models_simple/", verbose=1):
        super(SimpleDebugCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_distances = []
        self.current_episode_reward = 0
        self.best_distance = float('inf')
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if 'rewards' in self.locals:
            self.current_episode_reward += self.locals['rewards'][0]
        
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'distance' in info:
                current_distance = info['distance']
                if current_distance < self.best_distance:
                    self.best_distance = current_distance
        
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            final_distance = float('inf')
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'distance' in info:
                    final_distance = info['distance']
            
            self.episode_distances.append(final_distance)
            
            if self.verbose > 0:
                success = "SUCCESS!" if final_distance < 0.05 else "Failed"
                print(f"Episode {self.episode_count}: Reward={self.current_episode_reward:.1f}, "
                      f"Distance={final_distance:.3f} - {success}")
                
                # Summary every 25 episodes
                if self.episode_count % 25 == 0:
                    recent_distances = self.episode_distances[-25:]
                    recent_rewards = self.episode_rewards[-25:]
                    avg_reward = np.mean(recent_rewards)
                    avg_distance = np.mean(recent_distances)
                    min_distance = min(recent_distances)
                    success_count = sum(1 for d in recent_distances if d < 0.05)
                    
                    print(f"\n=== Last 25 Episodes ===")
                    print(f"Avg Reward: {avg_reward:.1f}")  
                    print(f"Avg Distance: {avg_distance:.3f}")
                    print(f"Best Distance: {min_distance:.3f}")
                    print(f"Success Rate: {success_count}/25 ({success_count*4}%)")
                    print(f"Best Ever: {self.best_distance:.3f}\n")
            
            self.current_episode_reward = 0
        
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"sac_simple_step_{self.n_calls}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved: {model_path}")
        
        return True


class SimpleCameraSearchEnv(gym.Env):
    def __init__(self, urdf_path, render_mode="human"):
        super(SimpleCameraSearchEnv, self).__init__()
        self.urdf_path = urdf_path
        self.render_mode = render_mode

        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Very small images for fastest learning
        self.camera_width = 32
        self.camera_height = 32

        self._load_scene()

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.camera_height, self.camera_width, 3),
            dtype=np.uint8
        )

        # Only one joint for simplest possible task
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Get joint limits for first joint only
        joint_info = p.getJointInfo(self.robot, 0)
        self.joint_limit = (joint_info[8], joint_info[9])
        
        self.episode_steps = 0
        self.max_episode_steps = 50  # Very short episodes
        self.initial_distance = None

    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot)

        # Find end effector
        self.camera_link_index = None
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "gripper_link":
                self.camera_link_index = i
                break

        if self.camera_link_index is None:
            self.camera_link_index = self.num_joints - 1

        self.reset_object_position()

    def reset_object_position(self):
        if hasattr(self, 'object_id'):
            try:
                p.removeBody(self.object_id)
            except:
                pass
        
        # Position object at robot's reachable height (around 0.57m based on test)
        # And closer horizontally for easier reaching
        self.object_id = p.loadURDF("cube_small.urdf", [0.1, 0.0, 0.57])
        
        # Make it bright red
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 0, 0, 1])
        
        # Make object static so it doesn't fall
        p.changeDynamics(self.object_id, -1, mass=0)  # Static object
        
        for _ in range(20):
            p.stepSimulation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1])
        
        # Start at center of joint range
        lower, upper = self.joint_limit
        center_pos = (lower + upper) / 2.0
        p.resetJointState(self.robot, 0, center_pos)
        
        # Reset other joints to neutral
        for i in range(1, min(3, self.num_joints)):
            p.resetJointState(self.robot, i, 0.0)

        self.reset_object_position()
        self.episode_steps = 0

        for _ in range(20):
            p.stepSimulation()
        
        # Calculate initial distance
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        ee_pos = p.getLinkState(self.robot, self.camera_link_index)[0]
        self.initial_distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))

        return self._get_observation(), {}

    def _get_observation(self):
        return self._get_camera_image()

    def _get_camera_image(self):
        if self.camera_link_index is not None:
            link_state = p.getLinkState(self.robot, self.camera_link_index, computeForwardKinematics=True)
            link_pos = link_state[0]
            
            # Simple camera pointing down
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[link_pos[0], link_pos[1], link_pos[2] + 0.1],
                cameraTargetPosition=[link_pos[0], link_pos[1], 0],
                cameraUpVector=[0, 1, 0]
            )
        else:
            view_matrix = p.computeViewMatrix([0.3, 0, 0.5], [0.3, 0, 0], [0, 1, 0])

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
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
            # Enlarge for viewing
            display_img = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Robot Camera", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return rgb_img

    def step(self, action):
        self.episode_steps += 1
        
        # Move only the first joint
        lower, upper = self.joint_limit
        target_pos = lower + (action[0] + 1) * 0.5 * (upper - lower)
        target_pos = np.clip(target_pos, lower, upper)
        
        p.setJointMotorControl2(
            self.robot, 0,
            p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=1000
        )

        # Minimal simulation steps
        for _ in range(3):
            p.stepSimulation()

        obs = self._get_observation()
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        ee_pos = p.getLinkState(self.robot, self.camera_link_index)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        
        # MUCH SIMPLER REWARD FUNCTION
        reward = 0.0
        
        # Main reward: just negative distance
        reward = -distance * 10.0
        
        # Big success bonus
        if distance < 0.05:  # More realistic success threshold
            reward += 100.0
        elif distance < 0.1:  # Getting close
            reward += 50.0
        
        # Small time penalty
        reward -= 0.5
        
        # Termination
        done = False
        if distance < 0.05:  # Success threshold
            done = True
            reward += 200.0  # Success bonus
        elif self.episode_steps >= self.max_episode_steps:
            done = True
        
        info = {
            'distance': distance,
            'ee_pos': ee_pos,
            'obj_pos': obj_pos,
            'episode_steps': self.episode_steps
        }
        
        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
        cv2.destroyAllWindows()


def manual_test():
    """Test if robot can physically reach the object"""
    print("=== Manual Reachability Test ===")
    
    env = SimpleCameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    
    obs = env.reset()
    print(f"Initial setup complete. Testing joint movements...")
    
    # Test different joint positions
    joint_info = p.getJointInfo(env.robot, 0)
    lower, upper = joint_info[8], joint_info[9]
    print(f"Joint 0 limits: {lower:.3f} to {upper:.3f}")
    
    # Try different positions
    test_positions = [lower, (lower + upper) / 2, upper]
    
    for i, pos in enumerate(test_positions):
        print(f"\nTesting position {i+1}: {pos:.3f}")
        p.resetJointState(env.robot, 0, pos)
        
        for _ in range(50):
            p.stepSimulation()
        
        obj_pos, _ = p.getBasePositionAndOrientation(env.object_id)
        ee_pos = p.getLinkState(env.robot, env.camera_link_index)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        
        print(f"End effector: {ee_pos}")
        print(f"Object: {obj_pos}")
        print(f"Distance: {distance:.3f}")
        
        time.sleep(2)
    
    min_distance = min([
        np.linalg.norm(np.array(p.getLinkState(env.robot, env.camera_link_index)[0]) - np.array(p.getBasePositionAndOrientation(env.object_id)[0]))
        for pos in test_positions
        for _ in [p.resetJointState(env.robot, 0, pos)]
        for _ in range(20)
        for _ in [p.stepSimulation()]
    ])
    
    print(f"\nMinimum achievable distance: {min_distance:.3f}")
    if min_distance > 0.15:
        print("WARNING: Object may be unreachable with single joint!")
    else:
        print("Object appears reachable!")
    
    env.close()


def train_simple_sac(total_timesteps=25000):
    print("=== Training Ultra-Simple SAC ===")
    
    env = SimpleCameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="rgb_array"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    callback = SimpleDebugCallback(save_freq=2500, verbose=1)

    # Very conservative hyperparameters
    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,     # Higher learning rate
        buffer_size=10000,      # Small buffer
        learning_starts=500,    # Start learning quickly
        batch_size=32,          # Small batch
        tau=0.02,               # Fast target updates
        gamma=0.9,              # Short horizon
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.2,           # Higher exploration
        policy_kwargs=dict(
            net_arch=[64, 64],      # Tiny network
        ),
        device='cpu'
    )

    print(f"Training for {total_timesteps} timesteps (should take ~30-60 minutes)...")
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("sac_simple_final")
        print("Training completed!")
        
        if len(callback.episode_distances) > 0:
            recent = callback.episode_distances[-10:]
            success_rate = sum(1 for d in recent if d < 0.05) / len(recent) * 100
            print(f"Final success rate: {success_rate:.1f}%")
            print(f"Best distance: {callback.best_distance:.3f}")
    except Exception as e:
        print(f"Error: {e}")
        model.save("sac_simple_error")


def test_simple_model():
    """Test the simple model"""
    print("=== Testing Simple Model ===")
    
    env = SimpleCameraSearchEnv(
        urdf_path="C:/Robotics/roarm_description/urdf/roarm.urdf",
        render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    try:
        model = SAC.load("sac_simple_final", env=env)
    except:
        print("Model not found!")
        return

    for episode in range(3):
        obs = env.reset()
        print(f"\n=== Episode {episode+1} ===")
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            distance = info[0]['distance']
            print(f"Step {step}: Distance={distance:.3f}, Reward={reward[0]:.1f}")
            
            if done[0]:
                if distance < 0.05:
                    print("SUCCESS!")
                else:
                    print("Failed")
                break
            
            time.sleep(0.1)
    
    env.close()


if __name__ == "__main__":
    # STEP 1: Test if robot can reach object at all
    # manual_test()
    
    # STEP 2: If reachable, train simple version
    train_simple_sac(total_timesteps=25000)
    
    # STEP 3: Test the result
    # test_simple_model()