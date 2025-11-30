import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import sys
import cv2

# ===========================================================
#                 Custom Environment
# ===========================================================

class RoArmCameraSearchEnv(gym.Env):
    """RoArm robot arm with gripper camera searching for target."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False, show_camera=False):
        super(RoArmCameraSearchEnv, self).__init__()
        self.render_mode = "human" if render else None
        self.show_camera = show_camera

        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep=1/240)

        # Action space: 3 control joints + gripper
        self.action_space = gym.spaces.Box(
            low=np.array([-3.14, 0.0, 1.0, 0.0]),    # Joint 2 min=1.0 to keep gripper angled forward
            high=np.array([3.14, 1.57, 3.14, 1.5]),  # Max angles
            dtype=np.float32
        )
        
        # Observation: RGB camera from gripper
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        self.robot_id = None
        self.target_id = None
        self.target_pos = None
        self.step_count = 0
        self.gripper_link_index = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load plane and robot
        p.loadURDF("plane.urdf")
        
        # Load the RoArm robot - ADJUST PATH TO YOUR URDF
        urdf_path = "roarm_description/urdf/roarm.urdf"
        
        # Check if file exists
        if not os.path.exists(urdf_path):
            # Try alternative path
            urdf_path = "./roarm_description.urdf"
            if not os.path.exists(urdf_path):
                print(f"❌ ERROR: Cannot find robot URDF file!")
                print(f"   Please place roarm_description.urdf in the current directory")
                print(f"   or update the urdf_path in the code")
                raise FileNotFoundError(f"Robot URDF not found at {urdf_path}")
        
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        
        # Find the gripper link index (where camera will be attached)
        num_joints = p.getNumJoints(self.robot_id)
        print(f"🤖 Robot loaded with {num_joints} joints")
        
        # Print joint info to find the correct indices
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            print(f"   Joint {i}: {joint_name}, Link: {link_name}")
        
        # Set gripper link - adjust based on your robot structure
        # Looking for link3 or gripper_link
        self.gripper_link_index = num_joints - 1  # Usually the last link
        
        # Joint indices for control (base, joint1, joint2, joint3)
        self.controlled_joints = [0, 1, 2, 3]  # Adjust if needed
        
        # Spawn target sphere in reachable workspace at table height
        # Raised to be more visible to gripper camera
        self.target_pos = np.array([
            np.random.uniform(0.15, 0.35),  # X: forward (15-35cm)
            np.random.uniform(-0.15, 0.15), # Y: left-right (±15cm)
            np.random.uniform(0.15, 0.30)   # Z: height (15-30cm - table height)
        ])
        
        self.target_id = p.loadURDF("sphere2.urdf", self.target_pos, globalScaling=0.05)
        p.changeVisualShape(self.target_id, -1, rgbaColor=[1, 0, 0, 1])
        
        # Set initial joint positions to point gripper toward workspace
        initial_angles = [
            0.0,    # Joint 0: Base rotation (centered)
            0.8,    # Joint 1: Shoulder (lift up)
            2.5,    # Joint 2: Elbow-to-gripper (ANGLE DOWN toward ground)
            0.5     # Joint 3: Gripper open/close
        ]
        for joint_idx, angle in zip(self.controlled_joints, initial_angles):
            p.resetJointState(self.robot_id, joint_idx, angle)
        
        self.step_count = 0
        
        # Stabilize physics
        for _ in range(50):
            p.stepSimulation()
        
        obs = self.get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Set joint positions based on action
        for joint_idx, target_angle in zip(self.controlled_joints, action):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=50,
                maxVelocity=1.0
            )
        
        # Step simulation multiple times for smooth motion
        for _ in range(10):
            p.stepSimulation()
        
        # Get gripper (end-effector) position
        if self.gripper_link_index is not None:
            link_state = p.getLinkState(self.robot_id, self.gripper_link_index)
            gripper_pos = np.array(link_state[0])
        else:
            # Fallback: use last joint position
            gripper_pos = np.array(p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id)-1)[0])
        
        # Calculate distance to target
        distance = np.linalg.norm(gripper_pos - self.target_pos)
        
        # Reward shaping
        reward = -distance * 10.0  # Penalty for distance
        
        # Bonus for getting close
        if distance < 0.05:
            reward += 50.0  # Big reward for reaching target
        elif distance < 0.1:
            reward += 10.0
        
        # Check termination
        terminated = distance < 0.05
        truncated = self.step_count > 100
        self.step_count += 1
        
        obs = self.get_obs()
        info = {
            "distance": distance,
            "gripper_pos": gripper_pos,
            "target_pos": self.target_pos
        }
        
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        """Get camera image from gripper perspective"""
        
        # Get gripper link position and orientation
        if self.gripper_link_index is not None:
            link_state = p.getLinkState(self.robot_id, self.gripper_link_index)
            cam_pos = link_state[0]
            cam_ori = link_state[1]
        else:
            # Fallback to last joint
            link_state = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id)-1)
            cam_pos = link_state[0]
            cam_ori = link_state[1]
        
        # Convert orientation to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(cam_ori)
        
        # Camera looks forward and slightly down from gripper
        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        
        # Tilt camera down by 30 degrees to see floor targets better
        import math
        tilt_angle = math.radians(30)  # 30 degrees down
        forward_tilted = [
            forward[0],
            forward[1], 
            forward[2] - math.sin(tilt_angle)  # Tilt downward
        ]
        
        # Normalize
        mag = math.sqrt(sum(f*f for f in forward_tilted))
        forward_tilted = [f/mag for f in forward_tilted]
        
        up = [0, 0, 1]  # World up
        
        # Compute view matrix
        cam_target = [
            cam_pos[0] + forward_tilted[0] * 0.5,
            cam_pos[1] + forward_tilted[1] * 0.5,
            cam_pos[2] + forward_tilted[2] * 0.5
        ]
        
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, up)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
        
        # Capture image
        img_arr = p.getCameraImage(64, 64, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(img_arr[2], dtype=np.uint8).reshape((64, 64, 4))[:, :, :3]
        
        # Visualization
        if self.show_camera:
            display_img = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_NEAREST)
            
            # Get current info
            link_state = p.getLinkState(self.robot_id, self.gripper_link_index if self.gripper_link_index else -1)
            gripper_pos = np.array(link_state[0])
            distance = np.linalg.norm(gripper_pos - self.target_pos)
            
            # Overlay info
            cv2.putText(display_img, f"Distance: {distance:.3f}m", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, f"Step: {self.step_count}", (5, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, f"Gripper: ({gripper_pos[0]:.2f}, {gripper_pos[1]:.2f}, {gripper_pos[2]:.2f})", 
                       (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(display_img, f"Target: ({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f})", 
                       (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw crosshair at center
            h, w = display_img.shape[:2]
            cv2.line(display_img, (w//2-10, h//2), (w//2+10, h//2), (0, 255, 0), 1)
            cv2.line(display_img, (w//2, h//2-10), (w//2, h//2+10), (0, 255, 0), 1)
            
            cv2.imshow("Gripper Camera View", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        return rgb

    def close(self):
        if self.show_camera:
            cv2.destroyAllWindows()
        p.disconnect(self.physics_client)


# ===========================================================
#                 Callback for Checkpointing
# ===========================================================

class TrainCallback(BaseCallback):
    def __init__(self, save_path, check_freq=5000, verbose=1):
        super(TrainCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_path, f"ppo_roarm_{self.n_calls}_steps.zip")
            self.model.save(path)
            if self.verbose:
                print(f"✅ Checkpoint saved at {path}")
        return True


# ===========================================================
#                 Training Functions
# ===========================================================

def train_roarm_ppo():
    """Train without visualization (faster)"""
    log_dir = "./logs_ppo_roarm/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = DummyVecEnv([lambda: Monitor(RoArmCameraSearchEnv(render=False), log_dir)])
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=True
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu",
    )

    callback = TrainCallback(log_dir, check_freq=10000)
    print("🚀 Training RoArm with PPO...")
    model.learn(total_timesteps=100_000, callback=callback, log_interval=10)
    model.save(os.path.join(log_dir, "ppo_roarm_final.zip"))
    print("✅ Training complete!")
    env.close()


def train_with_visualization():
    """Train WITH live visualization"""
    log_dir = "./logs_ppo_roarm/"
    os.makedirs(log_dir, exist_ok=True)
    
    print("🎮 Training with live visualization")
    print("⚠️  Training will be slower but you can watch it learn!")
    print("📹 You'll see both PyBullet sim AND gripper camera view\n")
    
    env = DummyVecEnv([lambda: Monitor(RoArmCameraSearchEnv(render=True, show_camera=True), log_dir)])
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=True
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu",
    )

    callback = TrainCallback(log_dir, check_freq=5000)
    
    try:
        model.learn(total_timesteps=50_000, callback=callback, log_interval=5)
        model.save(os.path.join(log_dir, "ppo_roarm_final.zip"))
        print("✅ Training complete!")
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted")
        model.save(os.path.join(log_dir, "ppo_roarm_interrupted.zip"))
        print("💾 Model saved")
    
    env.close()


# ===========================================================
#                 Testing and Evaluation
# ===========================================================

def test_trained_model():
    """Test with visualization"""
    model_path = "./logs_ppo_roarm/ppo_roarm_final.zip"
    
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("   Train first with: python script.py train-viz")
        return
    
    env = RoArmCameraSearchEnv(render=True, show_camera=True)
    model = PPO.load(model_path)
    
    print("🎮 Testing trained RoArm robot")
    print("Press Ctrl+C to stop\n")
    
    episode = 1
    try:
        obs, info = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"✅ Episode {episode}: SUCCESS! Distance: {info['distance']:.4f}m")
                episode += 1
                obs, info = env.reset()
            elif truncated:
                print(f"⏱️  Episode {episode}: Timeout. Final distance: {info['distance']:.4f}m")
                episode += 1
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    
    env.close()


def evaluate_model():
    """Evaluate without visualization"""
    model_path = "./logs_ppo_roarm/ppo_roarm_final.zip"
    
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        return
    
    env = RoArmCameraSearchEnv(render=False)
    model = PPO.load(model_path)
    
    print("📊 Evaluating over 10 episodes...")
    
    rewards = []
    distances = []
    success_count = 0
    
    for episode in range(10):
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        final_distance = info['distance']
        distances.append(final_distance)
        rewards.append(total_reward)
        
        if final_distance < 0.05:
            success_count += 1
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"
        
        print(f"Episode {episode+1}: reward={total_reward:.1f}, distance={final_distance:.4f}m {status}")
    
    env.close()
    
    print(f"\n📈 Results:")
    print(f"   Average reward: {np.mean(rewards):.2f}")
    print(f"   Average distance: {np.mean(distances):.4f}m")
    print(f"   Success rate: {success_count}/10 ({success_count*10}%)")


def test_joint_angles():
    """Test different joint angles to find best gripper orientation"""
    print("🔧 Testing JOINT 2 angles (the one connected to gripper)\n")
    
    env = RoArmCameraSearchEnv(render=True, show_camera=True)
    env.reset()
    
    # Test different joint 2 angles (elbow-to-gripper joint)
    test_configs = [
        [0.0, 0.8, 1.0, 0.5],   # Joint2 = 1.0
        [0.0, 0.8, 1.5, 0.5],   # Joint2 = 1.5
        [0.0, 0.8, 2.0, 0.5],   # Joint2 = 2.0
        [0.0, 0.8, 2.5, 0.5],   # Joint2 = 2.5
        [0.0, 0.8, 3.0, 0.5],   # Joint2 = 3.0
        [0.0, 0.8, 3.14, 0.5],  # Joint2 = 3.14 (max)
    ]
    
    print("Testing different Joint2 (elbow-to-gripper) angles:")
    print("Watch which angle makes the gripper camera see the red sphere!\n")
    
    for i, angles in enumerate(test_configs):
        print(f"\n{'='*50}")
        print(f"Test {i+1}: Joint2 = {angles[2]:.2f} radians ({angles[2]*57.3:.0f}°)")
        print(f"{'='*50}")
        
        # Set joint positions
        for joint_idx, angle in zip(env.controlled_joints, angles):
            p.resetJointState(env.robot_id, joint_idx, angle)
        
        # Hold position for 3 seconds
        for _ in range(300):
            p.stepSimulation()
            env.get_obs()  # Update camera view
        
        input("Press Enter for next angle...")
    
    env.close()
    print("\n✅ Done! Tell me which angle worked best!")


def test_all_joints():
    """Systematically test each joint to understand the robot"""
    print("🤖 Testing each joint individually\n")
    
    env = RoArmCameraSearchEnv(render=True, show_camera=True)
    env.reset()
    
    joint_names = ["Base (J0)", "Shoulder (J1)", "Elbow (J2)", "Wrist/Gripper (J3)"]
    
    for joint_idx in range(4):
        print(f"\n{'='*60}")
        print(f"Testing {joint_names[joint_idx]}")
        print(f"{'='*60}")
        
        # Reset to neutral
        neutral = [0.0, 0.5, 1.0, 0.5]
        for j, angle in zip(env.controlled_joints, neutral):
            p.resetJointState(env.robot_id, j, angle)
        
        print(f"Moving {joint_names[joint_idx]} through its range...")
        
        # Get joint limits
        joint_info = p.getJointInfo(env.robot_id, env.controlled_joints[joint_idx])
        low, high = joint_info[8], joint_info[9]
        
        # Sweep through range
        steps = 20
        for step in range(steps):
            angle = low + (high - low) * step / steps
            p.resetJointState(env.robot_id, env.controlled_joints[joint_idx], angle)
            
            for _ in range(10):
                p.stepSimulation()
                env.get_obs()
        
        input(f"Finished testing {joint_names[joint_idx]}. Press Enter for next joint...")
    
    env.close()
    print("\n✅ Done!")


# ===========================================================
#                 Main
# ===========================================================

if __name__ == "__main__":
    try:
        cmd = sys.argv[1] if len(sys.argv) > 1 else "train-viz"
        
        print("=" * 70)
        print("🦾 RoArm Camera-Based Target Search with PPO")
        print("=" * 70)
        print(f"Command: {cmd}\n")
        
        if cmd == "train":
            train_roarm_ppo()
        elif cmd == "train-viz":
            train_with_visualization()
        elif cmd == "test":
            test_trained_model()
        elif cmd == "eval":
            evaluate_model()
        elif cmd == "debug":
            test_joint_angles()
        elif cmd == "debug-all":
            test_all_joints()
        else:
            print("Usage: python script.py [command]\n")
            print("Commands:")
            print("  train-viz  - Train WITH visualization (DEFAULT)")
            print("  train      - Train without visualization (faster)")
            print("  test       - Test trained model")
            print("  eval       - Evaluate model performance")
            print("  debug      - Test joint angles to find correct gripper orientation")
            print("  debug-all  - Test each joint individually")
    
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)