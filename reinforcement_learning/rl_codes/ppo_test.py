import time
import numpy as np
from stable_baselines3 import PPO
from roarm_env import RoArmEnv

env = RoArmEnv(render=True)
model = PPO.load("ppo_roarm_grasp.zip")

print("🔍 Testing trained RoArm policy...")
num_episodes = 5

for ep in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        print(f"[Ep {ep+1} | Step {step+1}] Action: {action}, Reward: {reward:.2f}, Gripper: {obs[9]:.2f}")

        time.sleep(0.05)

        if terminated or truncated:
            print(f"✅ Episode ended | Success: {info['success']} | Total Reward: {total_reward:.2f}")
            break

    time.sleep(1)

env.close()
