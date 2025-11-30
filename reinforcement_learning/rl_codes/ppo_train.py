from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from roarm_env import RoArmEnv

env = RoArmEnv(render=False)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=500_000)  # Increased for better learning
model.save("ppo_roarm_grasp")

env.close()
