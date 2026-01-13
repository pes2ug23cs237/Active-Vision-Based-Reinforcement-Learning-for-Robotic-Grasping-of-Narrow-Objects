"""
Training Script for Pen Pickup Task using PPO
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from pen_pickup_env import PenPickupEnv
import argparse


def train_agent(total_timesteps=500000, save_dir="models", log_dir="logs"):
    """
    Train PPO agent to pick up pen
    
    Args:
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("Creating training environment...")
    
    # Create vectorized environment (4 parallel environments for faster training)
    env = make_vec_env(
        lambda: PenPickupEnv(render_mode=None),
        n_envs=4,
        seed=0
    )
    
    # Normalize observations and rewards for better training
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: PenPickupEnv(render_mode=None),
        n_envs=1,
        seed=100
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        training=False
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix='ppo_pen_pickup'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    print("Creating PPO model...")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger network
        )
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print("You can monitor training with: tensorboard --logdir logs/")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_pen_pickup_final")
    model.save(final_model_path)
    
    # Save normalization statistics
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(save_dir, 'best_model.zip')}")
    
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for pen pickup')
    parser.add_argument('--timesteps', type=int, default=500000,
                      help='Total training timesteps (default: 500000)')
    parser.add_argument('--save-dir', type=str, default='models',
                      help='Directory to save models (default: models)')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory for logs (default: logs)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pen Pickup RL Training")
    print("=" * 60)
    print(f"Total timesteps: {args.timesteps}")
    print(f"Model directory: {args.save_dir}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 60)
    
    train_agent(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )


if __name__ == "__main__":
    main()
