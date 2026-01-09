import numpy as np
import torch
import argparse
import os
import time

from roarm_env import RoArmPickEnv
from sac_agent import SAC


def evaluate_model(model_path, num_episodes=10, render=True, deterministic=True):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
    """
    
    print("=" * 60)
    print("Evaluating Trained RoArm Agent")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)
    
    # Create environment
    render_mode = "human" if render else None
    env = RoArmPickEnv(render_mode=render_mode, max_steps=250)
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = (env.action_space.low, env.action_space.high)
    
    # Create agent
    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_bounds=action_bounds
    )
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    agent.load(model_path)
    print(f"Model loaded successfully!")
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    successes = []
    distances = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_distances = []
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        print(f"Initial distance to pen: {info['distance_to_pen']:.3f}m")
        
        while True:
            # Select action
            action = agent.select_action(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            episode_distances.append(info['distance_to_pen'])
            
            # Print progress every 50 steps
            if episode_length % 50 == 0:
                print(f"  Step {episode_length}: Distance = {info['distance_to_pen']:.3f}m, "
                      f"Cumulative Reward = {episode_reward:.2f}")
            
            if render:
                time.sleep(1./240.)  # Match simulation speed
            
            if terminated or truncated:
                success = info.get('success', False)
                successes.append(1.0 if success else 0.0)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                distances.append(np.mean(episode_distances))
                
                print(f"Episode finished!")
                print(f"  Success: {success}")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Episode Length: {episode_length} steps")
                print(f"  Final Distance: {info['distance_to_pen']:.3f}m")
                print(f"  Avg Distance: {np.mean(episode_distances):.3f}m")
                break
    
    env.close()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Success Rate: {np.mean(successes):.2%} ({np.sum(successes):.0f}/{num_episodes})")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean Distance: {np.mean(distances):.3f}m ± {np.std(distances):.3f}m")
    print("=" * 60)
    
    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_distance': np.mean(distances)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RoArm agent')
    parser.add_argument('--model', type=str, default='../models/model_step_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to evaluate')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy instead of deterministic')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
