"""
Test and Visualize Trained Agent
"""

import os
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pen_pickup_env import PenPickupEnv
import argparse


def test_agent(model_path, num_episodes=10, render=True, record_video=False):
    """
    Test trained agent
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of test episodes
        render: Whether to render visualization
        record_video: Whether to record video
    """
    
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = PenPickupEnv(render_mode=render_mode)
    
    # Wrap in DummyVecEnv for normalization
    env = DummyVecEnv([lambda: env])
    
    # Load normalization statistics if they exist
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print(f"Loading normalization stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during testing
        env.norm_reward = False
    
    # Statistics
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"\nTesting for {num_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Print progress every 50 steps
            if episode_length % 50 == 0:
                print(f"  Step {episode_length}: Distance={info[0]['distance_to_pen']:.4f}m, "
                      f"Pen Height={info[0]['pen_height']:.4f}m")
            
            if render:
                time.sleep(1./240.)  # Match physics timestep
            
            if done:
                success = info[0]['success']
                success_count += success
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"  Episode ended at step {episode_length}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Success: {'✓ YES' if success else '✗ NO'}")
                print(f"  Final distance: {info[0]['distance_to_pen']:.4f}m")
                print(f"  Final pen height: {info[0]['pen_height']:.4f}m")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)
    
    env.close()


def test_random_agent(num_episodes=5):
    """Test with random actions to compare baseline"""
    
    print("Testing RANDOM agent (baseline)...")
    print("=" * 60)
    
    env = PenPickupEnv(render_mode="human")
    
    success_count = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done and step < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}: Distance={info['distance_to_pen']:.4f}m, "
                      f"Pen Height={info['pen_height']:.4f}m")
        
        success = info['success']
        success_count += success
        total_rewards.append(episode_reward)
        
        print(f"  Success: {'✓ YES' if success else '✗ NO'}")
        print(f"  Total reward: {episode_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("RANDOM AGENT SUMMARY")
    print("=" * 60)
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print("=" * 60)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Test trained pen pickup agent')
    parser.add_argument('--model', type=str, default='models/best_model.zip',
                      help='Path to trained model (default: models/best_model.zip)')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of test episodes (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                      help='Disable rendering')
    parser.add_argument('--random', action='store_true',
                      help='Test random agent instead of trained agent')
    
    args = parser.parse_args()
    
    if args.random:
        test_random_agent(num_episodes=args.episodes)
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model not found at {args.model}")
            print("Please train a model first using train.py")
            return
        
        test_agent(
            model_path=args.model,
            num_episodes=args.episodes,
            render=not args.no_render
        )


if __name__ == "__main__":
    main()
