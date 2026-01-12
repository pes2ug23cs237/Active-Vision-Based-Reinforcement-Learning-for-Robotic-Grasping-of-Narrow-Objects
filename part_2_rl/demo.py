"""
Quick Demo - Visualize the Environment and Task
"""

from pen_pickup_env import PenPickupEnv
import time


def demo():
    """Quick demo of the pen pickup environment"""
    
    print("=" * 60)
    print("PEN PICKUP ENVIRONMENT DEMO")
    print("=" * 60)
    print()
    print("This demo shows:")
    print("  1. Robot arm starting 3cm from pen")
    print("  2. Random actions (untrained behavior)")
    print("  3. Reward feedback")
    print()
    print("After training, the robot will learn to:")
    print("  - Approach the pen smoothly")
    print("  - Close gripper around pen")
    print("  - Lift pen above 15cm height")
    print()
    print("Press Ctrl+C to stop the demo")
    print("=" * 60)
    
    # Create environment with visualization
    env = PenPickupEnv(render_mode="human")
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n--- Episode {episode} ---")
            
            # Reset environment
            obs, info = env.reset()
            print(f"Initial distance to pen: {info['distance_to_pen']:.4f}m")
            print(f"Initial pen height: {info['pen_height']:.4f}m")
            print()
            
            total_reward = 0
            step = 0
            done = False
            
            while not done and step < 500:
                # Random action (replace with trained model later)
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                step += 1
                
                # Print progress
                if step % 100 == 0:
                    print(f"Step {step}:")
                    print(f"  Distance to pen: {info['distance_to_pen']:.4f}m")
                    print(f"  Pen height: {info['pen_height']:.4f}m")
                    print(f"  Cumulative reward: {total_reward:.2f}")
                
                time.sleep(1./240.)  # Match physics timestep
            
            # Episode summary
            print(f"\nEpisode {episode} Summary:")
            print(f"  Steps: {step}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Success: {'YES ✓' if info['success'] else 'NO ✗'}")
            print(f"  Final distance: {info['distance_to_pen']:.4f}m")
            print(f"  Final pen height: {info['pen_height']:.4f}m")
            
            # Wait a bit before next episode
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    demo()
