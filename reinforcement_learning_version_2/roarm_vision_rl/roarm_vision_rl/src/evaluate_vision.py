import numpy as np
import torch
import argparse
import os
import time
import cv2
import pybullet as p

from roarm_vision_env import RoArmVisionEnv
from vision_sac_agent import VisionSAC


def evaluate_model(
    model_path, 
    num_episodes=10, 
    render=True, 
    deterministic=True, 
    save_video=False,
    use_depth=False
):
    """
    Evaluate a trained vision-based model
    
    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        save_video: Whether to save episode videos
        use_depth: Whether model uses depth images
    """
    
    print("=" * 70)
    print("Evaluating Trained Vision-Based RoArm Agent")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Deterministic: {deterministic}")
    print(f"Using depth: {use_depth}")
    print("=" * 70)
    
    # Create environment
    render_mode = "human" if render else None
    env = RoArmVisionEnv(
        render_mode=render_mode,
        max_steps=300,
        image_size=(84, 84),
        use_depth=use_depth
    )

    env.reset_count = 1000  # Skip initial resets for evaluation
    
    # Get dimensions
    image_channels = 4 if use_depth else 3
    proprioception_dim = 8
    action_dim = env.action_space.shape[0]
    action_bounds = (env.action_space.low, env.action_space.high)
    
    # Create agent
    agent = VisionSAC(
        image_channels=image_channels,
        proprioception_dim=proprioception_dim,
        action_dim=action_dim,
        action_bounds=action_bounds
    )
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    agent.load(model_path)
    print(f"✓ Model loaded successfully!")
    print()
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    successes = []
    closest_distances = []
    final_distances = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Video recording
        frames = []
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        print(f"Initial distance to pen: {info['distance_to_pen']:.3f}m")
        
        while True:
            # Select action
            action = agent.select_action(obs, deterministic=deterministic)
            
            # joint_states = [p.getJointState(env.robot_id, i)[0] for i in range(4)]
            # print(f"Step {episode_length}: Joints={[f'{j:.2f}' for j in joint_states]}, Action={[f'{a:.2f}' for a in action]}")

            # === ADD DEBUG PRINTS HERE ===
            # Get current joint states
            joint_states = [p.getJointState(env.robot_id, i, physicsClientId=env.physics_client)[0] for i in range(4)]
            
            # Get pen position
            pen_pos, _ = p.getBasePositionAndOrientation(env.pen_id, physicsClientId=env.physics_client)
            
            # Get gripper position
            link_state = p.getLinkState(env.robot_id, env.tcp_link_idx, physicsClientId=env.physics_client)
            gripper_pos = link_state[0]
            
            # Calculate distance
            distance = np.linalg.norm(np.array(gripper_pos) - np.array(pen_pos))
            
            # Print every 20 steps (not too spammy)
            if episode_length % 20 == 0:
                print(f"\nStep {episode_length}:")
                print(f"  Joints: [{joint_states[0]:.2f}, {joint_states[1]:.2f}, {joint_states[2]:.2f}, {joint_states[3]:.2f}]")
                print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]")
                print(f"  Pen Pos: [{pen_pos[0]:.3f}, {pen_pos[1]:.3f}, {pen_pos[2]:.3f}]")
                print(f"  Gripper Pos: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
                print(f"  Distance: {distance:.3f}m")
            # === END DEBUG PRINTS ===


            # Save frame if recording
            if save_video:
                # Get the camera image from observation
                img = obs['image'].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                if use_depth:
                    img = img[:, :, :3]  # Only RGB channels
                frames.append((img * 255).astype(np.uint8))
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Print progress every 50 steps
            if episode_length % 50 == 0:
                print(f"  Step {episode_length:3d}: "
                      f"Distance = {info['distance_to_pen']:.4f}m, "
                      f"Reward = {episode_reward:7.2f}")
            
            if render:
                time.sleep(1./240.)
            
            if terminated or truncated:
                success = info.get('success', False)
                closest_dist = info.get('closest_distance', float('inf'))
                
                successes.append(1.0 if success else 0.0)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                closest_distances.append(closest_dist)
                final_distances.append(info['distance_to_pen'])
                
                print(f"\n{'─'*70}")
                print(f"Episode Complete!")
                print(f"  Success: {'✓ YES' if success else '✗ NO'}")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Episode Length: {episode_length} steps")
                print(f"  Closest Distance: {closest_dist:.4f}m")
                print(f"  Final Distance: {info['distance_to_pen']:.4f}m")
                print(f"{'─'*70}")
                
                # Save video if requested
                if save_video and frames:
                    video_dir = '../videos'
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(
                        video_dir, 
                        f'episode_{episode+1}_{"success" if success else "fail"}.mp4'
                    )
                    save_video_file(frames, video_path)
                    print(f"  ✓ Video saved: {video_path}")
                
                break
    
    env.close()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Success Rate: {np.mean(successes):.2%} "
          f"({int(np.sum(successes))}/{num_episodes})")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± "
          f"{np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± "
          f"{np.std(episode_lengths):.1f}")
    print(f"Mean Closest Distance: {np.mean(closest_distances):.4f}m ± "
          f"{np.std(closest_distances):.4f}m")
    print(f"Mean Final Distance: {np.mean(final_distances):.4f}m ± "
          f"{np.std(final_distances):.4f}m")
    print("=" * 70)
    
    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_closest_distance': np.mean(closest_distances),
        'mean_final_distance': np.mean(final_distances)
    }


def save_video_file(frames, output_path, fps=30):
    """Save frames as video file"""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained vision-based RoArm agent'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='../models/model_step_final.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=10,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--no-render', 
        action='store_true',
        help='Disable rendering'
    )
    parser.add_argument(
        '--stochastic', 
        action='store_true',
        help='Use stochastic policy instead of deterministic'
    )
    parser.add_argument(
        '--save-video', 
        action='store_true',
        help='Save episode videos'
    )
    parser.add_argument(
        '--use-depth', 
        action='store_true',
        help='Model uses RGB-D images'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        save_video=args.save_video,
        use_depth=args.use_depth
    )


if __name__ == "__main__":
    main()
