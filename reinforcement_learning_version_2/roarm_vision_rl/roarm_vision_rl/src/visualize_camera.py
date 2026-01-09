#!/usr/bin/env python3
"""
Visualize the camera view from the robot gripper
Useful for debugging and understanding what the agent sees
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

from roarm_vision_env import RoArmVisionEnv


def visualize_camera(use_depth=False, save_frames=False):
    """
    Visualize the camera view in real-time
    
    Args:
        use_depth: Whether to show depth channel
        save_frames: Whether to save frames to disk
    """
    print("=" * 70)
    print("Camera View Visualization")
    print("=" * 70)
    print("Controls:")
    print("  - Actions are random")
    print("  - Close window to exit")
    print("  - Press Ctrl+C to stop")
    print("=" * 70)
    
    # Create environment
    env = RoArmVisionEnv(
        render_mode="human",
        max_steps=500,
        image_size=(84, 84),
        use_depth=use_depth
    )
    
    # Setup plot
    if use_depth:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Robot Camera View (RGB + Depth)', fontsize=14)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        fig.suptitle('Robot Camera View (RGB)', fontsize=14)
        axes = [axes]
    
    # Initialize environment
    obs, info = env.reset()
    
    # Create initial image displays
    if use_depth:
        # RGB
        rgb_img = obs['image'][:3].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        im_rgb = axes[0].imshow(rgb_img)
        axes[0].set_title('RGB')
        axes[0].axis('off')
        
        # Depth
        depth_img = obs['image'][3]  # Depth channel
        im_depth = axes[1].imshow(depth_img, cmap='viridis')
        axes[1].set_title('Depth')
        axes[1].axis('off')
        plt.colorbar(im_depth, ax=axes[1])
    else:
        # RGB only
        rgb_img = obs['image'].transpose(1, 2, 0)
        im_rgb = axes[0].imshow(rgb_img)
        axes[0].axis('off')
    
    # Add text for proprioception
    text_str = f"Joint Positions: {obs['proprioception'][:4]}\n"
    text_str += f"Distance to Pen: {info['distance_to_pen']:.3f}m"
    text_box = fig.text(0.5, 0.02, text_str, ha='center', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    frame_count = 0
    
    def update_frame(frame):
        nonlocal obs, frame_count
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update images
        if use_depth:
            rgb_img = obs['image'][:3].transpose(1, 2, 0)
            depth_img = obs['image'][3]
            im_rgb.set_array(rgb_img)
            im_depth.set_array(depth_img)
        else:
            rgb_img = obs['image'].transpose(1, 2, 0)
            im_rgb.set_array(rgb_img)
        
        # Update text
        text_str = f"Step: {frame_count}\n"
        text_str += f"Joint Pos: [{obs['proprioception'][0]:.2f}, "
        text_str += f"{obs['proprioception'][1]:.2f}, "
        text_str += f"{obs['proprioception'][2]:.2f}, "
        text_str += f"{obs['proprioception'][3]:.2f}]\n"
        text_str += f"Distance to Pen: {info['distance_to_pen']:.3f}m\n"
        text_str += f"Reward: {reward:.2f}"
        if info.get('success', False):
            text_str += " - SUCCESS!"
        text_box.set_text(text_str)
        
        # Save frame if requested
        if save_frames and frame_count % 10 == 0:
            plt.savefig(f'../data/camera_frame_{frame_count:04d}.png', dpi=100)
        
        frame_count += 1
        
        # Reset if episode ends
        if terminated or truncated:
            obs, info = env.reset()
            print(f"Episode ended at step {frame_count}")
            frame_count = 0
        
        if use_depth:
            return [im_rgb, im_depth, text_box]
        else:
            return [im_rgb, text_box]
    
    # Create animation
    anim = FuncAnimation(fig, update_frame, interval=50, blit=True, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        env.close()
        plt.close()


def save_sample_images(num_samples=10, use_depth=False):
    """Save sample camera images to disk"""
    print(f"Saving {num_samples} sample camera images...")
    
    import os
    os.makedirs('../data', exist_ok=True)
    
    env = RoArmVisionEnv(
        render_mode=None,
        max_steps=500,
        image_size=(84, 84),
        use_depth=use_depth
    )
    
    for i in range(num_samples):
        obs, info = env.reset()
        
        # Take a few random steps to get interesting views
        for _ in range(np.random.randint(10, 50)):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
        
        # Save the image
        fig, ax = plt.subplots(figsize=(6, 6))
        if use_depth:
            rgb_img = obs['image'][:3].transpose(1, 2, 0)
        else:
            rgb_img = obs['image'].transpose(1, 2, 0)
        
        ax.imshow(rgb_img)
        ax.axis('off')
        ax.set_title(f"Sample {i+1} - Distance: {info['distance_to_pen']:.3f}m")
        
        plt.tight_layout()
        plt.savefig(f'../data/sample_view_{i+1:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved sample {i+1}/{num_samples}")
    
    env.close()
    print(f"✓ Saved {num_samples} images to ../data/")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize robot camera view'
    )
    parser.add_argument(
        '--use-depth',
        action='store_true',
        help='Show depth channel'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save frames to disk'
    )
    parser.add_argument(
        '--save-samples',
        type=int,
        metavar='N',
        help='Save N sample images and exit'
    )
    
    args = parser.parse_args()
    
    if args.save_samples:
        save_sample_images(args.save_samples, args.use_depth)
    else:
        visualize_camera(args.use_depth, args.save_frames)


if __name__ == "__main__":
    main()
