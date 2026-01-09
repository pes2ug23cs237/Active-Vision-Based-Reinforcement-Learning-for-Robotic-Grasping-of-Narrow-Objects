import numpy as np
import torch
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

from env_1 import RoArmVisionEnv
from vision_sac_agent import VisionSAC, ReplayBuffer


class VisionTrainer:
    """Training manager for vision-based SAC on RoArm"""
    def __init__(self, config):
        self.config = config
        
        # Create environment
        self.env = RoArmVisionEnv(
            render_mode=None,
            max_steps=config['max_episode_steps'],
            image_size=config['image_size'],
            use_depth=config['use_depth']
        )
        
        # Get dimensions
        self.image_channels = 4 if config['use_depth'] else 3
        self.proprioception_dim = 10  # 5 joint pos + 5 joint vel
        self.action_dim = self.env.action_space.shape[0]
        
        # Action bounds
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        if not isinstance(action_low, np.ndarray):
            action_low = np.array(action_low)
        if not isinstance(action_high, np.ndarray):
            action_high = np.array(action_high)
        action_bounds = (action_low, action_high)
        
        # Create SAC agent
        self.agent = VisionSAC(
            image_channels=self.image_channels,
            proprioception_dim=self.proprioception_dim,
            action_dim=self.action_dim,
            action_bounds=action_bounds,
            hidden_dim=config['hidden_dim'],
            lr_actor=config['learning_rate_actor'],
            lr_critic=config['learning_rate_critic'],
            gamma=config['gamma'],
            tau=config['tau'],
            alpha=config['alpha'],
            automatic_entropy_tuning=config['automatic_entropy_tuning']
        )
        
        # Replay buffer
        obs_shape = {
            'image': (self.image_channels, config['image_size'][0], config['image_size'][1]),
            'proprioception': (self.proprioception_dim,)
        }
        self.replay_buffer = ReplayBuffer(
            obs_shape=obs_shape,
            action_dim=self.action_dim,
            max_size=config['buffer_size']
        )
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.distance_history = []
        self.training_losses = {'critic': [], 'actor': [], 'alpha': []}
        
        # Moving averages for logging
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        self.recent_distances = deque(maxlen=100)
        
        # Create directories
        self.model_dir = config['model_dir']
        self.log_dir = config['log_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("Starting Vision-Based SAC Training for RoArm Pen Picking")
        print("=" * 70)
        print(f"Total steps: {self.config['total_steps']}")
        print(f"Expected episodes: ~{self.config['total_steps'] // self.config['max_episode_steps']}")
        print(f"Device: {self.agent.device}")
        print(f"Image size: {self.config['image_size']}")
        print(f"Using depth: {self.config['use_depth']}")
        print(f"Batch size: {self.config['batch_size']}")
        print("=" * 70)
        
        print("\nInitializing environment...")
        obs, _ = self.env.reset()
        print(f"✓ Environment initialized")
        print(f"✓ Image shape: {obs['image'].shape}")
        print(f"✓ Proprioception shape: {obs['proprioception'].shape}")
        print(f"✓ Starting training...\n")
        
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        progress_bar = tqdm(total=self.config['total_steps'], desc="Training")
        
        for step in range(self.config['total_steps']):
            # Select action
            if step < self.config['random_steps']:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(obs, deterministic=False)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Update agent
            if step >= self.config['learning_starts'] and step % self.config['train_freq'] == 0:
                for _ in range(self.config['gradient_steps']):
                    losses = self.agent.update(self.replay_buffer, self.config['batch_size'])
                    
                    # Log losses periodically
                    if step % 1000 == 0:
                        self.training_losses['critic'].append(losses['critic_loss'])
                        self.training_losses['actor'].append(losses['actor_loss'])
                        self.training_losses['alpha'].append(losses['alpha'])
            
            # Episode end
            if done:
                episode_count += 1
                
                # Store episode stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.recent_rewards.append(episode_reward)
                self.recent_successes.append(
                1.0 if (info.get('success', False) or info.get('pen_lifted', False)) else 0.0
                )
                self.recent_distances.append(info.get('closest_distance', float('inf')))
                
                # Calculate moving averages
                avg_reward = np.mean(self.recent_rewards)
                success_rate = np.mean(self.recent_successes)
                avg_distance = np.mean(self.recent_distances)
                
                # Update progress bar
                if episode_count % 5 == 0:
                    progress_bar.set_postfix({
                        'Ep': episode_count,
                        'Reward': f'{avg_reward:.1f}',
                        'Success': f'{success_rate:.2%}',
                        'Dist': f'{avg_distance:.3f}'
                    })
                    
                    # Log to history
                    self.success_rate_history.append(success_rate)
                    self.distance_history.append(avg_distance)
                
                # Detailed logging every 50 episodes
                if episode_count % 50 == 0:
                    print(f"\n[Episode {episode_count}]")
                    print(f"  Steps: {step}")
                    print(f"  Avg Reward (100 ep): {avg_reward:.2f}")
                    print(f"  Success Rate (100 ep): {success_rate:.2%}")
                    print(f"  Avg Distance (100 ep): {avg_distance:.3f}m")
                    print(f"  Buffer Size: {self.replay_buffer.size}")
                
                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            progress_bar.update(1)
            
            # Save model periodically
            if step > 0 and step % self.config['save_freq'] == 0:
                self.save_checkpoint(step)
                print(f"\n✓ Model saved at step {step}")
            
            # Evaluate periodically
            if step > 0 and step % self.config['eval_freq'] == 0:
                eval_stats = self.evaluate()
                print(f"\n{'='*70}")
                print(f"[EVALUATION at step {step}]")
                print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                print(f"  Success Rate: {eval_stats['success_rate']:.2%}")
                print(f"  Mean Distance: {eval_stats['mean_distance']:.3f}m")
                print(f"  Mean Episode Length: {eval_stats['mean_length']:.1f}")
                print(f"{'='*70}\n")
        
        progress_bar.close()
        
        # Final save and evaluation
        print("\n" + "=" * 70)
        print("Training Complete! Running final evaluation...")
        print("=" * 70)
        
        self.save_checkpoint('final')
        final_eval = self.evaluate(num_episodes=50)
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Success Rate: {final_eval['success_rate']:.2%}")
        print(f"Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"Mean Distance: {final_eval['mean_distance']:.3f}m ± {final_eval['std_distance']:.3f}m")
        print(f"Mean Episode Length: {final_eval['mean_length']:.1f}")
        print("=" * 70)
        
        # Plot results
        self.plot_training_curves()
        
        self.env.close()
    
    def evaluate(self, num_episodes=10):
        """Evaluate the current policy"""
        eval_env = RoArmVisionEnv(
            render_mode=None,
            max_steps=self.config['max_episode_steps'],
            image_size=self.config['image_size'],
            use_depth=self.config['use_depth']
        )
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_distances = []
        
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    eval_rewards.append(episode_reward)
                    eval_lengths.append(episode_length)
                    eval_successes.append(1.0 if info.get('success', False) else 0.0)
                    eval_distances.append(info.get('closest_distance', float('inf')))
                    break
        
        eval_env.close()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'mean_distance': np.mean(eval_distances),
            'std_distance': np.std(eval_distances)
        }
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.model_dir, f'model_step_{step}.pt')
        self.agent.save(checkpoint_path)
        
        # Save training stats
        stats_path = os.path.join(self.log_dir, f'stats_step_{step}.npz')
        np.savez(
            stats_path,
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            success_rate_history=np.array(self.success_rate_history),
            distance_history=np.array(self.distance_history)
        )
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
            if len(self.episode_rewards) > 50:
                smoothed = np.convolve(
                    self.episode_rewards, 
                    np.ones(50)/50, 
                    mode='valid'
                )
                axes[0, 0].plot(smoothed, label='Smoothed (50 ep)', linewidth=2)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Success rate
        if self.success_rate_history:
            axes[0, 1].plot(self.success_rate_history, linewidth=2)
            axes[0, 1].set_xlabel('Episode (x5)')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Success Rate (Rolling 100 episodes)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1.05])
        
        # Distance to pen
        if self.distance_history:
            axes[0, 2].plot(self.distance_history, linewidth=2, color='green')
            axes[0, 2].set_xlabel('Episode (x5)')
            axes[0, 2].set_ylabel('Distance (m)')
            axes[0, 2].set_title('Avg Closest Distance to Pen')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.episode_lengths:
            axes[1, 0].plot(self.episode_lengths, alpha=0.3)
            if len(self.episode_lengths) > 50:
                smoothed = np.convolve(
                    self.episode_lengths, 
                    np.ones(50)/50, 
                    mode='valid'
                )
                axes[1, 0].plot(smoothed, linewidth=2)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Length')
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training losses
        if self.training_losses['critic']:
            axes[1, 1].plot(
                self.training_losses['critic'], 
                label='Critic Loss', 
                alpha=0.7
            )
            axes[1, 1].plot(
                self.training_losses['actor'], 
                label='Actor Loss', 
                alpha=0.7
            )
            axes[1, 1].set_xlabel('Update Step (x1000)')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Alpha (entropy coefficient)
        if self.training_losses['alpha']:
            axes[1, 2].plot(
                self.training_losses['alpha'], 
                linewidth=2, 
                color='purple'
            )
            axes[1, 2].set_xlabel('Update Step (x1000)')
            axes[1, 2].set_ylabel('Alpha')
            axes[1, 2].set_title('Entropy Coefficient (Alpha)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training curves saved to: {plot_path}")
        plt.close()


def main():
    """Main function to start training"""
    
    # Training configuration
    config = {
        # Environment
        'max_episode_steps': 300,
        'image_size': (84, 84),
        'use_depth': False,  # Set to True to use RGB-D
        
        # Training
        'total_steps': 1000000,        # Total training steps (increased for vision)
        'random_steps': 10000,         # Random exploration at start
        'learning_starts': 10000,      # When to start learning
        'train_freq': 1,               # Train every N steps
        'gradient_steps': 1,           # Gradient steps per train call
        'batch_size': 128,             # Smaller batch for memory efficiency
        'buffer_size': 50000,          # Smaller buffer for vision (memory)
        
        # SAC parameters
        'hidden_dim': 256,
        'learning_rate_actor': 1e-4,
        'learning_rate_critic': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.5,
        'automatic_entropy_tuning': True,
        
        # Logging
        'save_freq': 15000,
        'eval_freq': 15000,
        'model_dir': '../models',
        'log_dir': '../logs'
    }
    
    print("\n" + "=" * 70)
    print("ROARM VISION-BASED REINFORCEMENT LEARNING")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")
    
    # Create trainer
    trainer = VisionTrainer(config)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        trainer.save_checkpoint('interrupted')
        trainer.plot_training_curves()
        print("✓ Model saved successfully!")
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
