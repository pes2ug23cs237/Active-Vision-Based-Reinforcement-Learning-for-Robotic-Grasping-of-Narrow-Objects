import numpy as np
import torch
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from roarm_env import RoArmPickEnv
from sac_agent import SAC, ReplayBuffer


class Trainer:
    """Training manager for SAC on RoArm"""
    def __init__(self, config):
        self.config = config
        
        # Create environment
        self.env = RoArmPickEnv(render_mode=None, max_steps=config['max_episode_steps'])
        
        # Get dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Convert action bounds to numpy arrays if needed
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        if not isinstance(action_low, np.ndarray):
            action_low = np.array(action_low)
        if not isinstance(action_high, np.ndarray):
            action_high = np.array(action_high)
        
        action_bounds = (action_low, action_high)
        
        # Create SAC agent
        self.agent = SAC(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_bounds=action_bounds,
            hidden_dim=config['hidden_dim'],
            lr=config['learning_rate'],
            gamma=config['gamma'],
            tau=config['tau'],
            alpha=config['alpha'],
            automatic_entropy_tuning=config['automatic_entropy_tuning']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.obs_dim, 
            self.action_dim, 
            max_size=config['buffer_size']
        )
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.training_losses = {'critic': [], 'actor': [], 'alpha': []}
        
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
        print("=" * 60)
        print("Starting SAC Training for RoArm Pen Picking")
        print("=" * 60)
        print(f"Total steps: {self.config['total_steps']}")
        print(f"Episodes: ~{self.config['total_steps'] // self.config['max_episode_steps']}")
        print(f"Device: {self.agent.device}")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print("=" * 60)
        
        print("\nInitializing environment...")
        obs, _ = self.env.reset()
        print(f"✓ Environment initialized successfully")
        print(f"✓ Initial observation shape: {obs.shape}")
        print(f"✓ Starting training...\n")
        
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        recent_successes = []
        
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
                    
                    if step % 1000 == 0:
                        self.training_losses['critic'].append(losses['critic_loss'])
                        self.training_losses['actor'].append(losses['actor_loss'])
                        self.training_losses['alpha'].append(losses['alpha'])
            
            # Episode end
            if done:
                episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                recent_successes.append(1.0 if info.get('success', False) else 0.0)
                
                # Keep only recent successes for success rate
                if len(recent_successes) > 100:
                    recent_successes.pop(0)
                
                success_rate = np.mean(recent_successes) if recent_successes else 0.0
                
                # Log progress
                if episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    progress_bar.set_postfix({
                        'Episode': episode_count,
                        'Avg Reward': f'{avg_reward:.1f}',
                        'Success Rate': f'{success_rate:.2%}',
                        'Avg Length': f'{avg_length:.0f}'
                    })
                    self.success_rate_history.append(success_rate)
                
                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            progress_bar.update(1)
            
            # Save model periodically
            if step > 0 and step % self.config['save_freq'] == 0:
                self.save_checkpoint(step)
            
            # Evaluate periodically
            if step > 0 and step % self.config['eval_freq'] == 0:
                eval_stats = self.evaluate()
                print(f"\n[Evaluation at step {step}]")
                print(f"  Mean Reward: {eval_stats['mean_reward']:.2f}")
                print(f"  Success Rate: {eval_stats['success_rate']:.2%}")
                print(f"  Mean Episode Length: {eval_stats['mean_length']:.1f}")
        
        progress_bar.close()
        
        # Final save and evaluation
        self.save_checkpoint('final')
        final_eval = self.evaluate(num_episodes=50)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final Success Rate: {final_eval['success_rate']:.2%}")
        print(f"Final Mean Reward: {final_eval['mean_reward']:.2f}")
        print("=" * 60)
        
        # Plot results
        self.plot_training_curves()
        
        self.env.close()
    
    def evaluate(self, num_episodes=10):
        """Evaluate the current policy"""
        eval_env = RoArmPickEnv(render_mode=None, max_steps=self.config['max_episode_steps'])
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
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
                    break
        
        eval_env.close()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes)
        }
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.model_dir, f'model_step_{step}.pt')
        self.agent.save(checkpoint_path)
        
        # Also save training stats
        stats_path = os.path.join(self.log_dir, f'stats_step_{step}.npz')
        np.savez(
            stats_path,
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            success_rate_history=np.array(self.success_rate_history)
        )
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
            if len(self.episode_rewards) > 20:
                smoothed = np.convolve(self.episode_rewards, np.ones(20)/20, mode='valid')
                axes[0, 0].plot(smoothed, label='Smoothed (20 episodes)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Success rate
        if self.success_rate_history:
            axes[0, 1].plot(self.success_rate_history)
            axes[0, 1].set_xlabel('Episode (x10)')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Success Rate (Last 100 episodes)')
            axes[0, 1].grid(True)
        
        # Episode lengths
        if self.episode_lengths:
            axes[1, 0].plot(self.episode_lengths, alpha=0.3)
            if len(self.episode_lengths) > 20:
                smoothed = np.convolve(self.episode_lengths, np.ones(20)/20, mode='valid')
                axes[1, 0].plot(smoothed)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Length')
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].grid(True)
        
        # Losses
        if self.training_losses['critic']:
            axes[1, 1].plot(self.training_losses['critic'], label='Critic Loss', alpha=0.7)
            axes[1, 1].plot(self.training_losses['actor'], label='Actor Loss', alpha=0.7)
            axes[1, 1].set_xlabel('Update Step (x1000)')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to: {plot_path}")
        plt.close()


def main():
    """Main function to start training"""
    
    # Training configuration
    config = {
        # Environment
        'max_episode_steps': 250,
        
        # Training
        'total_steps': 100000,        # Total training steps
        'random_steps': 5000,          # Random exploration steps at start
        'learning_starts': 5000,       # When to start learning
        'train_freq': 1,               # Train every N steps
        'gradient_steps': 1,           # Gradient steps per train call
        'batch_size': 256,             # Batch size for training
        'buffer_size': 100000,         # Replay buffer size
        
        # SAC parameters
        'hidden_dim': 256,             # Hidden layer size
        'learning_rate': 3e-4,         # Learning rate
        'gamma': 0.99,                 # Discount factor
        'tau': 0.005,                  # Target network update rate
        'alpha': 0.2,                  # Entropy temperature
        'automatic_entropy_tuning': True,  # Auto-tune alpha
        
        # Logging
        'save_freq': 10000,            # Save model every N steps
        'eval_freq': 10000,            # Evaluate every N steps
        'model_dir': '../models',
        'log_dir': '../logs'
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()