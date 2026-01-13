"""
Training Configuration
Modify these parameters to customize training behavior
"""

# Training Configuration
TRAINING_CONFIG = {
    # Total training timesteps
    'total_timesteps': 500000,
    
    # Number of parallel environments (more = faster training, more RAM)
    'n_envs': 4,
    
    # Random seed for reproducibility
    'seed': 0,
    
    # Save directories
    'save_dir': 'models',
    'log_dir': 'logs',
}

# PPO Hyperparameters
PPO_CONFIG = {
    # Learning rate
    'learning_rate': 3e-4,
    
    # Number of steps to run for each environment per update
    'n_steps': 2048,
    
    # Minibatch size
    'batch_size': 64,
    
    # Number of epochs when optimizing the surrogate loss
    'n_epochs': 10,
    
    # Discount factor
    'gamma': 0.99,
    
    # Factor for trade-off of bias vs variance for GAE
    'gae_lambda': 0.95,
    
    # Clipping parameter
    'clip_range': 0.2,
    
    # Entropy coefficient for the loss calculation
    'ent_coef': 0.01,
    
    # Value function coefficient for the loss calculation
    'vf_coef': 0.5,
    
    # The maximum value for the gradient clipping
    'max_grad_norm': 0.5,
    
    # Network architecture
    'policy_kwargs': {
        'net_arch': dict(
            pi=[256, 256],  # Policy network: 2 layers of 256 units
            vf=[256, 256]   # Value network: 2 layers of 256 units
        )
    }
}

# Environment Configuration
ENV_CONFIG = {
    # Maximum steps per episode
    'max_steps': 500,
    
    # Initial distance from pen (meters)
    'distance_from_pen': 0.03,  # 3cm
    
    # Height threshold for successful pickup (meters)
    'pickup_height_threshold': 0.15,  # 15cm
    
    # Pen initial height (meters)
    'pen_initial_height': 0.05,  # 5cm
}

# Callback Configuration
CALLBACK_CONFIG = {
    # How often to save checkpoints (in timesteps)
    'checkpoint_freq': 10000,
    
    # How often to evaluate (in timesteps)
    'eval_freq': 5000,
    
    # Number of episodes for evaluation
    'n_eval_episodes': 5,
}

# Observation Normalization
NORMALIZE_CONFIG = {
    # Normalize observations
    'norm_obs': True,
    
    # Normalize rewards
    'norm_reward': True,
    
    # Clip observations to this range
    'clip_obs': 10.0,
    
    # Clip rewards to this range
    'clip_reward': 10.0,
}


# Preset Configurations for Different Scenarios

PRESETS = {
    'quick_test': {
        'total_timesteps': 50000,
        'n_envs': 2,
        'checkpoint_freq': 5000,
        'eval_freq': 2500,
    },
    
    'standard': {
        'total_timesteps': 500000,
        'n_envs': 4,
        'checkpoint_freq': 10000,
        'eval_freq': 5000,
    },
    
    'long_training': {
        'total_timesteps': 2000000,
        'n_envs': 8,
        'checkpoint_freq': 20000,
        'eval_freq': 10000,
    },
    
    'high_performance': {
        'total_timesteps': 1000000,
        'n_envs': 4,
        'learning_rate': 1e-4,  # Lower learning rate for stability
        'n_steps': 4096,        # Longer rollouts
        'batch_size': 128,      # Larger batches
        'policy_kwargs': {
            'net_arch': dict(
                pi=[512, 512, 256],  # Deeper network
                vf=[512, 512, 256]
            )
        }
    },
}


def get_config(preset='standard'):
    """
    Get configuration for training
    
    Args:
        preset: Configuration preset name
        
    Returns:
        Dictionary with all configurations
    """
    config = {
        'training': TRAINING_CONFIG.copy(),
        'ppo': PPO_CONFIG.copy(),
        'env': ENV_CONFIG.copy(),
        'callback': CALLBACK_CONFIG.copy(),
        'normalize': NORMALIZE_CONFIG.copy(),
    }
    
    # Apply preset overrides
    if preset in PRESETS:
        preset_config = PRESETS[preset]
        for key, value in preset_config.items():
            # Find which config dict this belongs to
            for config_dict in config.values():
                if key in config_dict:
                    config_dict[key] = value
                    break
    
    return config


if __name__ == "__main__":
    # Print available presets
    print("Available Configuration Presets:")
    print("=" * 60)
    
    for preset_name, preset_config in PRESETS.items():
        print(f"\n{preset_name}:")
        for key, value in preset_config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("\nTo use a preset in train.py:")
    print("  from config import get_config")
    print("  config = get_config('quick_test')")
