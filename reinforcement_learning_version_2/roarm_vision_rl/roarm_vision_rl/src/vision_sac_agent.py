import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ReplayBuffer:
    """Experience replay buffer for vision-based SAC"""
    def __init__(self, obs_shape, action_dim, max_size=50000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # For dictionary observations
        self.images = np.zeros((max_size, *obs_shape['image']), dtype=np.uint8)
        self.proprioception = np.zeros((max_size, *obs_shape['proprioception']), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_images = np.zeros((max_size, *obs_shape['image']), dtype=np.uint8)
        self.next_proprioception = np.zeros((max_size, *obs_shape['proprioception']), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, obs, action, reward, next_obs, done):
        self.images[self.ptr] = obs['image']
        self.proprioception[self.ptr] = obs['proprioception']
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_images[self.ptr] = next_obs['image']
        self.next_proprioception[self.ptr] = next_obs['proprioception']
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, device):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to tensors and normalize images
        images = torch.FloatTensor(self.images[indices]).to(device) / 255.0
        proprioception = torch.FloatTensor(self.proprioception[indices]).to(device)
        actions = torch.FloatTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_images = torch.FloatTensor(self.next_images[indices]).to(device) / 255.0
        next_proprioception = torch.FloatTensor(self.next_proprioception[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return {
            'obs': {'image': images, 'proprioception': proprioception},
            'actions': actions,
            'rewards': rewards,
            'next_obs': {'image': next_images, 'proprioception': next_proprioception},
            'dones': dones
        }


class CNNEncoder(nn.Module):
    """
    Convolutional encoder for processing camera images
    Architecture inspired by DeepMind's DQN
    """
    def __init__(self, input_channels=3, output_dim=256):
        super(CNNEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size (for 84x84 input)
        self.feature_size = 64 * 7 * 7  # After conv layers
        
        # Fully connected layer
        self.fc = nn.Linear(self.feature_size, output_dim)
        
        # Layer normalization for stability
        self.ln = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc(x))
        x = self.ln(x)
        
        return x


class VisionActor(nn.Module):
    """
    Policy network (actor) for vision-based SAC
    Processes images with CNN and combines with proprioception
    """
    def __init__(
        self, 
        image_channels, 
        proprioception_dim, 
        action_dim, 
        hidden_dim=256, 
        action_bounds=None,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(VisionActor, self).__init__()

        self.device = device
        
        # Image encoder
        self.image_encoder = CNNEncoder(image_channels, output_dim=256)
        
        # Proprioception encoder
        self.proprio_fc = nn.Linear(proprioception_dim, 64)
        
        # Combined processing
        combined_dim = 256 + 64  # image features + proprioception
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Action bounds
        if action_bounds is not None:
            self.action_low = torch.FloatTensor(action_bounds[0])
            self.action_high = torch.FloatTensor(action_bounds[1])
        else:
            self.action_low = torch.FloatTensor([-1.0] * action_dim)
            self.action_high = torch.FloatTensor([1.0] * action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs):
        # Extract features from image
        image_features = self.image_encoder(obs['image'])
        
        # Process proprioception
        proprio_features = F.relu(self.proprio_fc(obs['proprioception']))
        
        # Combine features
        combined = torch.cat([image_features, proprio_features], dim=-1)
        
        # Process combined features
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Output mean and log_std
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        # Move action_bounds to same device as action
        action_low = self.action_low.to(action.device)
        action_high = self.action_high.to(action.device)
        
        # Scale to action bounds
        action_scaled = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
        
        # Compute log probability
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(action_high - action_low + 1e-6)
        log_prob -= 2 * (np.log(2) - z - F.softplus(-2 * z))
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action_scaled, log_prob
    
    def get_action(self, obs, deterministic=False):
        """Get action for environment interaction"""
        with torch.no_grad():
            # Convert observation to tensors
            image = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device) / 255.0
            proprioception = torch.FloatTensor(obs['proprioception']).unsqueeze(0).to(self.device)
            
            obs_tensor = {
                'image': image,
                'proprioception': proprioception
            }
            
            mean, log_std = self.forward(obs_tensor)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
            
            # Scale to bounds
            action_low = self.action_low.to(action.device)
            action_high = self.action_high.to(action.device)
            action_scaled = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
            
            return action_scaled.cpu().numpy().flatten()


class VisionCritic(nn.Module):
    """
    Q-network (critic) for vision-based SAC
    Twin Q-networks for stability
    """
    def __init__(self, image_channels, proprioception_dim, action_dim, hidden_dim=256):
        super(VisionCritic, self).__init__()
        
        # Shared image encoder for both Q networks
        self.image_encoder = CNNEncoder(image_channels, output_dim=256)
        
        # Q1 network
        self.proprio_fc1_q1 = nn.Linear(proprioception_dim, 64)
        combined_dim = 256 + 64 + action_dim
        self.fc1_q1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.proprio_fc1_q2 = nn.Linear(proprioception_dim, 64)
        self.fc1_q2 = nn.Linear(combined_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, action):
        # Extract image features
        image_features = self.image_encoder(obs['image'])
        
        # Q1
        proprio_features_q1 = F.relu(self.proprio_fc1_q1(obs['proprioception']))
        x_q1 = torch.cat([image_features, proprio_features_q1, action], dim=-1)
        q1 = F.relu(self.fc1_q1(x_q1))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        # Q2
        proprio_features_q2 = F.relu(self.proprio_fc1_q2(obs['proprioception']))
        x_q2 = torch.cat([image_features, proprio_features_q2, action], dim=-1)
        q2 = F.relu(self.fc1_q2(x_q2))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2


class VisionSAC:
    """
    Soft Actor-Critic algorithm for vision-based control
    """
    def __init__(
        self,
        image_channels,
        proprioception_dim,
        action_dim,
        action_bounds=None,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Networks
        self.actor = VisionActor(
            image_channels, 
            proprioception_dim, 
            action_dim, 
            hidden_dim, 
            action_bounds,
            device=device
        ).to(device)
        
        self.critic = VisionCritic(
            image_channels, 
            proprioception_dim, 
            action_dim, 
            hidden_dim
        ).to(device)
        
        self.critic_target = VisionCritic(
            image_channels, 
            proprioception_dim, 
            action_dim, 
            hidden_dim
        ).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Automatic entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
    
    def select_action(self, obs, deterministic=False):
        return self.actor.get_action(obs, deterministic)
    
    def update(self, replay_buffer, batch_size=128):
        # Sample batch
        batch = replay_buffer.sample(batch_size, self.device)
        
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        for param, target_param in zip(
            self.critic.parameters(), 
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
