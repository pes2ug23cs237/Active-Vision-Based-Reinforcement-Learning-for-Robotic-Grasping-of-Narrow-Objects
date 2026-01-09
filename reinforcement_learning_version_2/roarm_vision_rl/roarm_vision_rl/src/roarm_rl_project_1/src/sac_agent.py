import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    def __init__(self, obs_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.obs[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_obs[indices]),
            torch.FloatTensor(self.dones[indices])
        )


class Actor(nn.Module):
    """Policy network (actor) for SAC"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, action_bounds=None):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Action bounds for scaling
        if action_bounds is not None:
            self.action_low = torch.FloatTensor(action_bounds[0])
            self.action_high = torch.FloatTensor(action_bounds[1])
        else:
            self.action_low = torch.FloatTensor([-1.0] * action_dim)
            self.action_high = torch.FloatTensor([1.0] * action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)
        
        # Scale action to actual bounds
        action_scaled = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        
        # Compute log probability
        log_prob = normal.log_prob(z)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_high - self.action_low + 1e-6)
        log_prob -= 2 * (np.log(2) - z - F.softplus(-2 * z))
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action_scaled, log_prob
    
    def get_action(self, obs, deterministic=False):
        """Get action for environment interaction"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(0)
            mean, log_std = self.forward(obs)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
            
            # Scale to bounds
            action_scaled = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
            
            return action_scaled.cpu().numpy().flatten()


class Critic(nn.Module):
    """Q-network (critic) for SAC"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1_q1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc1_q2 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        # Q2
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2
    
    def q1(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        return q1


class SAC:
    """Soft Actor-Critic algorithm"""
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        action_bounds=None,
        hidden_dim=256,
        lr=3e-4,
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
        self.actor = Actor(obs_dim, action_dim, hidden_dim, action_bounds).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, obs, deterministic=False):
        return self.actor.get_action(obs, deterministic)
    
    def update(self, replay_buffer, batch_size=256):
        # Sample batch
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
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
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature parameter)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
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
