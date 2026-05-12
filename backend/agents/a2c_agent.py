import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque

class ActorNetwork(nn.Module):
    """Actor network for A2C"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, x, deterministic=False):
        probs = self.forward(x)
        if deterministic:
            action = probs.argmax(dim=-1)
            return action, probs
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action, probs


class CriticNetwork(nn.Module):
    """Critic network for A2C"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class A2CAgent:
    """Advantage Actor-Critic Agent"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Actor and Critic networks
        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(obs_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_observations': []
        }
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """Select action using actor network"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, action_probs = self.actor.get_action(obs_tensor, deterministic)
            
            # Calculate log probability
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)
            
            # Get value
            value = self.critic(obs_tensor)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), action_probs.cpu().numpy()
    
    def store_transition(self, obs, action, reward, value, log_prob, done, next_obs):
        """Store transition in buffer"""
        self.buffer['observations'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
        self.buffer['next_observations'].append(next_obs)
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.buffer['observations']) == 0:
            return {"loss": 0.0}
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array(self.buffer['observations'])).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer['actions'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        
        # Calculate returns
        returns = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get current values
        values = self.critic(obs).squeeze()
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Calculate actor loss
        action_probs = self.actor(obs)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        # Policy gradient loss
        actor_loss = -(new_log_probs * advantages).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - self.entropy_coef * entropy
        
        # Calculate critic loss
        critic_loss = nn.MSELoss()(values, returns)
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Clear buffer
        self.clear_buffer()
        
        self.episode_losses.append(total_loss.item())
        
        return {
            "loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }
    
    def clear_buffer(self):
        """Clear the experience buffer"""
        for key in self.buffer:
            self.buffer[key] = []
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
