import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Deep Q-Network for DQN"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0
        
        # Q-network and target network
        self.q_network = DQNNetwork(obs_dim, action_dim).to(device)
        self.target_network = DQNNetwork(obs_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """Select action using epsilon-greedy policy"""
        if not deterministic and random.random() < self.epsilon:
            return np.random.randint(self.action_dim), 0.0, None
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action, 0.0, None
    
    def store_transition(self, obs, action, reward, value, log_prob, done, next_obs):
        """Store transition in replay buffer"""
        self.replay_buffer.append((obs, action, reward, done, next_obs))
    
    def update(self):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, dones, next_obs = zip(*batch)
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        
        # Compute Q-values
        current_q_values = self.q_network(obs).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_obs).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.episode_losses.append(loss.item())
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
