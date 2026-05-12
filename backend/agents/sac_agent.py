import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque

class SACActorNetwork(nn.Module):
    """Actor network for SAC (discrete version)"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(SACActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_action_probs(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs


class SACCriticNetwork(nn.Module):
    """Critic network for SAC (Q-function)"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(SACCriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class SACAgent:
    """Soft Actor-Critic Agent (discrete version)"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        tau: float = 0.005,
        target_update_freq: int = 1,
        device: str = 'cpu'
    ):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0
        
        # Actor network
        self.actor = SACActorNetwork(obs_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Two critic networks (Q1 and Q2)
        self.critic1 = SACCriticNetwork(obs_dim, action_dim).to(device)
        self.critic2 = SACCriticNetwork(obs_dim, action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Target critic networks
        self.target_critic1 = SACCriticNetwork(obs_dim, action_dim).to(device)
        self.target_critic2 = SACCriticNetwork(obs_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """Select action using actor network"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor.get_action_probs(obs_tensor, temperature=0.1 if deterministic else 1.0)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            # Calculate log probability
            log_prob = torch.log(action_probs + 1e-8)
            log_prob = log_prob.gather(1, action.unsqueeze(1))
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), action_probs.cpu().numpy()
    
    def store_transition(self, obs, action, reward, value, log_prob, done, next_obs):
        """Store transition in replay buffer"""
        self.replay_buffer.append((obs, action, reward, done, next_obs))
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        obs, actions, rewards, dones, next_obs = zip(*[self.replay_buffer[i] for i in batch])
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        
        # Get current Q-values
        q1_values = self.critic1(obs)
        q2_values = self.critic2(obs)
        
        # Get Q-values for taken actions
        q1_a = q1_values.gather(1, actions.unsqueeze(1)).squeeze()
        q2_a = q2_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_action_probs = self.actor.get_action_probs(next_obs)
            next_q1 = self.target_critic1(next_obs)
            next_q2 = self.target_critic2(next_obs)
            
            # Soft Q-target
            next_q = torch.min(next_q1, next_q2)
            next_q = next_action_probs * (next_q - self.alpha * torch.log(next_action_probs + 1e-8))
            next_q = next_q.sum(dim=-1)
            
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # Critic loss
        critic1_loss = nn.MSELoss()(q1_a, target_q)
        critic2_loss = nn.MSELoss()(q2_a, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic2_optimizer.step()
        
        # Actor loss
        action_probs = self.actor.get_action_probs(obs)
        q1 = self.critic1(obs)
        q2 = self.critic2(obs)
        q = torch.min(q1, q2)
        
        # SAC actor loss
        actor_loss = (action_probs * (self.alpha * torch.log(action_probs + 1e-8) - q)).sum(dim=-1).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        
        total_loss = actor_loss.item() + critic1_loss.item() + critic2_loss.item()
        self.episode_losses.append(total_loss)
        
        return {
            "loss": total_loss,
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item()
        }
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
