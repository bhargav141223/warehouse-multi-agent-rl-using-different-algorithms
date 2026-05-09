import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """Actor network for MAPPO - outputs action probabilities
    
    Architecture:
    Input: obs_dim = 8 (fixed feature vector)
    Linear(8, 128) -> ReLU
    Linear(128, 64) -> ReLU
    Linear(64, 5) -> Softmax (actions: Up, Down, Left, Right, Stay)
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=None):
        super(ActorNetwork, self).__init__()
        
        # Fixed observation dimension for feature vector representation
        self.obs_dim = 8  # [agent_x, agent_y, goal_x, goal_y, nearest_obstacle_dist, nearest_agent_dist, collision_flag, remaining_steps]
        self.action_dim = action_dim
        self.is_image = False
        
        # Simplified architecture: 8 -> 128 -> 64 -> action_dim
        self.fc1 = nn.Linear(self.obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.pi = nn.Linear(64, action_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(64)
        
    def forward(self, obs):
        # Ensure observation is 2D (batch, obs_dim)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Forward through simplified FC layers: 8 -> 128 -> 64 -> action_dim
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        # Output action probabilities
        logits = self.pi(x)
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs, logits
    
    def get_action(self, obs, deterministic=False):
        """Sample action from the policy"""
        action_probs, logits = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        return action, action_probs


class CriticNetwork(nn.Module):
    """Centralized Critic network for MAPPO - estimates joint value function
    
    Architecture:
    Input: obs_dim * num_agents = 8 * num_agents (concatenated feature vectors from all agents)
    For 4 agents: 32 -> 256 -> 128 -> 1
    Linear(total_obs_dim, 256) -> ReLU
    Linear(256, 128) -> ReLU
    Linear(128, 1) (state value)
    """
    
    def __init__(self, obs_dim, num_agents, hidden_dim=None):
        super(CriticNetwork, self).__init__()
        
        # Fixed observation dimension per agent (feature vector representation)
        self.obs_dim = 8  # [agent_x, agent_y, goal_x, goal_y, nearest_obstacle_dist, nearest_agent_dist, collision_flag, remaining_steps]
        self.num_agents = num_agents
        self.is_image = False
        
        # For centralized critic, we concatenate all agents' observations
        # Each agent contributes 8 features
        self.total_obs_dim = self.obs_dim * num_agents
        
        # Simplified architecture: total_obs_dim -> 256 -> 128 -> 1
        self.fc1 = nn.Linear(self.total_obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.v = nn.Linear(128, 1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)
        
    def forward(self, obs_list):
        """
        obs_list: list of observations from all agents
        Each obs shape: (batch, 8) - fixed 8-dimensional feature vector
        """
        processed_obs = []
        
        for obs in obs_list:
            # Ensure 2D tensor (batch, 8)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            processed_obs.append(obs)
        
        # Concatenate all agents' observations along feature dimension
        # Result: (batch, 8 * num_agents)
        x = torch.cat(processed_obs, dim=-1)
        
        # Forward through simplified FC layers: total_obs_dim -> 256 -> 128 -> 1
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        # Output value estimate
        value = self.v(x)
        
        return value
