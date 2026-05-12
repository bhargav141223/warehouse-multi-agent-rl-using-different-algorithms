import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy

from .mappo import MAPPOAgent, MAPPOTrainer
from .dqn_agent import DQNAgent
from .a2c_agent import A2CAgent
from .ppo_agent import PPOAgent
from .sac_agent import SACAgent


class HeterogeneousTrainer:
    """Trainer for heterogeneous multi-agent system with different algorithms"""
    
    def __init__(
        self,
        env,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        algorithm_config: List[str],  # List of algorithm names for each agent
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algorithm_config = algorithm_config
        self.device = device
        
        # Initialize agents with different algorithms
        self.agents = []
        self.agent_types = []
        
        for i in range(num_agents):
            algo = algorithm_config[i] if i < len(algorithm_config) else 'mappo'
            self.agent_types.append(algo)
            
            if algo == 'mappo':
                # MAPPO agent (will be managed by MAPPOTrainer)
                agent = MAPPOAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_agents=num_agents,
                    agent_id=i,
                    lr=lr,
                    gamma=gamma,
                    device=device
                )
                self.agents.append(agent)
            elif algo == 'dqn':
                agent = DQNAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    agent_id=i,
                    lr=lr,
                    gamma=gamma,
                    device=device
                )
                self.agents.append(agent)
            elif algo == 'a2c':
                agent = A2CAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    agent_id=i,
                    lr=lr,
                    gamma=gamma,
                    device=device
                )
                self.agents.append(agent)
            elif algo == 'ppo':
                agent = PPOAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    agent_id=i,
                    lr=lr,
                    gamma=gamma,
                    device=device
                )
                self.agents.append(agent)
            elif algo == 'sac':
                agent = SACAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    agent_id=i,
                    lr=lr,
                    gamma=gamma,
                    device=device
                )
                self.agents.append(agent)
            else:
                # Default to MAPPO
                agent = MAPPOAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_agents=num_agents,
                    agent_id=i,
                    lr=lr,
                    gamma=gamma,
                    device=device
                )
                self.agents.append(agent)
        
        # Initialize MAPPO trainer if any MAPPO agents exist
        self.mappo_trainer = None
        if 'mappo' in algorithm_config:
            # Create centralized critic for MAPPO agents
            from .networks import CriticNetwork
            critic = CriticNetwork(obs_dim, num_agents).to(device)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
            
            # Set critic for MAPPO agents
            for i, agent in enumerate(self.agents):
                if self.agent_types[i] == 'mappo':
                    agent.set_critic(critic, critic_optimizer)
            
            # Create MAPPO trainer for managing MAPPO-specific updates
            self.mappo_trainer = MAPPOTrainer(
                env=env,
                num_agents=sum(1 for t in algorithm_config if t == 'mappo'),
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=lr,
                gamma=gamma,
                device=device
            )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_counts = []
        self.success_rates = []
        
        # Per-agent metrics
        self.per_agent_rewards = [[] for _ in range(num_agents)]
        self.per_agent_success = [[] for _ in range(num_agents)]
        self.per_agent_collisions = [[] for _ in range(num_agents)]
        self.per_agent_losses = [[] for _ in range(num_agents)]
    
    def select_actions(self, observations: List[np.ndarray], deterministic: bool = False):
        """Select actions for all agents"""
        actions = []
        log_probs = []
        action_probs = []
        
        for i, agent in enumerate(self.agents):
            action, log_prob, probs = agent.select_action(observations[i], deterministic)
            actions.append(action)
            log_probs.append(log_prob)
            action_probs.append(probs)
        
        return actions, log_probs, action_probs
    
    def store_transitions(self, obs, actions, rewards, values, log_probs, done, next_obs):
        """Store transitions for all agents"""
        for i, agent in enumerate(self.agents):
            agent.store_transition(obs[i], actions[i], rewards[i], values[i], log_probs[i], done, next_obs[i])
    
    def update(self):
        """Update all agents"""
        update_info = {}
        
        for i, agent in enumerate(self.agents):
            algo = self.agent_types[i]
            
            if algo == 'mappo':
                # MAPPO agents are updated by the MAPPO trainer
                if self.mappo_trainer:
                    # Get MAPPO-specific update info
                    info = agent.update()
                    update_info[f'agent_{i}_{algo}'] = info
            else:
                # Other agents update independently
                info = agent.update()
                update_info[f'agent_{i}_{algo}'] = info
            
            # Track losses
            if 'loss' in info:
                self.per_agent_losses[i].append(info['loss'])
        
        return update_info
    
    def save(self, filepath: str):
        """Save all agents"""
        save_data = {
            'agent_types': self.agent_types,
            'agents': []
        }
        
        for i, agent in enumerate(self.agents):
            agent_path = f"{filepath}_agent_{i}_{self.agent_types[i]}.pt"
            agent.save(agent_path)
            save_data['agents'].append({
                'agent_id': i,
                'algorithm': self.agent_types[i],
                'path': agent_path
            })
        
        # Save metadata
        import json
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(save_data, f)
    
    def load(self, filepath: str):
        """Load all agents"""
        import json
        with open(f"{filepath}_metadata.json", 'r') as f:
            save_data = json.load(f)
        
        for agent_data in save_data['agents']:
            agent_id = agent_data['agent_id']
            agent_path = agent_data['path']
            self.agents[agent_id].load(agent_path)
    
    def get_algorithm_info(self):
        """Get information about algorithms used by each agent"""
        return [
            {
                'agent_id': i,
                'algorithm': self.agent_types[i]
            }
            for i in range(self.num_agents)
        ]
