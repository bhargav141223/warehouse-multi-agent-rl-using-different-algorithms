import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import copy

from .networks import ActorNetwork, CriticNetwork

class MAPPOAgent:
    """Multi-Agent PPO Agent with Centralized Critic"""
    
    def __init__(
        self,
        obs_dim,
        action_dim: int,
        num_agents: int,
        agent_id: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Actor network (each agent has its own or shared)
        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Reference to centralized critic (shared across agents)
        self.critic = None
        self.critic_optimizer = None
        
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
        
    def set_critic(self, critic: CriticNetwork, critic_optimizer):
        """Set the centralized critic"""
        self.critic = critic
        self.critic_optimizer = critic_optimizer
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """Select action for this agent"""
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        with torch.no_grad():
            action, action_probs = self.actor.get_action(obs_tensor, deterministic)
            
            # Calculate log probability
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)
        
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
    
    def clear_buffer(self):
        """Clear the experience buffer"""
        for key in self.buffer:
            self.buffer[key] = []
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, batch_size: int = 64, update_epochs: int = 4):
        """Update policy using PPO"""
        if len(self.buffer['observations']) == 0:
            return {}
        
        # Prepare data
        obs = np.array(self.buffer['observations'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        old_values = np.array(self.buffer['values'])
        old_log_probs = np.array(self.buffer['log_probs'])
        dones = np.array(self.buffer['dones'])
        next_obs = np.array(self.buffer['next_observations'])
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Compute advantages and returns using GAE
        # Use the last stored value as next_value (it's already from the centralized critic)
        next_value = old_values[-1] if len(old_values) > 0 else 0
        
        advantages, returns = self.compute_gae(
            rewards.tolist(), old_values.tolist(), dones.tolist(), next_value
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        dataset_size = len(obs)
        indices = np.arange(dataset_size)
        
        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get current action probabilities
                action_probs, logits = self.actor(batch_obs)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio and clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_entropy += entropy.item()
        
        # Update critic (using all agents' data)
        # This should be done once for all agents, not per agent
        
        self.clear_buffer()
        
        return {
            'actor_loss': total_actor_loss / (update_epochs * (dataset_size // batch_size + 1)),
            'entropy': total_entropy / (update_epochs * (dataset_size // batch_size + 1))
        }


class MAPPOTrainer:
    """Trainer for Multi-Agent PPO"""
    
    def __init__(
        self,
        env,
        num_agents: int,
        obs_dim,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        shared_actor: bool = False
    ):
        self.env = env
        self.num_agents = num_agents
        self.device = device
        self.shared_actor = shared_actor
        
        # Create agents
        self.agents: List[MAPPOAgent] = []
        
        if shared_actor:
            # Shared actor network
            shared_actor_net = ActorNetwork(obs_dim, action_dim).to(device)
            shared_actor_opt = optim.Adam(shared_actor_net.parameters(), lr=lr)
        
        for i in range(num_agents):
            agent = MAPPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                agent_id=i,
                lr=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
                max_grad_norm=max_grad_norm,
                device=device
            )
            
            if shared_actor:
                agent.actor = shared_actor_net
                agent.actor_optimizer = shared_actor_opt
            
            self.agents.append(agent)
        
        # Centralized critic
        self.critic = CriticNetwork(obs_dim, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Set critic for all agents
        for agent in self.agents:
            agent.set_critic(self.critic, self.critic_optimizer)
        
        # Critic buffer
        self.critic_buffer = {
            'observations': [],
            'values': [],
            'returns': []
        }
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.collision_counts = []
        
        # Per-agent tracking for detailed analysis
        self.per_agent_rewards = [[] for _ in range(num_agents)]
        self.per_agent_success = [[] for _ in range(num_agents)]
        self.per_agent_collisions = [[] for _ in range(num_agents)]
    
    def collect_rollout(self, max_steps: int = 2048):
        """Collect a rollout from all agents"""
        obs, info = self.env.reset()
        
        rollout_data = {
            'observations': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'values': [[] for _ in range(self.num_agents)],
            'log_probs': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
            'next_observations': [[] for _ in range(self.num_agents)],
            'joint_observations': []
        }
        
        episode_reward = 0
        episode_collisions = 0
        steps = 0
        
        for step in range(max_steps):
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []
            
            # Get joint observation for critic
            obs_tensors = [torch.FloatTensor(o).unsqueeze(0).to(self.device) for o in obs]
            with torch.no_grad():
                joint_value = self.critic(obs_tensors).cpu().numpy()[0, 0]
            
            for i, agent in enumerate(self.agents):
                action, log_prob, action_probs = agent.select_action(obs[i])
                actions.append(action[0])
                log_probs.append(log_prob[0])
                values.append(joint_value)
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # Store transitions
            joint_obs = np.concatenate([o.flatten() for o in obs])
            next_joint_obs = np.concatenate([o.flatten() for o in next_obs])
            
            rollout_data['joint_observations'].append(joint_obs)
            
            for i in range(self.num_agents):
                rollout_data['observations'][i].append(obs[i])
                rollout_data['actions'][i].append(actions[i])
                rollout_data['rewards'][i].append(rewards[i])
                rollout_data['values'][i].append(values[i])
                rollout_data['log_probs'][i].append(log_probs[i])
                rollout_data['dones'][i].append(done)
                rollout_data['next_observations'][i].append(next_obs[i])
                
                self.agents[i].store_transition(
                    obs[i], actions[i], rewards[i], values[i],
                    log_probs[i], done, next_obs[i]
                )
            
            episode_reward += sum(rewards)
            episode_collisions += info.get('collisions', 0)
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        return {
            'episode_reward': episode_reward,
            'episode_length': steps,
            'collisions': episode_collisions,
            'success_rate': info.get('goals_reached', 0) / self.num_agents,
            'info': info
        }
    
    def train(self, total_episodes: int = 1000, steps_per_update: int = 2048, 
              callback=None):
        """Train MAPPO agents"""
        
        for episode in range(total_episodes):
            # Collect rollout
            rollout_info = self.collect_rollout(steps_per_update)
            
            self.episode_rewards.append(rollout_info['episode_reward'])
            self.episode_lengths.append(rollout_info['episode_length'])
            self.collision_counts.append(rollout_info['collisions'])
            self.success_rates.append(rollout_info['success_rate'])
            
            # Update agents
            update_info = {}
            for i, agent in enumerate(self.agents):
                agent_info = agent.update()
                update_info[f'agent_{i}_actor_loss'] = agent_info.get('actor_loss', 0)
                update_info[f'agent_{i}_entropy'] = agent_info.get('entropy', 0)
            
            # Logging
            if callback and episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_success = np.mean(self.success_rates[-10:])
                avg_collisions = np.mean(self.collision_counts[-10:])
                
                callback({
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'avg_success_rate': avg_success,
                    'avg_collisions': avg_collisions,
                    **update_info,
                    **rollout_info
                })
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward={rollout_info['episode_reward']:.2f}, "
                      f"Success={rollout_info['success_rate']:.2f}, "
                      f"Collisions={rollout_info['collisions']}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'collision_counts': self.collision_counts
        }
    
    def save(self, path: str):
        """Save models"""
        checkpoint = {
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'agents': []
        }
        
        for agent in self.agents:
            checkpoint['agents'].append({
                'actor_state_dict': agent.actor.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict()
            })
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load models"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint['agents'][i]['actor_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['agents'][i]['actor_optimizer_state_dict'])
