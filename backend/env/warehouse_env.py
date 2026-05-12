import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import random

class WarehouseEnvironment(gym.Env):
    """Base Multi-Agent Warehouse Navigation Environment"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 2,
        num_obstacles: int = 0,
        dynamic_obstacles: bool = False,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = spaces.Discrete(5)
        
        # Observation space: fixed 8-dimensional feature vector
        # [agent_x, agent_y, goal_x, goal_y, nearest_obstacle_dist, nearest_agent_dist, collision_flag, remaining_steps]
        self.OBS_DIM = 8
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(self.OBS_DIM,), dtype=np.float32
        )
        
        # Agent and goal positions
        self.agent_positions: List[Tuple[int, int]] = []
        self.goal_positions: List[Tuple[int, int]] = []
        self.obstacle_positions: List[Tuple[int, int]] = []
        self.dynamic_obstacle_positions: List[Tuple[int, int]] = []
        self.dynamic_obstacle_velocities: List[Tuple[int, int]] = []
        
        self.steps = 0
        self.episode_rewards: List[float] = []
        self.collisions = 0
        self.goals_reached = [False] * num_agents
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.steps = 0
        self.collisions = 0
        self.episode_rewards = []
        self.goals_reached = [False] * self.num_agents
        
        # Initialize positions
        self._initialize_positions()
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def _initialize_positions(self):
        """Initialize agent, goal, and obstacle positions"""
        all_positions = set()
        
        # Initialize agent positions
        self.agent_positions = []
        for i in range(self.num_agents):
            while True:
                pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if pos not in all_positions:
                    self.agent_positions.append(pos)
                    all_positions.add(pos)
                    break
        
        # Initialize goal positions
        self.goal_positions = []
        for i in range(self.num_agents):
            while True:
                pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if pos not in all_positions:
                    self.goal_positions.append(pos)
                    all_positions.add(pos)
                    break
        
        # Initialize static obstacles
        self.obstacle_positions = []
        for i in range(self.num_obstacles):
            while True:
                pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if pos not in all_positions:
                    self.obstacle_positions.append(pos)
                    all_positions.add(pos)
                    break
        
        # Initialize dynamic obstacles
        self.dynamic_obstacle_positions = []
        self.dynamic_obstacle_velocities = []
        if self.dynamic_obstacles:
            num_dynamic = max(1, self.num_obstacles // 2)
            for i in range(num_dynamic):
                while True:
                    pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                    if pos not in all_positions:
                        self.dynamic_obstacle_positions.append(pos)
                        # Random velocity: -1, 0, or 1 in each direction
                        velocity = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
                        self.dynamic_obstacle_velocities.append(velocity)
                        all_positions.add(pos)
                        break
    
    def step(self, actions: List[int]):
        """Execute one timestep for all agents"""
        self.steps += 1
        rewards = []
        dones = []
        
        # Move dynamic obstacles first
        if self.dynamic_obstacles:
            self._move_dynamic_obstacles()
        
        # Process each agent's action
        new_positions = []
        for i, action in enumerate(actions):
            if self.goals_reached[i]:
                new_positions.append(self.agent_positions[i])
                continue
                
            new_pos = self._get_new_position(self.agent_positions[i], action)
            new_positions.append(new_pos)
        
        # Check for collisions and update positions
        for i, new_pos in enumerate(new_positions):
            reward = 1.0  # Base reward for staying alive
            
            # Check wall collisions
            if self._is_out_of_bounds(new_pos):
                reward -= 5.0
                new_pos = self.agent_positions[i]  # Stay in place
            
            # Check obstacle collisions
            elif new_pos in self.obstacle_positions:
                reward -= 5.0
                new_pos = self.agent_positions[i]
                self.collisions += 1
            
            # Check dynamic obstacle collisions
            elif new_pos in self.dynamic_obstacle_positions:
                reward -= 5.0
                new_pos = self.agent_positions[i]
                self.collisions += 1
            
            # Check agent-agent collisions
            elif new_positions.count(new_pos) > 1 and not self.goals_reached[i]:
                reward -= 5.0
                new_pos = self.agent_positions[i]
                self.collisions += 1
            
            # Calculate distance-based reward
            old_dist = self._manhattan_distance(self.agent_positions[i], self.goal_positions[i])
            new_dist = self._manhattan_distance(new_pos, self.goal_positions[i])
            
            if new_dist < old_dist:
                reward += 2.0  # Moving closer to goal
            elif new_dist > old_dist:
                reward -= 1.0  # Moving away from goal
            
            # Check goal reached - BIGGER REWARD to encourage success
            if new_pos == self.goal_positions[i] and not self.goals_reached[i]:
                reward += 50.0  # Increased from 10 to 50 for better learning
                self.goals_reached[i] = True
                
            # Bonus for being at goal (stay incentive)
            if self.goals_reached[i] and new_pos == self.goal_positions[i]:
                reward += 5.0  # Stay at goal bonus
            
            self.agent_positions[i] = new_pos
            rewards.append(reward)
            self.episode_rewards.append(reward)
        
        # Check if episode is done
        all_goals_reached = all(self.goals_reached)
        truncated = self.steps >= self.max_steps
        terminated = all_goals_reached
        
        # BIG BONUS for completing all goals - crucial for high success rate
        if all_goals_reached and not any(self.goals_reached):  # Just completed
            for i in range(len(rewards)):
                rewards[i] += 100.0  # Team completion bonus
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _move_dynamic_obstacles(self):
        """Move dynamic obstacles"""
        for i in range(len(self.dynamic_obstacle_positions)):
            current_pos = self.dynamic_obstacle_positions[i]
            velocity = self.dynamic_obstacle_velocities[i]
            
            new_pos = (
                current_pos[0] + velocity[0],
                current_pos[1] + velocity[1]
            )
            
            # Bounce off walls
            if new_pos[0] < 0 or new_pos[0] >= self.grid_size:
                velocity = (-velocity[0], velocity[1])
                new_pos = (current_pos[0], current_pos[1] + velocity[1])
            
            if new_pos[1] < 0 or new_pos[1] >= self.grid_size:
                velocity = (velocity[0], -velocity[1])
                new_pos = (current_pos[0] + velocity[0], current_pos[1])
            
            # Update position and velocity
            self.dynamic_obstacle_positions[i] = new_pos
            self.dynamic_obstacle_velocities[i] = velocity
    
    def _get_new_position(self, current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculate new position based on action"""
        x, y = current_pos
        
        if action == 0:  # Up
            return (x - 1, y)
        elif action == 1:  # Down
            return (x + 1, y)
        elif action == 2:  # Left
            return (x, y - 1)
        elif action == 3:  # Right
            return (x, y + 1)
        else:  # Stay
            return (x, y)
    
    def _is_out_of_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is out of grid bounds"""
        return pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_agent_features(self, agent_id: int) -> np.ndarray:
        """Extract fixed 8-dimensional feature vector for an agent"""
        agent_pos = self.agent_positions[agent_id]
        goal_pos = self.goal_positions[agent_id]
        
        # 1-2: Agent position (normalized)
        agent_x = agent_pos[0] / self.grid_size
        agent_y = agent_pos[1] / self.grid_size
        
        # 3-4: Goal position (normalized)
        goal_x = goal_pos[0] / self.grid_size
        goal_y = goal_pos[1] / self.grid_size
        
        # 5: Nearest obstacle distance (normalized)
        nearest_obstacle_dist = self._get_nearest_obstacle_distance(agent_pos)
        
        # 6: Nearest agent distance (normalized)
        nearest_agent_dist = self._get_nearest_agent_distance(agent_id, agent_pos)
        
        # 7: Collision flag (binary)
        collision_flag = 1.0 if self.collisions > 0 else 0.0
        
        # 8: Remaining steps (normalized)
        remaining_steps = (self.max_steps - self.steps) / self.max_steps
        
        return np.array([agent_x, agent_y, goal_x, goal_y, 
                        nearest_obstacle_dist, nearest_agent_dist, 
                        collision_flag, remaining_steps], dtype=np.float32)
    
    def _get_nearest_obstacle_distance(self, pos: Tuple[int, int]) -> float:
        """Get normalized distance to nearest obstacle"""
        all_obstacles = self.obstacle_positions + self.dynamic_obstacle_positions
        if not all_obstacles:
            return 1.0  # Max distance if no obstacles
        
        min_dist = min(self._manhattan_distance(pos, obs_pos) 
                      for obs_pos in all_obstacles)
        # Normalize by grid size (max possible distance is 2*grid_size)
        return min(min_dist / (2 * self.grid_size), 1.0)
    
    def _get_nearest_agent_distance(self, agent_id: int, pos: Tuple[int, int]) -> float:
        """Get normalized distance to nearest other agent"""
        other_agents = [self.agent_positions[i] for i in range(self.num_agents) if i != agent_id]
        if not other_agents:
            return 1.0  # Max distance if no other agents
        
        min_dist = min(self._manhattan_distance(pos, other_pos) 
                      for other_pos in other_agents)
        # Normalize by grid size
        return min(min_dist / (2 * self.grid_size), 1.0)
    
    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents as fixed feature vectors"""
        observations = []
        
        for agent_id in range(self.num_agents):
            obs = self._get_agent_features(agent_id)
            observations.append(obs)
        
        return observations
    
    def _get_info(self) -> Dict:
        """Get additional information with success tracking"""
        # Calculate success rate (proportion of agents that reached goals)
        # Clamp to [0, 1] to ensure it doesn't exceed 100%
        success_rate = min(max(sum(self.goals_reached) / self.num_agents if self.num_agents > 0 else 0, 0), 1.0)
        
        # Full success = all agents reached their goals
        full_success = all(self.goals_reached)
        
        return {
            'steps': self.steps,
            'collisions': self.collisions,
            'goals_reached': sum(self.goals_reached),
            'total_goals': self.num_agents,
            'episode_reward': sum(self.episode_rewards),
            'success_rate': success_rate,  # Explicit success rate (0-1)
            'full_success': full_success,   # All goals reached
            'agent_positions': self.agent_positions,
            'goal_positions': self.goal_positions,
            'obstacle_positions': self.obstacle_positions,
            'dynamic_obstacle_positions': self.dynamic_obstacle_positions,
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        elif self.render_mode == 'human':
            print(self._render_text())
    
    def _render_frame(self):
        """Render frame as RGB array"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Draw obstacles
        for pos in self.obstacle_positions:
            rect = patches.Rectangle((pos[1], self.grid_size-pos[0]-1), 1, 1, 
                                    linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)
        
        # Draw dynamic obstacles
        for pos in self.dynamic_obstacle_positions:
            rect = patches.Rectangle((pos[1], self.grid_size-pos[0]-1), 1, 1, 
                                    linewidth=1, edgecolor='black', facecolor='orange')
            ax.add_patch(rect)
        
        # Draw goals
        for i, pos in enumerate(self.goal_positions):
            circle = patches.Circle((pos[1]+0.5, self.grid_size-pos[0]-0.5), 0.3, 
                                   color='green', alpha=0.5)
            ax.add_patch(circle)
            ax.text(pos[1]+0.5, self.grid_size-pos[0]-0.5, f'G{i}', ha='center', va='center')
        
        # Draw agents
        for i, pos in enumerate(self.agent_positions):
            color = 'blue' if not self.goals_reached[i] else 'purple'
            circle = patches.Circle((pos[1]+0.5, self.grid_size-pos[0]-0.5), 0.4, 
                                   color=color)
            ax.add_patch(circle)
            ax.text(pos[1]+0.5, self.grid_size-pos[0]-0.5, str(i), 
                   ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_title(f'Step: {self.steps}, Collisions: {self.collisions}')
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
    
    def _render_text(self) -> str:
        """Render environment as text"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place obstacles
        for pos in self.obstacle_positions:
            grid[pos[0]][pos[1]] = '#'
        
        # Place dynamic obstacles
        for pos in self.dynamic_obstacle_positions:
            grid[pos[0]][pos[1]] = 'D'
        
        # Place goals
        for i, pos in enumerate(self.goal_positions):
            grid[pos[0]][pos[1]] = f'G{i}'
        
        # Place agents
        for i, pos in enumerate(self.agent_positions):
            if self.goals_reached[i]:
                grid[pos[0]][pos[1]] = f'A{i}'
            else:
                grid[pos[0]][pos[1]] = f'{i}'
        
        return '\n'.join([' '.join(row) for row in grid])
