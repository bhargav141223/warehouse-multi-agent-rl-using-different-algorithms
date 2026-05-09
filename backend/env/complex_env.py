from .warehouse_env import WarehouseEnvironment
import random

class ComplexEnvironment(WarehouseEnvironment):
    """Complex 10x10 grid with 4 agents and narrow corridors"""
    
    def __init__(self, render_mode=None):
        super().__init__(
            grid_size=10,
            num_agents=4,
            num_obstacles=15,
            dynamic_obstacles=False,
            max_steps=150,
            render_mode=render_mode
        )
    
    def _initialize_positions(self):
        """Override to create narrow corridor layout"""
        all_positions = set()
        
        # Create corridor-like obstacle pattern
        self.obstacle_positions = []
        
        # Horizontal corridors
        for i in range(2, 8, 3):
            for j in range(1, 9):
                if j not in [3, 6]:  # Leave gaps
                    pos = (i, j)
                    self.obstacle_positions.append(pos)
                    all_positions.add(pos)
        
        # Vertical corridors
        for j in range(2, 8, 3):
            for i in range(1, 9):
                if i not in [3, 6]:  # Leave gaps
                    pos = (i, j)
                    self.obstacle_positions.append(pos)
                    all_positions.add(pos)
        
        # Add some random obstacles
        while len(self.obstacle_positions) < self.num_obstacles:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in all_positions:
                self.obstacle_positions.append(pos)
                all_positions.add(pos)
        
        # Initialize agent positions in corners
        self.agent_positions = []
        corner_positions = [
            (0, 0), (0, self.grid_size-1),
            (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)
        ]
        
        for i in range(min(self.num_agents, len(corner_positions))):
            pos = corner_positions[i]
            if pos not in all_positions:
                self.agent_positions.append(pos)
                all_positions.add(pos)
        
        # If not enough corners, random placement
        while len(self.agent_positions) < self.num_agents:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in all_positions:
                self.agent_positions.append(pos)
                all_positions.add(pos)
        
        # Initialize goals in opposite corners
        self.goal_positions = []
        goal_positions = [
            (self.grid_size-1, self.grid_size-1), (self.grid_size-1, 0),
            (0, self.grid_size-1), (0, 0)
        ]
        
        for i in range(min(self.num_agents, len(goal_positions))):
            pos = goal_positions[i]
            if pos not in all_positions:
                self.goal_positions.append(pos)
                all_positions.add(pos)
        
        # If not enough goal positions, random placement
        while len(self.goal_positions) < self.num_agents:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in all_positions:
                self.goal_positions.append(pos)
                all_positions.add(pos)
        
        self.dynamic_obstacle_positions = []
        self.dynamic_obstacle_velocities = []
