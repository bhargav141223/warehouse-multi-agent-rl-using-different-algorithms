from .warehouse_env import WarehouseEnvironment

class SimpleEnvironment(WarehouseEnvironment):
    """Simple 5x5 grid with 2 agents and no obstacles"""
    
    def __init__(self, render_mode=None):
        super().__init__(
            grid_size=5,
            num_agents=2,
            num_obstacles=0,
            dynamic_obstacles=False,
            max_steps=50,
            render_mode=render_mode
        )
