from .warehouse_env import WarehouseEnvironment

class DynamicEnvironment(WarehouseEnvironment):
    """8x8 grid with 4 agents and moving obstacles"""
    
    def __init__(self, render_mode=None):
        super().__init__(
            grid_size=8,
            num_agents=4,
            num_obstacles=6,
            dynamic_obstacles=True,
            max_steps=120,
            render_mode=render_mode
        )
