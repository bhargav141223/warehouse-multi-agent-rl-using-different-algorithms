from .warehouse_env import WarehouseEnvironment

class MediumEnvironment(WarehouseEnvironment):
    """Medium 8x8 grid with 3 agents and static obstacles"""
    
    def __init__(self, render_mode=None):
        super().__init__(
            grid_size=8,
            num_agents=3,
            num_obstacles=8,
            dynamic_obstacles=False,
            max_steps=100,
            render_mode=render_mode
        )
