import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class GridEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent moves in a N x N grid layout.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, size=6):
        super(GridEnv, self).__init__()
        self.size = size
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # 0=Empty, 1=Agent, 2=Goal, 3=Wall
        self.observation_space = spaces.Box(low=0, high=3, shape=(size, size), dtype=np.int32)

        # Initialize fixed agent/goal
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        
        # Will hold the map layout (walls vs empty)
        self.static_grid = None 

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed)
        
        # 1. Create an empty grid
        self.static_grid = np.zeros((self.size, self.size), dtype=np.int32)
        
        # 2. Add Random Walls (approx 20% of the grid)
        num_obstacles = int(self.size * self.size * 0.2)
        for _ in range(num_obstacles):
            # Pick a random spot
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            
            # Don't overwrite start or goal!
            if (r == 0 and c == 0) or (r == self.size-1 and c == self.size-1):
                continue
            
            self.static_grid[r, c] = 3 # 3 represents a Wall

        self.agent_pos = np.array([0, 0])
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Helper to generate the current grid state.
        """
        # Start with the static layout (walls + empty)
        obs = self.static_grid.copy()
        
        # Overlay Agent and Goal
        # Note: We overwrite walls if agent is "on top" (though logic prevents this)
        obs[tuple(self.goal_pos)] = 2
        obs[tuple(self.agent_pos)] = 1
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        direction = {
            0: np.array([-1, 0]), # Up
            1: np.array([1, 0]),  # Down
            2: np.array([0, -1]), # Left
            3: np.array([0, 1])   # Right
        }

        current_pos = self.agent_pos
        step_vector = direction.get(action, np.array([0, 0]))
        new_pos = current_pos + step_vector

        # 1. Check Bounds via Clipping
        new_pos = np.clip(new_pos, 0, self.size - 1)

        # 2. Check Collision (Is the new spot a Wall?)
        # We check the static_grid for value 3
        if self.static_grid[tuple(new_pos)] == 3:
            # Hit a wall! Agent stays in current_pos
            new_pos = current_pos 
        
        self.agent_pos = new_pos

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10.0 if terminated else -0.1
        
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        """
        Calculates the grid and draws it to the current plot.
        Does NOT clear or pause (we do that in main loop now).
        """
        grid = self._get_obs()
        
        # Create a custom color map for clarity
        # 0: White (Empty)
        # 1: Blue (Agent)
        # 2: Green (Goal)
        # 3: Black (Wall)
        cmap = colors.ListedColormap(['white', 'blue', 'green', 'black'])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(grid, cmap=cmap, norm=norm)
        
        # Draw grid lines for better visibility
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(-.5, self.size, 1)); 
        plt.yticks(np.arange(-.5, self.size, 1));