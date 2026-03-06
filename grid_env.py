import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent moves in a N x N grid layout.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, size=10):
        super(GridEnv, self).__init__()
        self.size = size
        self.window_size = 512  # The size of the PyGame window (if we used pygame)

        # Action Space: 4 discrete actions (0: Up, 1: Down, 2: Left, 3: Right)
        self.action_space = spaces.Discrete(4)

        # Observation Space: The grid itself. 
        # 0 = Empty, 1 = Agent, 2 = Goal, 3 = Obstacle
        self.observation_space = spaces.Box(low=0, high=3, shape=(size, size), dtype=np.int32)

        self.agent_pos = None
        self.goal_pos = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed)

        # Initialize the grid as empty
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)

        # Set fixed positions for now (Top-Left start, Bottom-Right goal)
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size - 1, self.size - 1])

        # Return observation and info
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Helper to generate the current grid state.
        """
        obs = np.zeros((self.size, self.size), dtype=np.int32)
        obs[tuple(self.goal_pos)] = 2
        obs[tuple(self.agent_pos)] = 1
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Map actions to direction vectors
        # 0: Up, 1: Down, 2: Left, 3: Right
        direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }

        # Calculate tentative new position
        current_pos = self.agent_pos
        step_vector = direction.get(action, np.array([0, 0]))
        new_pos = current_pos + step_vector

        # Check Bounds (Stay inside the grid 0 to size-1)
        new_pos = np.clip(new_pos, 0, self.size - 1)

        # Move the agent
        self.agent_pos = new_pos

        # Check if goal reached
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Reward function
        reward = 1.0 if terminated else -0.1  # Small penalty for each step to encourage speed

        truncated = False # Typically used for time limits
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """
        Simple Matplotlib rendering
        """
        grid_viz = self._get_obs()
        plt.imshow(grid_viz, cmap='viridis') # viridis allows us to see different values clearly
        plt.title(f"Agent Config: Pos {self.agent_pos}")
        plt.pause(0.1) # Pause to let the plot update
        plt.clf() # Clear for next frame