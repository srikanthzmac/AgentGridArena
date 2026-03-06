from grid_env import GridEnv
import matplotlib.pyplot as plt
import time

def main():
    # 1. Initialize Environment
    print("Initializing AgentGrid Arena...")
    env = GridEnv(size=6) # Smaller grid for easier visualization
    obs, info = env.reset()

    # Setup rendering
    plt.ion() # Interactive mode on
    
    print("Starting Simulation Loop...")
    for _ in range(20): # Run for 20 steps
        env.render()
        
        # 2. Random Agent Policy
        action = env.action_space.sample() # Pick a random action (0-3)
        
        # 3. Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action: {action} | Reward: {reward} | Done: {terminated}")
        
        if terminated:
            print("Goal Reached! 🏆")
            break
            
    print("Simulation finished.")
    plt.ioff()
    plt.show() # Keep the last frame open

if __name__ == "__main__":
    main()