import json
import os
from datetime import datetime
from grid_env import GridEnv
import matplotlib.pyplot as plt
import numpy as np

# Configuration
LOG_DIR = "sim_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def save_run_data(run_data):
    """Appends data to history.json"""
    history_file = os.path.join(LOG_DIR, "history.json")
    
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    
    history.append(run_data)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Data saved to {history_file}")

def save_snapshot(env, run_id, success, step_count):
    """
     renders the final state of the grid to a PNG file.
    """
    plt.figure(figsize=(6, 6))
    env.render() # Draw the grid using the Env's render method
    
    status = "SUCCESS" if success else "FAILED"
    plt.title(f"Run: {run_id} | {status} | Steps: {step_count}")
    
    image_path = os.path.join(LOG_DIR, f"run_{run_id}.png")
    plt.savefig(image_path)
    plt.close() # Close memory immediately
    print(f"Snapshot saved to {image_path}")

def main():
    # 1. Setup Environment
    env = GridEnv(size=6)
    obs, info = env.reset()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_taken = [env.agent_pos.tolist()] 
    step_count = 0
    max_steps = 50 # Increased slightly to give random agent a chance
    success = False

    print(f"Starting Fast Run ID: {run_id}")
    
    # 2. FAST Simulation Loop (No Rendering here)
    for _ in range(max_steps):
        # Action: Random (Place for AI logic later)
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        path_taken.append(env.agent_pos.tolist())
        step_count += 1
        
        if terminated:
            print(f"Goal Reached in {step_count} steps! 🏆")
            success = True
            break

    # 3. Post-Run Processing
    # Now we generate the visual artifact effectively
    save_snapshot(env, run_id, success, step_count)

    # 4. Save Numerical Data
    run_stats = {
        "run_id": run_id,
        "timestamp": str(datetime.now()),
        "grid_size": env.size,
        "success": success,
        "steps_taken": step_count,
        "path_length": len(path_taken),
        "obstacle_count": int(np.sum(env.static_grid == 3))
    }
    
    save_run_data(run_stats)

if __name__ == "__main__":
    main()