import json
import os
import matplotlib.pyplot as plt

LOG_FILE = "sim_logs/history.json"

def analyze():
    if not os.path.exists(LOG_FILE):
        print("No simulation history found. Run main_test.py first!")
        return

    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    if not data:
        print("Log file is empty.")
        return

    # 1. Calculate Metrics
    total_runs = len(data)
    successful_runs = [run for run in data if run['success']]
    success_rate = (len(successful_runs) / total_runs) * 100
    
    # Calculate avg steps ONLY for successful runs (failures hit max steps usually)
    if successful_runs:
        avg_steps = sum(run['steps_taken'] for run in successful_runs) / len(successful_runs)
    else:
        avg_steps = 0

    print("="*40)
    print(f"📊 AGENT PERFORMANCE REPORT ({total_runs} Runs)")
    print("="*40)
    print(f"✅ Success Rate: {success_rate:.2f}%")
    print(f"👣 Avg Steps to Goal: {avg_steps:.2f}")
    print(f"📉 Total Failures: {total_runs - len(successful_runs)}")
    print("="*40)

    # 2. Visualization: Success Rate Over Time
    # (Simple plot to show how your agent gets better/worse)
    outcomes = [1 if run['success'] else 0 for run in data]
    
    plt.figure(figsize=(10, 5))
    plt.plot(outcomes, marker='o', linestyle='-', color='b')
    plt.title("Run Outcomes (1=Success, 0=Fail)")
    plt.xlabel("Run Number")
    plt.ylabel("Outcome")
    plt.yticks([0, 1], ["Fail", "Success"])
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    analyze()