"""
Phase 3 Evaluation Pipeline
Runs episodes using specific agents, logs data, computes metrics, and generates plots.
"""

import json
import logging
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from server.sre_decision_env_environment import SreDecisionEnvironment, MAX_STEPS
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleBasedAgent
from agents.trained_agent import TrainedAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def run_episode(env: SreDecisionEnvironment, agent: Any, max_steps: int = MAX_STEPS) -> Dict:
    obs = env.reset()
    episode_id = env.state.episode_id
    
    episode_log = []
    total_reward = 0.0
    actions = []
    correct_actions = 0
    wrong_actions = 0

    for t in range(max_steps):
        action = agent.act(obs)
        
        # We step the environment
        next_obs = env.step(action)
        reward = next_obs.reward or 0.0
        done = next_obs.done or next_obs.incident_resolved
        
        # SRE logic context
        # Correct actions yield high reward, wrong actions yield low reward
        is_correct = reward >= 0.75
        is_wrong = reward <= 0.35
        
        step_data = {
            "episode_id": episode_id,
            "step": t,
            "obs": obs.model_dump(),
            "action": action.action_name,
            "reward": reward,
            "done": done,
            "resolved": next_obs.incident_resolved,
            "correct_action": is_correct if is_correct else (False if is_wrong else None)
        }
        
        episode_log.append(step_data)
        total_reward += reward
        actions.append(action.action_name)
        
        if is_correct:
            correct_actions += 1
        elif is_wrong:
            wrong_actions += 1
            
        obs = next_obs
        
        if done:
            break
            
    return {
        "episode_id": episode_id,
        "steps": episode_log,
        "total_reward": total_reward,
        "num_steps": len(episode_log),
        "success": episode_log[-1]["resolved"] if episode_log else False,
        "correct_actions": correct_actions,
        "wrong_actions": wrong_actions,
        "actions": actions
    }


def run_evaluation(agent_type: str, num_episodes: int = 10):
    logger.info(f"Starting evaluation for agent: {agent_type} over {num_episodes} episodes.")
    
    env = SreDecisionEnvironment()
    
    if agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "rule":
        agent = RuleBasedAgent()
    elif agent_type == "trained":
        agent = TrainedAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    all_episodes_data = []
    all_steps_data = []
    
    successes = 0
    total_rewards = []
    steps_to_success = []
    total_correct = 0
    total_wrong = 0
    total_actions = 0
    
    for i in range(num_episodes):
        ep_result = run_episode(env, agent)
        
        # Aggregate step logs
        all_steps_data.extend(ep_result["steps"])
        
        # Calculate episode level metrics
        success = ep_result["success"]
        ep_steps = ep_result["num_steps"]
        total_actions += ep_steps
        total_correct += ep_result["correct_actions"]
        total_wrong += ep_result["wrong_actions"]
        
        if success:
            successes += 1
            steps_to_success.append(ep_steps)
            
        total_rewards.append(ep_result["total_reward"])
        
        efficiency = ep_result["correct_actions"] / ep_steps if ep_steps > 0 else 0.0
        
        ep_data = {
            "episode_id": ep_result["episode_id"],
            "agent_type": agent_type,
            "total_reward": ep_result["total_reward"],
            "steps": ep_steps,
            "success": success,
            "efficiency": efficiency
        }
        all_episodes_data.append(ep_data)
        
        if (i+1) % 5 == 0:
            logger.info(f"Completed {i+1}/{num_episodes} episodes.")
            
    # Calculate overall metrics
    metrics = {
        "success_rate": successes / num_episodes if num_episodes > 0 else 0,
        "avg_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        "MTTR": sum(steps_to_success) / len(steps_to_success) if steps_to_success else 0,
        "efficiency": total_correct / total_actions if total_actions > 0 else 0,
        "wrong_rate": total_wrong / total_actions if total_actions > 0 else 0
    }
    
    # 4. DATA STORAGE
    # 4.1 Step-Level Dataset
    steps_path = DATA_DIR / f"{agent_type}_steps.json"
    with open(steps_path, "w") as f:
        json.dump(all_steps_data, f, indent=2)
        
    # 4.2 Episode-Level Dataset
    episodes_path = DATA_DIR / f"{agent_type}_episodes.csv"
    with open(episodes_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "agent_type", "total_reward", "steps", "success", "efficiency"])
        writer.writeheader()
        writer.writerows(all_episodes_data)
        
    # 7. FINAL REPORT
    report_path = REPORTS_DIR / f"{agent_type}_summary.json"
    with open(report_path, "w") as f:
        json.dump({"agent": agent_type, "metrics": metrics}, f, indent=2)
        
    # 9. HUGGING FACE DATASET EXPORT
    txt_path = DATA_DIR / f"{agent_type}_hf_dataset.txt"
    with open(txt_path, "w") as f:
        for step in all_steps_data:
            # Constraint: keep only reward > 0.75 or correct_action == True
            is_correct = step.get("correct_action", False)
            reward = step.get("reward", 0.0)
            
            if not (reward > 0.7 or is_correct):
                continue

            logs = step["obs"].get("logs", {})
            observer = step["obs"].get("observer", {})
            
            # Format logs inline
            logs_list = []
            for k, v in logs.items():
                key = k.replace("log_anomaly_score", "anomaly_score").replace("five_xx_error_rate", "5xx_error_rate")
                val = f"{v:.2f}" if isinstance(v, float) else str(v)
                logs_list.append(f"{key}={val}")
            logs_str = ", ".join(logs_list)
            
            # Format observer inline
            obs_list = []
            for k, v in observer.items():
                key = k.replace("cpu_usage", "cpu").replace("memory_usage", "memory").replace("server_b_health", "health").replace("db_connections", "db_conn")
                val = f"{v:.2f}" if isinstance(v, float) else str(v)
                obs_list.append(f"{key}={val}")
            obs_str = ", ".join(obs_list)
            
            # Pure text block using Instruction / Response format
            state_description = f"Logs: {logs_str}\\nObserver: {obs_str}"
            text_block = f"### Instruction:\\n{state_description}\\n\\n### Response:\\n{step['action']}"
            
            f.write(text_block + "\\n\\n---\\n\\n")
            
    logger.info(f"Evaluation complete for {agent_type}. Metrics: {metrics}")
    
    # Run plot generation across all agents present in data dir
    generate_comparison_plots()


# 5. GRAPH GENERATION & 6. COMPARISON SYSTEM
def generate_comparison_plots():
    logger.info("Generating comparison plots...")
    import glob
    
    # Find all summary files
    summary_files = glob.glob(str(REPORTS_DIR / "*_summary.json"))
    if not summary_files:
        return
        
    agents = []
    success_rates = []
    avg_rewards = []
    mttrs = []
    efficiencies = []
    wrong_rates = []
    
    for sf in summary_files:
        with open(sf, "r") as f:
            data = json.load(f)
            ag = data["agent"]
            m = data["metrics"]
            
            agents.append(ag)
            success_rates.append(m["success_rate"])
            avg_rewards.append(m["avg_reward"])
            mttrs.append(m["MTTR"])
            efficiencies.append(m["efficiency"])
            wrong_rates.append(m["wrong_rate"])
            
    # 1. reward_vs_episode
    plt.figure()
    plt.bar(agents, avg_rewards, color=["#3498db", "#2ecc71", "#e74c3c"])
    plt.title("Average Reward per Agent")
    plt.ylabel("Reward")
    plt.savefig(PLOTS_DIR / "reward_vs_episode.png")
    plt.close()
    
    # 2. success_rate_vs_episode
    plt.figure()
    plt.bar(agents, success_rates, color=["#3498db", "#2ecc71", "#e74c3c"])
    plt.title("Success Rate per Agent")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.0)
    plt.savefig(PLOTS_DIR / "success_rate_vs_episode.png")
    plt.close()
    
    # 3. MTTR_vs_episode
    plt.figure()
    plt.bar(agents, mttrs, color=["#3498db", "#2ecc71", "#e74c3c"])
    plt.title("Mean Time To Resolution (Steps)")
    plt.ylabel("Steps")
    plt.savefig(PLOTS_DIR / "MTTR_vs_episode.png")
    plt.close()
    
    # 4. efficiency_vs_episode
    plt.figure()
    plt.bar(agents, efficiencies, color=["#3498db", "#2ecc71", "#e74c3c"])
    plt.title("Efficiency (Correct Actions / Total Actions)")
    plt.ylabel("Efficiency Rate")
    plt.ylim(0, 1.0)
    plt.savefig(PLOTS_DIR / "efficiency_vs_episode.png")
    plt.close()
    
    # 5. wrong_action_rate
    plt.figure()
    plt.bar(agents, wrong_rates, color=["#3498db", "#2ecc71", "#e74c3c"])
    plt.title("Wrong Action Rate")
    plt.ylabel("Rate")
    plt.ylim(0, 1.0)
    plt.savefig(PLOTS_DIR / "wrong_action_rate.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "rule", "trained"], required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    args = parser.parse_args()
    
    run_evaluation(args.agent, args.num_episodes)
