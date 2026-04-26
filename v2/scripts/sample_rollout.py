import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from v2.env.sre_env import SreDecisionEnvV2
from v2.agents.action_space import InvestigatorAction, AnalystAction, CoordinatorAction, ExecutorAction

def main():
    env = SreDecisionEnvV2()
    print("Resetting Environment...")
    obs = env.reset(difficulty="medium")
    
    print("Initial Observations:")
    print(json.dumps(obs, indent=2))
    
    # Step 1: Agents gather info
    print("\n--- STEP 1 ---")
    actions = {
        "investigator": InvestigatorAction(action_type="query_logs", target_service="auth_service"),
        "analyst": AnalystAction(action_type="query_metrics", target_service="user_db"),
        "coordinator": CoordinatorAction(action_type="idle"),
        "executor": ExecutorAction(action_type="idle")
    }
    obs, rewards, dones, info = env.step(actions)
    print(f"Rewards: {rewards}")
    print("Investigator Logs:", obs["investigator"]["logs"])
    print("Analyst Metrics:", obs["analyst"]["metrics"])
    
    # Step 2: Agents communicate
    print("\n--- STEP 2 ---")
    actions = {
        "investigator": InvestigatorAction(action_type="share_info", message="Logs look bad on auth_service"),
        "analyst": AnalystAction(action_type="share_info", message="Metrics showing high latency on user_db"),
        "coordinator": CoordinatorAction(action_type="idle"),
        "executor": ExecutorAction(action_type="idle")
    }
    obs, rewards, dones, info = env.step(actions)
    print(f"Rewards: {rewards}")
    print("Message Bus for Coordinator:", [m["content"] for m in obs["coordinator"]["message_bus"]])

    # Step 3: Coordinator Hypothesis
    print("\n--- STEP 3 ---")
    actions = {
        "investigator": InvestigatorAction(action_type="idle"),
        "analyst": AnalystAction(action_type="idle"),
        "coordinator": CoordinatorAction(action_type="propose_hypothesis", hypothesis="db_connection_leak"),
        "executor": ExecutorAction(action_type="idle")
    }
    obs, rewards, dones, info = env.step(actions)
    print(f"Rewards: {rewards}")

    # Step 4: Executor Fix
    print("\n--- STEP 4 ---")
    actions = {
        "investigator": InvestigatorAction(action_type="idle"),
        "analyst": AnalystAction(action_type="idle"),
        "coordinator": CoordinatorAction(action_type="idle"),
        "executor": ExecutorAction(action_type="execute_fix", fix_type="restart", target_service="user_db")
    }
    obs, rewards, dones, info = env.step(actions)
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Info (True Root Cause): {info}")

if __name__ == "__main__":
    main()
