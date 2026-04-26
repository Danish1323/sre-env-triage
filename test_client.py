import logging
from openenv.core import EnvClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models import SreDecisionAction
from server.sre_decision_env_environment import SreDecisionEnvironment

logging.basicConfig(level=logging.DEBUG)

def test_local_env():
    # Directly wrapping the server to test client compatibility without needing Docker
    print("Initializing environment...")
    env = SreDecisionEnvironment()
    
    print("\n--- RESET ---")
    obs = env.reset()
    print("Initial Obs:", obs.model_dump())
    
    actions_to_take = [
        "inspect_logs",
        "inspect_metrics",
        "restart_service"
    ]
    
    for action_name in actions_to_take:
        print(f"\n--- STEP: {action_name} ---")
        action = SreDecisionAction(action_name=action_name, rationale="Testing")
        # Direct call simulating what the client would do over REST/WS
        obs = env.step(action)
        print("Observation:", obs.model_dump())
        print(f"Reward: {obs.metadata.get('total_reward_so_far')}")
        print(f"Done: {obs.done}")

if __name__ == "__main__":
    test_local_env()
